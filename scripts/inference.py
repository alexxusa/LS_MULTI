# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from collections import defaultdict

from omegaconf import OmegaConf
import torch
from accelerate import dispatch_model
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper


_GIB = 1024**3


def _build_two_gpu_device_map(unet):
    """Map whole UNet stages to cuda:0/cuda:1; deliberately avoid CPU offload.

    Keep the input-side modules and the first down block on GPU 0. Put the
    remaining down blocks and the middle block on GPU 1, then send the first
    half of the up blocks to GPU 1 and the final up blocks/output head to GPU 0.
    The model's block boundaries are important: Accelerate hooks move inputs,
    including skip tensors, between the devices at these boundaries.
    """
    children = dict(unet.named_children())
    required = ("down_blocks", "mid_block", "up_blocks")
    missing = [name for name in required if name not in children]
    if missing:
        raise RuntimeError(
            "Unexpected LatentSync UNet structure; required module(s) missing: "
            + ", ".join(missing)
            + ". Refusing to guess a device map."
        )

    down_count = len(unet.down_blocks)
    up_count = len(unet.up_blocks)
    if down_count < 2 or up_count < 2:
        raise RuntimeError(
            "This multi-GPU mapping expects at least two down blocks and two up "
            "blocks; the loaded UNet config has down_blocks="
            f"{down_count}, up_blocks={up_count}."
        )

    device_map = {}

    # Root-level modules outside the down/mid/up stages (e.g. time embedding,
    # conv_in and output layers) stay on GPU 0, where pipeline tensors begin/end.
    for name in children:
        if name not in ("down_blocks", "mid_block", "up_blocks"):
            device_map[name] = "cuda:0"

    # Split at whole block boundaries. This is intentionally explicit rather
    # than "auto": auto mapping is based mainly on parameter bytes and can put
    # all weights on GPU 0 even when activations require a second GPU.
    for index in range(down_count):
        device_map[f"down_blocks.{index}"] = "cuda:0" if index == 0 else "cuda:1"

    device_map["mid_block"] = "cuda:1"

    # Lower-resolution up blocks run on GPU 1; final, higher-resolution blocks
    # and the output head run on GPU 0 so the pipeline scheduler/VAE stay there.
    up_gpu1_count = max(1, up_count // 2)
    for index in range(up_count):
        device_map[f"up_blocks.{index}"] = (
            "cuda:1" if index < up_gpu1_count else "cuda:0"
        )

    return device_map


def _log_cuda_memory(stage, device_indices):
    print(f"\n[GPU memory] {stage}")
    for index in device_indices:
        torch.cuda.synchronize(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        allocated = torch.cuda.memory_allocated(index) / _GIB
        reserved = torch.cuda.memory_reserved(index) / _GIB
        print(
            f"  cuda:{index} ({torch.cuda.get_device_name(index)}): "
            f"allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB, "
            f"driver_free={free_bytes / _GIB:.2f}/{total_bytes / _GIB:.2f} GiB"
        )


def _dispatch_unet_to_two_gpus(unet):
    device_map = _build_two_gpu_device_map(unet)
    print("\n[Multi-GPU] Applying Accelerate device map (no CPU offload):")
    for module_name, device_name in device_map.items():
        print(f"  {module_name:24s} -> {device_name}")

    # dispatch_model installs hooks that transfer activations at mapped module
    # boundaries. Do not call unet.to(...) or pipeline.to(...) after this point.
    try:
        unet = dispatch_model(unet, device_map=device_map, force_hooks=True)
    except torch.cuda.OutOfMemoryError as exc:
        for index in (0, 1):
            torch.cuda.empty_cache()
        raise RuntimeError(
            "Not enough free VRAM to dispatch the UNet across cuda:0/cuda:1. "
            "Restart the Kaggle session to clear stale allocations, then retry. "
            "This patch does not offload model weights to CPU."
        ) from exc

    actual_devices = {str(parameter.device) for parameter in unet.parameters()}
    if not {"cuda:0", "cuda:1"}.issubset(actual_devices):
        raise RuntimeError(
            "Multi-GPU dispatch did not place UNet parameters on both GPUs. "
            f"Observed parameter devices: {sorted(actual_devices)}"
        )
    unexpected_devices = actual_devices - {"cuda:0", "cuda:1"}
    if unexpected_devices:
        raise RuntimeError(
            "Unexpected non-GPU UNet parameter placement; CPU/disk offload is "
            f"not allowed here: {sorted(unexpected_devices)}"
        )

    # Summarize actual parameter placement as a sanity check (this is not a
    # substitute for observing activation/memory use during a real inference).
    parameter_bytes = defaultdict(int)
    for parameter in unet.parameters():
        parameter_bytes[str(parameter.device)] += parameter.numel() * parameter.element_size()
    print("\n[Multi-GPU] Confirmed UNet parameter placement:")
    for device_name in ("cuda:0", "cuda:1"):
        print(f"  {device_name}: {parameter_bytes[device_name] / _GIB:.2f} GiB of parameters")
    print(f"[Multi-GPU] hf_device_map = {getattr(unet, 'hf_device_map', device_map)}")
    return unet


def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")
    if not torch.cuda.is_available():
        raise RuntimeError("LatentSync inference requires a CUDA GPU; no CUDA device was detected.")

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("PyTorch reports no CUDA GPUs.")

    # T4 is compute capability 7.5 and supports FP16. The upstream condition
    # used major > 7, which incorrectly selected FP32 on T4 (major == 7).
    major_capability, _ = torch.cuda.get_device_capability(0)
    dtype = torch.float16 if major_capability >= 7 else torch.float32
    device0 = torch.device("cuda:0")
    torch.cuda.set_device(0)

    use_multi_gpu = gpu_count >= 2 and not args.single_gpu
    if use_multi_gpu:
        print(f"[Multi-GPU] Found {gpu_count} CUDA GPUs; using cuda:0 and cuda:1.")
    else:
        print("[Single-GPU] Using cuda:0. This may OOM on a 16-GiB GPU with LatentSync 1.6.")
        if gpu_count < 2 and not args.single_gpu:
            print("[Single-GPU] Only one CUDA GPU is visible to this process.")

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")
    print(f"CUDA GPUs visible: {gpu_count}; inference dtype: {dtype}")

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    # Keep the original LatentSync checkpoint loader: its checkpoint contains a
    # wrapper with "state_dict" and its custom load_state_dict filters certain
    # keys. Dispatch only AFTER that project-specific load has completed.
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )
    unet = unet.to(dtype=dtype)

    if use_multi_gpu:
        unet = _dispatch_unet_to_two_gpus(unet)
        _log_cuda_memory("after UNet multi-GPU dispatch", (0, 1))
    else:
        unet = unet.to(device0)
        _log_cuda_memory("after UNet placement", (0,))

    # Keep non-UNet components on cuda:0. Do not call pipeline.to("cuda") after
    # dispatching the UNet: that would attempt to move the distributed UNet.
    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda:0",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=dtype
    ).to(device0)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    )

    if use_multi_gpu and args.enable_deepcache:
        raise RuntimeError(
            "DeepCache was not validated with Accelerate's multi-GPU dispatch in this patch. "
            "For the first correctness test, rerun without --enable_deepcache. Once baseline "
            "multi-GPU inference works, DeepCache can be tested separately."
        )

    # use DeepCache (kept for the original single-GPU path)
    if args.enable_deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")
    _log_cuda_memory("before inference", (0, 1) if use_multi_gpu else (0,))

    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=args.temp_dir,
    )

    _log_cuda_memory("after inference", (0, 1) if use_multi_gpu else (0,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="Disable two-GPU dispatch and keep the UNet on cuda:0.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)
    main(config, args)
