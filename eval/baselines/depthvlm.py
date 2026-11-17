# Reference: https://github.com/hanxunyu/DepthVLM (DepthVLM-4B, Qwen3-VL-4B + DPT depth head)
# Follows the official inference code (examples/run_demo.py, data_process/curate_datasets/*):
#   1. resize the image so that its focal length becomes CANONICAL_FX = 1000 px
#      (canonical_size = round(W * 1000 / fx), round(H * 1000 / fx)), assuming square pixels;
#   2. a single forward pass with the fixed depth prompt returns `outputs.depth_pred` in meters for the
#      canonical image (no extra rescaling: depth is invariant to image resizing);
#   3. resize the prediction back to the input resolution.
# The model needs the true focal length, so only the GT-intrinsics condition is meaningful.
# Runs in the dedicated `depthvlm` conda env (transformers 5.2.0, the version the checkpoint was saved with).
import os
import sys
from pathlib import Path
from typing import *

import click
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from moge.test.baseline import MGEBaselineInterface

CANONICAL_FX = 1000.0
PROMPT = 'Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image.'


def _resolve_repo_path(repo_path: str) -> str:
    repo_path = os.path.abspath(repo_path)
    if not Path(repo_path, 'model', 'modeling_qwen3_vl.py').is_file():
        raise FileNotFoundError(
            f'Cannot find DepthVLM code at {repo_path}. '
            'Clone https://github.com/hanxunyu/DepthVLM and pass its path via --repo.'
        )
    return repo_path


class Baseline(MGEBaselineInterface):
    def __init__(self, repo_path: str, pretrained: str, device: str):
        repo_path = _resolve_repo_path(repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from model import Qwen3VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
        self._process_vision_info = process_vision_info

        device = torch.device(device)
        pretrained = os.path.abspath(pretrained) if Path(pretrained).exists() else pretrained
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.processor.tokenizer.padding_side = 'left'
        # The official code uses flash-attention 2 (not installed here). With torch 2.5, the cuDNN backend of
        # SDPA fails on the vision encoder ("No execution plans support the graph"); disable only that
        # backend so SDPA falls back to the flash / memory-efficient kernels.
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
        self.device = device

    @click.command()
    @click.option('--repo', 'repo_path', type=click.Path(), default='../third_party/DepthVLM', help='Path to a local DepthVLM clone.')
    @click.option('--pretrained', type=str, default='../weights/depthvlm-4b', help='Local directory or Hugging Face repo id (JonnyYu828/DepthVLM-4B).')
    @click.option('--device', type=str, default='cuda:0')
    @staticmethod
    def load(repo_path: str, pretrained: str, device: str = 'cuda:0'):
        return Baseline(repo_path, pretrained, device)

    @torch.inference_mode()
    def _infer_one(self, image: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        # image: (3, H, W) in [0, 1]; intrinsics normalized by image width/height.
        _, height, width = image.shape
        fx = intrinsics[0, 0].item() * width
        canonical_w, canonical_h = round(width * CANONICAL_FX / fx), round(height * CANONICAL_FX / fx)
        rgb = (image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        img = Image.fromarray(rgb).resize((canonical_w, canonical_h), Image.BILINEAR)

        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}]},
            {'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': PROMPT}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = self._process_vision_info(messages)
        inputs = self.processor(text=[text], images=[image_inputs], padding=True, return_tensors='pt').to(self.device)
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
        )
        if outputs.depth_pred is None:
            raise RuntimeError('DepthVLM returned no depth prediction.')
        pred = outputs.depth_pred[0].float()
        while pred.ndim > 2:
            pred = pred[0]
        return F.interpolate(pred[None, None], (height, width), mode='bilinear', align_corners=False)[0, 0]

    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None) -> Dict[str, torch.Tensor]:
        if intrinsics is None:
            raise click.UsageError('DepthVLM needs GT intrinsics; run eval_depth.py with --with_intrinsics.')
        omit_batch_dim = image.ndim == 3
        if omit_batch_dim:
            image, intrinsics = image.unsqueeze(0), intrinsics.unsqueeze(0)
        depth = torch.stack([self._infer_one(image[i], intrinsics[i]) for i in range(image.shape[0])], 0)
        if omit_batch_dim:
            depth = depth.squeeze(0)
        return {'depth_metric': depth}

    @torch.inference_mode()
    def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:
        return self.infer(image, intrinsics)
