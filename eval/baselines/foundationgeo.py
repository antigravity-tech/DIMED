# Reference: https://github.com/mx-liu6/FoundationGeo (Stage-II FoundationGeo, metric point map)
# FoundationGeo pins utils3d@c5daf6f. The only utils3d call on its inference path is
# `utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)`; the utils3d installed for MoGe
# builds the float cx/cy on CPU and fails with CUDA fx/fy, so we patch in a device-aware version.
import os
import sys
import types
from pathlib import Path
from typing import *

import click
import torch

from moge.test.baseline import MGEBaselineInterface


def _resolve_repo_path(repo_path: str) -> str:
    repo_path = os.path.abspath(repo_path)
    if not Path(repo_path, 'foundationgeo', 'model', 'v1.py').is_file():
        raise FileNotFoundError(
            f'Cannot find FoundationGeo code at {repo_path}. '
            'Clone https://github.com/mx-liu6/FoundationGeo and pass its path via --repo.'
        )
    return repo_path


def _intrinsics_from_focal_center(fx, fy, cx, cy) -> torch.Tensor:
    fx, fy, cx, cy = torch.broadcast_tensors(*(torch.as_tensor(v, dtype=fx.dtype, device=fx.device) for v in (fx, fy, cx, cy)))
    zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
    return torch.stack([fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim=-1).unflatten(-1, (3, 3))


def _import_foundationgeo(repo_path: str):
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    import foundationgeo.model.v1 as fg_v1
    # Give v1 its own `utils3d` namespace so the patch does not leak into MoGe's utils3d.
    patched_torch = types.SimpleNamespace(**{k: getattr(fg_v1.utils3d.torch, k) for k in dir(fg_v1.utils3d.torch) if not k.startswith('__')})
    patched_torch.intrinsics_from_focal_center = _intrinsics_from_focal_center
    fg_v1.utils3d = types.SimpleNamespace(torch=patched_torch, numpy=getattr(fg_v1.utils3d, 'numpy', None))
    return fg_v1.FoundationGeo


class Baseline(MGEBaselineInterface):
    def __init__(self, repo_path: str, pretrained: str, num_tokens: Optional[int], resolution_level: int, use_fp16: bool, device: str):
        repo_path = _resolve_repo_path(repo_path)
        FoundationGeo = _import_foundationgeo(repo_path)

        device = torch.device(device)
        pretrained = os.path.abspath(pretrained) if Path(pretrained).exists() else pretrained
        if not Path(pretrained).exists():
            # The HF repo stores the checkpoint as FoundationGeo.pt rather than the default model.pt.
            from huggingface_hub import hf_hub_download
            pretrained = hf_hub_download(pretrained, 'FoundationGeo.pt')
        self.model = FoundationGeo.from_pretrained(pretrained).to(device).eval()

        self.device = device
        self.num_tokens = num_tokens
        self.resolution_level = resolution_level
        self.use_fp16 = use_fp16

    @click.command()
    @click.option('--repo', 'repo_path', type=click.Path(), default='../third_party/FoundationGeo', help='Path to a local FoundationGeo clone.')
    @click.option('--pretrained', type=str, default='../weights/foundationgeo-1.1/FoundationGeo.pt', help='Local checkpoint path or Hugging Face repo id (mxliu-hku/FoundationGeo-1.1).')
    @click.option('--num_tokens', type=int, default=None)
    @click.option('--resolution_level', type=int, default=9)
    @click.option('--fp16', 'use_fp16', is_flag=True)
    @click.option('--device', type=str, default='cuda:0')
    @staticmethod
    def load(repo_path: str, pretrained: str, num_tokens: Optional[int], resolution_level: int, use_fp16: bool, device: str = 'cuda:0'):
        return Baseline(repo_path, pretrained, num_tokens, resolution_level, use_fp16, device)

    def _infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor], apply_mask: bool) -> Dict[str, torch.Tensor]:
        if intrinsics is not None:
            # Horizontal FoV from normalized intrinsics (fx in units of image width).
            fov_x = torch.rad2deg(2 * torch.atan(0.5 / intrinsics[..., 0, 0]))
        else:
            fov_x = None
        output = self.model.infer(
            image,
            fov_x=fov_x,
            resolution_level=self.resolution_level,
            num_tokens=self.num_tokens,
            apply_mask=apply_mask,
            use_fp16=self.use_fp16,
        )
        return {
            'points_metric': output['points_metric'],
            'depth_metric': output['depth_metric'],
            'intrinsics': output['intrinsics'],
            'mask': output['mask'],
        }

    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None) -> Dict[str, torch.Tensor]:
        return self._infer(image, intrinsics, apply_mask=True)

    @torch.inference_mode()
    def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:
        return self._infer(image, intrinsics, apply_mask=False)
