# Reference: https://github.com/lpiccinelli-eth/UniK3D (UniK3D, CVPR 2025)
import os
import sys
from pathlib import Path
from typing import *

import click
import torch

from moge.test.baseline import MGEBaselineInterface


def _resolve_repo_path(repo_path: str) -> str:
    repo_path = os.path.abspath(repo_path)
    if not Path(repo_path, 'unik3d', 'models', 'unik3d.py').is_file():
        raise FileNotFoundError(
            f'Cannot find UniK3D code at {repo_path}. '
            'Clone https://github.com/lpiccinelli-eth/UniK3D and pass its path via --repo.'
        )
    return repo_path


class Baseline(MGEBaselineInterface):
    def __init__(self, repo_path: str, pretrained: str, resolution_level: int, device: str):
        repo_path = _resolve_repo_path(repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from unik3d.models import UniK3D
        from unik3d.utils.camera import Pinhole
        self._pinhole = Pinhole

        device = torch.device(device)
        self.model = UniK3D.from_pretrained(pretrained).to(device).eval()
        self.model.resolution_level = resolution_level
        self.device = device

    @click.command()
    @click.option('--repo', 'repo_path', type=click.Path(), default='../third_party/UniK3D', help='Path to a local UniK3D clone.')
    @click.option('--pretrained', type=str, default='../weights/unik3d-vitl', help='Local directory (config.json + model.safetensors) or Hugging Face repo id (lpiccinelli/unik3d-vitl).')
    @click.option('--resolution_level', type=int, default=9, help='Resolution level in [0, 10); the official demo uses 9.')
    @click.option('--device', type=str, default='cuda:0')
    @staticmethod
    def load(repo_path: str, pretrained: str, resolution_level: int, device: str = 'cuda:0'):
        return Baseline(repo_path, pretrained, resolution_level, device)

    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None) -> Dict[str, torch.Tensor]:
        # image: (B, 3, H, W) or (3, H, W) in [0, 1]; intrinsics normalized by image width/height.
        omit_batch_dim = image.ndim == 3
        if omit_batch_dim:
            image = image.unsqueeze(0)
            if intrinsics is not None:
                intrinsics = intrinsics.unsqueeze(0)
        _, _, height, width = image.shape

        camera = None
        if intrinsics is not None:
            # UniK3D expects a pinhole camera in pixel units of the input image.
            K = intrinsics.clone().float()
            K[:, 0, :] *= width
            K[:, 1, :] *= height
            camera = self._pinhole(K=K)

        # UniK3D normalizes internally from 0-255 inputs.
        output = self.model.infer(image * 255.0, camera)
        result = {
            'depth_metric': output['depth'][:, 0].float(),
            'points_metric': output['points'].permute(0, 2, 3, 1).float(),
        }
        if omit_batch_dim:
            result = {k: v.squeeze(0) for k, v in result.items()}
        return result

    @torch.inference_mode()
    def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:
        return self.infer(image, intrinsics)
