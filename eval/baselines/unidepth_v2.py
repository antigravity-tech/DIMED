# Reference: https://github.com/lpiccinelli-eth/UniDepth (UniDepthV2)
import os
import sys
from pathlib import Path
from typing import *

import click
import torch

from moge.test.baseline import MGEBaselineInterface


def _resolve_repo_path(repo_path: str) -> str:
    repo_path = os.path.abspath(repo_path)
    if not Path(repo_path, 'unidepth', 'models', 'unidepthv2').is_dir():
        raise FileNotFoundError(
            f'Cannot find UniDepth code at {repo_path}. '
            'Clone https://github.com/lpiccinelli-eth/UniDepth and pass its path via --repo.'
        )
    return repo_path


class Baseline(MGEBaselineInterface):
    def __init__(self, repo_path: str, pretrained: str, resolution_level: Optional[int], device: str):
        repo_path = _resolve_repo_path(repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from unidepth.models import UniDepthV2

        device = torch.device(device)
        self.model = UniDepthV2.from_pretrained(pretrained).to(device).eval()
        if resolution_level is not None:
            self.model.resolution_level = resolution_level
        self.device = device

    @click.command()
    @click.option('--repo', 'repo_path', type=click.Path(), default='../third_party/UniDepth', help='Path to a local UniDepth clone.')
    @click.option('--pretrained', type=str, default='../weights/unidepth-v2-vitl14', help='Local directory (config.json + model.safetensors) or Hugging Face repo id (lpiccinelli/unidepth-v2-vitl14).')
    @click.option('--resolution_level', type=int, default=9, help='Resolution level in [0, 10); 9 as in the MoGe-2 evaluation of UniDepthV2 (and as for UniK3D).')
    @click.option('--device', type=str, default='cuda:0')
    @staticmethod
    def load(repo_path: str, pretrained: str, resolution_level: Optional[int], device: str = 'cuda:0'):
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
            # UniDepth expects pixel-space intrinsics of the input image.
            camera = intrinsics.clone().float()
            camera[:, 0, :] *= width
            camera[:, 1, :] *= height

        # UniDepth normalizes internally from 0-255 inputs.
        output = self.model.infer(image * 255.0, camera)
        depth = output['depth'][:, 0]
        points = output['points'].permute(0, 2, 3, 1)
        pred_intrinsics = output['intrinsics'].clone()
        pred_intrinsics[:, 0, :] /= width
        pred_intrinsics[:, 1, :] /= height

        result = {
            'depth_metric': depth,
            'points_metric': points,
            'intrinsics': pred_intrinsics,
        }
        if omit_batch_dim:
            result = {k: v.squeeze(0) for k, v in result.items()}
        return result

    @torch.inference_mode()
    def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:
        return self.infer(image, intrinsics)
