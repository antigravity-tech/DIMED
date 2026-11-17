# Reference: https://github.com/apple/ml-depth-pro (Depth Pro, ICLR 2025)
import os
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import *

import click
import torch

from moge.test.baseline import MGEBaselineInterface


def _resolve_repo_path(repo_path: str) -> str:
    repo_path = os.path.abspath(repo_path)
    if not Path(repo_path, 'src', 'depth_pro', 'depth_pro.py').is_file():
        raise FileNotFoundError(
            f'Cannot find Depth Pro code at {repo_path}. '
            'Clone https://github.com/apple/ml-depth-pro and pass its path via --repo.'
        )
    return repo_path


class Baseline(MGEBaselineInterface):
    def __init__(self, repo_path: str, pretrained: str, use_fp16: bool, device: str):
        repo_path = _resolve_repo_path(repo_path)
        src = os.path.join(repo_path, 'src')
        if src not in sys.path:
            sys.path.insert(0, src)
        # Import the model module directly: `depth_pro/__init__.py` pulls in utils.py, which needs
        # pillow_heif only for reading HEIC files. Register a bare package so the relative imports work.
        if 'depth_pro' not in sys.modules:
            pkg = types.ModuleType('depth_pro')
            pkg.__path__ = [os.path.join(src, 'depth_pro')]
            sys.modules['depth_pro'] = pkg
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, create_model_and_transforms

        device = torch.device(device)
        config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=os.path.abspath(pretrained))
        self.model, _ = create_model_and_transforms(config=config, device=device,
                                                    precision=torch.half if use_fp16 else torch.float32)
        self.model.eval()
        self.dtype = torch.half if use_fp16 else torch.float32
        self.device = device

    @click.command()
    @click.option('--repo', 'repo_path', type=click.Path(), default='../third_party/ml-depth-pro', help='Path to a local ml-depth-pro clone.')
    @click.option('--pretrained', type=click.Path(), default='../weights/depthpro/depth_pro.pt', help='Path to depth_pro.pt (apple/DepthPro).')
    @click.option('--fp16', 'use_fp16', is_flag=True)
    @click.option('--device', type=str, default='cuda:0')
    @staticmethod
    def load(repo_path: str, pretrained: str, use_fp16: bool, device: str = 'cuda:0'):
        return Baseline(repo_path, pretrained, use_fp16, device)

    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None) -> Dict[str, torch.Tensor]:
        # image: (B, 3, H, W) or (3, H, W) in [0, 1]; intrinsics normalized by image width/height.
        omit_batch_dim = image.ndim == 3
        if omit_batch_dim:
            image = image.unsqueeze(0)
            if intrinsics is not None:
                intrinsics = intrinsics.unsqueeze(0)
        _, _, height, width = image.shape

        # Same preprocessing as the official transform: Normalize(mean=0.5, std=0.5).
        x = ((image - 0.5) / 0.5).to(self.device, self.dtype)
        depths, focals = [], []
        for i in range(x.shape[0]):
            # Depth Pro uses a single focal length in pixels of the input width.
            f_px = (intrinsics[i, 0, 0] * width).to(self.device) if intrinsics is not None else None
            out = self.model.infer(x[i:i + 1], f_px=f_px)
            depths.append(out['depth'].float())
            focals.append(out['focallength_px'].float().reshape(()))
        depth = torch.stack(depths, 0)
        fx = torch.stack(focals, 0) / width
        fy = torch.stack(focals, 0) / height
        K = torch.zeros(len(depths), 3, 3, device=depth.device)
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = fx, fy, 0.5, 0.5, 1.0

        result = {'depth_metric': depth, 'intrinsics': K}
        if omit_batch_dim:
            result = {k: v.squeeze(0) for k, v in result.items()}
        return result

    @torch.inference_mode()
    def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:
        return self.infer(image, intrinsics)
