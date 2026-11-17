# Reference: https://github.com/ByteDance-Seed/Depth-Anything-3 (DA3METRIC-LARGE)
# Official recipe (README FAQ): metric_depth = focal * net_output / 300, with `focal` the focal length in
# pixels (mean of fx, fy) of the image the network actually processed. Like Metric3D, the model needs
# the true focal length to produce metric depth, so only the GT-intrinsics condition is meaningful.
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


def _resolve_repo_path(repo_path: str) -> str:
    repo_path = os.path.abspath(repo_path)
    if not Path(repo_path, 'src', 'depth_anything_3', 'api.py').is_file():
        raise FileNotFoundError(
            f'Cannot find Depth-Anything-3 code at {repo_path}. '
            'Clone https://github.com/ByteDance-Seed/Depth-Anything-3 and pass its path via --repo.'
        )
    return repo_path


class Baseline(MGEBaselineInterface):
    def __init__(self, repo_path: str, pretrained: str, process_res: int, device: str):
        repo_path = _resolve_repo_path(repo_path)
        src = os.path.join(repo_path, 'src')
        if src not in sys.path:
            sys.path.insert(0, src)
        from depth_anything_3.api import DepthAnything3

        device = torch.device(device)
        pretrained = os.path.abspath(pretrained) if Path(pretrained).exists() else pretrained
        self.model = DepthAnything3.from_pretrained(pretrained).to(device).eval()
        self.process_res = process_res
        self.device = device

    @click.command()
    @click.option('--repo', 'repo_path', type=click.Path(), default='../third_party/Depth-Anything-3', help='Path to a local Depth-Anything-3 clone.')
    @click.option('--pretrained', type=str, default='../weights/da3metric-large', help='Local directory or Hugging Face repo id (depth-anything/DA3METRIC-LARGE).')
    @click.option('--process_res', type=int, default=504, help='Longest-side processing resolution; 504 is the official default.')
    @click.option('--device', type=str, default='cuda:0')
    @staticmethod
    def load(repo_path: str, pretrained: str, process_res: int, device: str = 'cuda:0'):
        return Baseline(repo_path, pretrained, process_res, device)

    @torch.inference_mode()
    def _infer_one(self, image: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        # image: (3, H, W) in [0, 1]; intrinsics normalized by image width/height.
        _, height, width = image.shape
        rgb = (image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        pred = self.model.inference([Image.fromarray(rgb)], process_res=self.process_res,
                                    process_res_method='upper_bound_resize')
        net_output = torch.from_numpy(np.asarray(pred.depth[0], dtype=np.float32)).to(self.device)
        proc_h, proc_w = net_output.shape
        # Focal length (px) of the processed image the network saw.
        fx = intrinsics[0, 0].item() * proc_w
        fy = intrinsics[1, 1].item() * proc_h
        depth = (fx + fy) / 2 * net_output / 300.0
        return F.interpolate(depth[None, None], (height, width), mode='bilinear', align_corners=False)[0, 0]

    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None) -> Dict[str, torch.Tensor]:
        if intrinsics is None:
            # Without a focal length the network output is not metric; report it as scale-invariant only.
            raise click.UsageError('DA3METRIC-LARGE needs GT intrinsics; run eval_depth.py with --with_intrinsics.')
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
