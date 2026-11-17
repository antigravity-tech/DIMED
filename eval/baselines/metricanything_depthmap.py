# Reference: https://github.com/yjh001/metric-anything (models/student_depthmap)
import os
import sys
from pathlib import Path
from typing import *

import click
import torch
import torch.nn.functional as F

from moge.test.baseline import MGEBaselineInterface


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_repo_path(repo_path: str) -> str:
    repo_path = os.path.abspath(repo_path)
    if not Path(repo_path, 'depth_model.py').is_file():
        raise FileNotFoundError(
            f'Cannot find MetricAnything student_depthmap code at {repo_path} (expected depth_model.py). '
            'Clone https://github.com/yjh001/metric-anything and pass models/student_depthmap via --repo.'
        )
    return repo_path


class Baseline(MGEBaselineInterface):
    def __init__(self, repo_path: str, pretrained: str, device: Union[torch.device, str]):
        repo_path = _resolve_repo_path(repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from depth_model import MetricAnythingDepthMap, DEFAULT_CONFIG

        device = torch.device(device)
        pretrained_path = os.path.abspath(pretrained) if Path(pretrained).exists() else pretrained

        # `vit_factory.create_vit` calls `torch.hub.load("network", ..., source="local")` with a
        # path relative to the current working directory, so we must chdir into `repo_path` while
        # the backbone (and thus the checkpoint-loading `create_model` call) is being constructed.
        previous_cwd = os.getcwd()
        try:
            os.chdir(repo_path)
            self.model = MetricAnythingDepthMap.from_pretrained(
                pretrained_path,
                model_kwargs={'device': device, 'config': DEFAULT_CONFIG},
                filename='student_depthmap.pt',
            ).to(device).eval()
        finally:
            os.chdir(previous_cwd)

        self.device = device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @click.command()
    @click.option('--repo', 'repo_path', type=click.Path(), default='../third_party/metric-anything/models/student_depthmap', help='Path to the MetricAnything student_depthmap code directory.')
    @click.option('--pretrained', type=str, default='../weights/metricanything-student-depthmap/student_depthmap.pt', help='Local checkpoint path or Hugging Face repo id (yjh001/metricanything_student_depthmap).')
    @click.option('--device', type=str, default='cuda:0')
    @staticmethod
    def load(repo_path: str, pretrained: str, device: str = 'cuda:0'):
        return Baseline(repo_path, pretrained, device)

    @torch.inference_mode()
    def _infer_batched(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor]) -> torch.Tensor:
        # image: (B, 3, H, W) in [0, 1]
        _, _, height, width = image.shape
        x = (image - self.mean) / self.std

        if intrinsics is not None:
            # intrinsics is normalized (fx, fy in units of 1/width, 1/height); f_px is in pixels of the input image.
            f_px = (intrinsics[:, 0, 0] * width).to(x.device)
        else:
            f_px = torch.full((x.shape[0],), float(width), device=x.device)

        depths = []
        for i in range(x.shape[0]):
            pred = self.model.infer(x[i:i + 1], f_px=f_px[i].item())
            depths.append(pred['depth'])
        return torch.stack(depths, dim=0)

    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None) -> Dict[str, torch.Tensor]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
            if intrinsics is not None:
                intrinsics = intrinsics.unsqueeze(0)
            omit_batch_dim = True
        else:
            omit_batch_dim = False

        depth = self._infer_batched(image, intrinsics)

        if omit_batch_dim:
            depth = depth.squeeze(0)

        if intrinsics is not None:
            return {'depth_metric': depth}
        else:
            # Without known intrinsics the model falls back to f_px = image width,
            # which is only an assumption, not a scale-invariant guarantee; still returned
            # under depth_metric since the model always outputs depth in meters by design.
            return {'depth_metric': depth}

    @torch.inference_mode()
    def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:
        return self.infer(image, intrinsics)
