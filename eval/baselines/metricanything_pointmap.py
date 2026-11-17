# Reference: https://github.com/yjh001/metric-anything (models/student_pointmap)
# The Student PointMap model shares the MoGe-2 network architecture and checkpoint format;
# only the pretrained weights differ (finetuned by the MetricAnything authors).
import os
from pathlib import Path
from typing import *

import click
import torch
try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d

from moge.test.baseline import MGEBaselineInterface


class Baseline(MGEBaselineInterface):

    def __init__(self, pretrained_model_name_or_path: str, num_tokens: int, resolution_level: int, use_fp16: bool, device: str = 'cuda:0'):
        super().__init__()
        from moge.model.v2 import MoGeModel

        pretrained_model_name_or_path = os.path.abspath(pretrained_model_name_or_path) if Path(pretrained_model_name_or_path).exists() else pretrained_model_name_or_path
        self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

        self.device = torch.device(device)
        self.num_tokens = num_tokens
        self.resolution_level = resolution_level
        self.use_fp16 = use_fp16

    @click.command()
    @click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='../weights/metricanything-student-pointmap/student_pointmap.pt', help='Local checkpoint path or Hugging Face repo id (yjh001/metricanything_student_pointmap).')
    @click.option('--num_tokens', type=int, default=None)
    @click.option('--resolution_level', type=int, default=9)
    @click.option('--fp16', 'use_fp16', is_flag=True)
    @click.option('--device', type=str, default='cuda:0')
    @staticmethod
    def load(pretrained_model_name_or_path: str, num_tokens: int, resolution_level: int, use_fp16: bool, device: str = 'cuda:0'):
        return Baseline(pretrained_model_name_or_path, num_tokens, resolution_level, use_fp16, device)

    def _infer(self, image: torch.FloatTensor, fov_x: Optional[torch.Tensor], apply_mask: bool) -> Dict[str, torch.Tensor]:
        return self.model.infer(
            image,
            fov_x=fov_x,
            apply_mask=apply_mask,
            num_tokens=self.num_tokens,
            resolution_level=self.resolution_level,
            use_fp16=self.use_fp16,
        )

    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None):
        if intrinsics is not None:
            fov_x, _ = utils3d.pt.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None
        output = self._infer(image, fov_x, apply_mask=True)
        return {
            'points_metric': output['points'],
            'depth_metric': output['depth'],
            'intrinsics': output['intrinsics'],
        }

    @torch.inference_mode()
    def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: torch.FloatTensor = None):
        if intrinsics is not None:
            fov_x, _ = utils3d.pt.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None
        output = self._infer(image, fov_x, apply_mask=False)
        result = {
            'points_metric': output['points'],
            'depth_metric': output['depth'],
            'intrinsics': output['intrinsics'],
        }
        if 'mask' in output:
            result['mask'] = output['mask']
        return result
