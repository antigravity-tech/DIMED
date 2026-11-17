# Reference: https://github.com/YvanYin/Metric3D
import os
from pathlib import Path
from typing import *

import click
import torch
import torch.nn.functional as F
import cv2

from moge.test.baseline import MGEBaselineInterface


BACKBONE_MAP = {
    'vits': 'metric3d_vit_small',
    'vitl': 'metric3d_vit_large',
    'vitg': 'metric3d_vit_giant2',
}


def _load_metric3d_model(backbone: str, repo_path: Optional[str], pretrained: Optional[str]):
    model_name = BACKBONE_MAP[backbone]
    load_pretrained = pretrained is None

    if repo_path is not None:
        repo_path = os.path.abspath(repo_path)
        if not Path(repo_path, 'hubconf.py').is_file():
            raise FileNotFoundError(
                f'Cannot find Metric3D hubconf.py at {repo_path}. '
                'Clone https://github.com/YvanYin/Metric3D and pass its path via --repo.'
            )
        model = torch.hub.load(repo_path, model_name, source='local', pretrain=load_pretrained, trust_repo=True)
    else:
        try:
            model = torch.hub.load('yvanyin/metric3d', model_name, pretrain=load_pretrained, trust_repo=True)
        except Exception as e:
            raise RuntimeError(
                'Failed to load Metric3D from torch.hub (often GitHub rate limit). '
                'Clone the repo locally and pass --repo /path/to/Metric3D.'
            ) from e

    if pretrained is not None:
        checkpoint = torch.load(pretrained, map_location='cpu', weights_only=True)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

    return model


class Baseline(MGEBaselineInterface):
    def __init__(self, backbone: Literal['vits', 'vitl', 'vitg'], repo_path: Optional[str], pretrained: Optional[str], device: str):
        device = torch.device(device)
        model = _load_metric3d_model(backbone, repo_path, pretrained)
        model.to(device).eval()

        self.model = model
        self.device = device

    @click.command()
    @click.option('--repo', 'repo_path', type=click.Path(), default='../third_party/Metric3D', help='Path to a local Metric3D clone. Avoids torch.hub GitHub rate limits.')
    @click.option('--pretrained', 'pretrained', type=click.Path(), default=None, help='Local checkpoint (.pth), e.g. ../weights/metric3d/metric_depth_vit_large_800k.pth. If omitted, weights are downloaded via torch.hub.')
    @click.option('--backbone', type=click.Choice(['vits', 'vitl', 'vitg']), default='vitl', help='Encoder architecture.')
    @click.option('--device', type=str, default='cuda', help='Device to use.')
    @staticmethod
    def load(repo_path: Optional[str], pretrained: Optional[str], backbone: str = 'vitl', device: str = 'cuda'):
        return Baseline(backbone, repo_path, pretrained, device)

    @torch.inference_mode()
    def inference_one_image(self, image: torch.Tensor, intrinsics: torch.Tensor = None):
        # Reference: https://github.com/YvanYin/Metric3D/blob/main/mono/utils/do_test.py

        # rgb_origin: RGB, 0-255, uint8
        rgb_origin = image.cpu().numpy().transpose((1, 2, 0)) * 255

        # keep ratio resize
        input_size = (616, 1064) # for vit model
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        if intrinsics is not None:
            focal = intrinsics[0, 0] * int(w * scale)
        
        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        # normalize rgb
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].to(self.device)

        # inference 
        pred_depth, confidence, output_dict = self.model.inference({'input': rgb})

        # un pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        pred_depth = pred_depth.clamp_min(0.5)  # clamp to 0.5m, since metric3d could yield very small depth values, resulting in crashed the scale shift alignment.
        
        # upsample to original size
        pred_depth = F.interpolate(pred_depth[None, None, :, :], image.shape[-2:], mode='bilinear').squeeze()
        
        if intrinsics is not None:
            # de-canonical transform
            canonical_to_real_scale = focal / 1000.0 # 1000.0 is the focal length of canonical camera
            pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
            pred_depth = torch.clamp(pred_depth, 0, 300)

        pred_normal, normal_confidence = output_dict['prediction_normal'].split([3, 1], dim=1) # see https://arxiv.org/abs/2109.09881 for details

        # un pad and resize to some size if needed
        pred_normal = pred_normal.squeeze(0)
        pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]

        # you can now do anything with the normal
        pred_normal = F.interpolate(pred_normal[None, :, :, :], image.shape[-2:], mode='bilinear').squeeze(0)
        pred_normal = F.normalize(pred_normal, p=2, dim=0)

        return pred_depth, pred_normal.permute(1, 2, 0)

    @torch.inference_mode()
    def infer(self, image: torch.Tensor, intrinsics: torch.Tensor = None):    
        # image: (B, H, W, 3) or (H, W, 3)
        if image.ndim == 3:
            pred_depth, pred_normal = self.inference_one_image(image, intrinsics)
        else:
            for i in range(image.shape[0]):
                pred_depth_i, pred_normal_i = self.inference_one_image(image[i], intrinsics[i] if intrinsics is not None else None)
                pred_depth.append(pred_depth_i)
                pred_normal.append(pred_normal_i)
            pred_depth = torch.stack(pred_depth, dim=0)
            pred_normal = torch.stack(pred_normal, dim=0)
        
        if intrinsics is not None:
            return {
                "depth_metric": pred_depth,
            }
        else:
            return {
                "depth_scale_invariant": pred_depth,
            }
