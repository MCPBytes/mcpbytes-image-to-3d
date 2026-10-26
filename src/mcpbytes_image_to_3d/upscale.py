"""RealESRGAN x4 for Hunyuan3D's paint step, without the realesrgan and basicsr packages.

Tencent's hy3dpaint/utils/image_super_utils.py imports `realesrgan.RealESRGANer` and
`basicsr.archs.rrdbnet_arch.RRDBNet`; `install()` registers this module under both names so the vendored code runs
unchanged. RRDBNet is basicsr 1.4.2's basicsr/archs/rrdbnet_arch.py (Copyright XPixel Group, Apache-2.0) and
RealESRGANer.enhance is Real-ESRGAN 0.3.0's realesrgan/utils.py (Copyright Xintao Wang, BSD-3-Clause), reduced to the
one call Tencent makes: x4, no tiling, no pre-padding, half precision, an RGB uint8 array in. Both load the same
RealESRGAN_x4plus.pth.
"""

import sys
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        if scale != 4:  # shortcut: only the x4 net Tencent uses; port basicsr's pixel_unshuffle branch for x2/x1
            raise ValueError("only the x4 RRDBNet is implemented")
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class RealESRGANer:
    def __init__(self, scale, model_path, dni_weight=None, model=None, tile=0, tile_pad=10, pre_pad=10, half=False, device=None, gpu_id=None):
        if tile or pre_pad or dni_weight is not None or gpu_id:
            raise ValueError("only untiled, unpadded single-model upscaling is implemented")
        self.scale, self.half = scale, half
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights["params_ema" if "params_ema" in weights else "params"], strict=True)
        self.model = model.eval().to(self.device)
        if half:
            self.model = self.model.half()

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler="realesrgan"):
        """(the image `scale` times larger, "RGB") for an HxWx3 uint8 array, as Real-ESRGAN returns it."""
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8 or outscale not in (None, self.scale):
            raise ValueError("expects an HxWx3 uint8 image at the model's own scale")
        # Real-ESRGAN takes OpenCV's BGR and swaps it for the net; Tencent hands it RGB, so the net sees the channels
        # reversed and they are reversed back after. Kept as is: it is the output Tencent's paint was released with.
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).float().div_(255.0)
        x = x.unsqueeze(0).to(self.device)
        if self.half:
            x = x.half()
        y = self.model(x).squeeze(0).float().cpu().clamp_(0, 1).numpy()
        return (y[::-1].transpose(1, 2, 0) * 255.0).round().astype(np.uint8), "RGB"


def install():
    """Serve `realesrgan` and `basicsr.archs.rrdbnet_arch` from this module."""
    me = sys.modules[__name__]
    basicsr, archs = types.ModuleType("basicsr"), types.ModuleType("basicsr.archs")
    basicsr.archs, archs.rrdbnet_arch = archs, me
    sys.modules.update({"realesrgan": me, "basicsr": basicsr, "basicsr.archs": archs, "basicsr.archs.rrdbnet_arch": me})
