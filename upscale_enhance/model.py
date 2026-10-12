"""Super-resolution networks, the built-in model list, and loading.

Two architectures cover the Real-ESRGAN family and most community models:
  compact  SRVGGNetCompact (plain conv stack + pixel shuffle): fast
  rrdb     RRDBNet (ESRGAN's residual-in-residual dense blocks): slower, sharper
Any compatible weights file can be loaded; its architecture, size and scale
are read from the file itself (`load_model(weights=...)`).
"""
import math
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOME = Path(os.environ.get("UPSCALE_ENHANCE_HOME", Path.home() / ".cache" / "upscale_enhance"))
_REL = "https://github.com/xinntao/Real-ESRGAN/releases/download/"


@dataclass(frozen=True)
class ModelInfo:
    file: str
    url: str
    tile: int          # rows per model pass that fit a ~6 GB GPU at 1080p width
    note: str


# Real-ESRGAN's published models (BSD-3-Clause). "general" is the default.
MODELS = {
    "general": ModelInfo("realesr-general-x4v3.pth", _REL + "v0.2.5.0/realesr-general-x4v3.pth", 540,
                         "4x, fast, photos / renders / mixed content (default)"),
    "general-denoise": ModelInfo("realesr-general-wdn-x4v3.pth", _REL + "v0.2.5.0/realesr-general-wdn-x4v3.pth", 540,
                                 "4x, fast, stronger denoising for noisy or heavily compressed sources"),
    "x4plus": ModelInfo("RealESRGAN_x4plus.pth", _REL + "v0.1.0/RealESRGAN_x4plus.pth", 96,
                        "4x, large model: most detail, ~10x slower"),
    "x2plus": ModelInfo("RealESRGAN_x2plus.pth", _REL + "v0.2.1/RealESRGAN_x2plus.pth", 192,
                        "2x, large model: for 2x output or light cleanup"),
    "anime": ModelInfo("RealESRGAN_x4plus_anime_6B.pth", _REL + "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth", 192,
                       "4x, drawn art / anime illustrations"),
    "anime-video": ModelInfo("realesr-animevideov3.pth", _REL + "v0.2.5.0/realesr-animevideov3.pth", 540,
                             "4x, fast, anime video"),
}
DEFAULT_MODEL = "general"
DEFAULT_WEIGHTS = HOME / MODELS[DEFAULT_MODEL].file


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class _RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = _RDB(nf, gc), _RDB(nf, gc), _RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN generator. scale 2 / 1 models take pixel-unshuffled input."""

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.upscale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.upscale == 2:
            x = F.pixel_unshuffle(x, 2)
        elif self.upscale == 1:
            x = F.pixel_unshuffle(x, 4)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def build_from_state(state):
    """A network matching a weights file's tensors (architecture, size, scale)."""
    if "conv_first.weight" in state:
        blocks = len({k.split(".")[1] for k in state if k.startswith("body.")})
        in_ch = state["conv_first.weight"].shape[1]
        scale = {3: 4, 12: 2, 48: 1}.get(in_ch)
        if scale is None:
            raise ValueError(f"unsupported RRDBNet input channels: {in_ch}")
        feat = state["conv_first.weight"].shape[0]
        grow = state["body.0.rdb1.conv1.weight"].shape[0]
        return RRDBNet(scale=scale, num_feat=feat, num_block=blocks, num_grow_ch=grow), "rrdb"
    if "body.0.weight" in state:
        convs = sorted(int(k.split(".")[1]) for k in state if k.startswith("body.") and k.endswith(".weight")
                       and state[k].dim() == 4)
        last = state[f"body.{convs[-1]}.weight"]
        feat = state["body.0.weight"].shape[0]
        scale = int(round(math.sqrt(last.shape[0] / 3)))
        return SRVGGNetCompact(num_feat=feat, num_conv=len(convs) - 2, upscale=scale), "compact"
    raise ValueError("unrecognised weights file (expected an SRVGGNetCompact or RRDBNet state dict)")


def pick_device(want="auto"):
    if want != "auto":
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_path(model=DEFAULT_MODEL):
    return HOME / MODELS[model].file


def fetch_weights(model=DEFAULT_MODEL, dest=None):
    """Download a built-in model's weights (once). Returns the path."""
    info = MODELS[model]
    dest = Path(dest) if dest else HOME / info.file
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {info.url}\n  -> {dest}", file=sys.stderr, flush=True)
    with urllib.request.urlopen(info.url, timeout=60) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dest)
    return dest


def load_model(model=DEFAULT_MODEL, weights=None, device="auto", half=None):
    """-> (network, device, dtype). `weights` (any compatible .pth) overrides
    `model`. The network has .upscale (its native factor) and .tile (rows
    per pass suggested for it). half: fp16 (default: on for CUDA)."""
    device = pick_device(device)
    if weights:
        path, tile = Path(weights), None
    else:
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; built-in: {', '.join(MODELS)}")
        path, tile = weights_path(model), MODELS[model].tile
    if not path.exists():
        raise FileNotFoundError(f"model weights not found: {path}\n"
                                f"run `python -m upscale_enhance fetch-model --model {model}` or pass --weights PATH")
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    net, arch = build_from_state(state)
    net.load_state_dict(state, strict=True)
    net.tile = tile or (540 if arch == "compact" else 128)
    net.arch = arch
    if half is None:
        half = device.type == "cuda"
    dtype = torch.float16 if half else torch.float32
    return net.eval().to(device=device, dtype=dtype), device, dtype
