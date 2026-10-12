"""Per-frame processing: super-resolution, blend, finishing and colour grade.

All of it runs on the model's device (GPU when available); only the finished
frame comes back to the CPU.
"""
import re
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class Look:
    """How a frame is finished.

    keep       share of the original mixed back into the model's output (0..1).
               Clean, high-quality sources look waxy with the model alone;
               0.3-0.6 keeps their fine texture. Use 0 for blurry / noisy sources.
    sharpen    unsharp-mask amount (0 = off), sigma its radius in pixels
    vibrance   colour boost that favours dull colours (1 = off)
    contrast   contrast around mid grey (1 = off)
    grade      film-style colour grade (None = off), see cinema_grade()
    """
    keep: float = 0.5
    sharpen: float = 0.45
    sigma: float = 1.4
    vibrance: float = 1.08
    contrast: float = 1.04
    grade: dict = field(default=None)


CINEMATIC = dict(curve=0.22, teal=0.02, warm=0.035, glow=0.10, vignette=0.20, lift=0.012)
CINEMATIC_STRONG = dict(curve=0.32, teal=0.03, warm=0.045, glow=0.14, vignette=0.30, lift=0.012)

PRESETS = {
    "upscale": Look(keep=0.0, sharpen=0.0, vibrance=1.0, contrast=1.0),
    "clean": Look(keep=0.5, sharpen=0.0, vibrance=1.0, contrast=1.0),
    "enhance": Look(),
    "cinematic": Look(grade=CINEMATIC),
    "cinematic-strong": Look(grade=CINEMATIC_STRONG),
}


def make_look(preset="enhance", **overrides):
    """A preset with some fields changed (None values are ignored)."""
    look = PRESETS[preset]
    changes = {k: v for k, v in overrides.items() if v is not None}
    return replace(look, **changes) if changes else look


def _gauss(sigma, device):
    r = max(int(sigma * 3 + 0.5), 1)
    k = torch.exp(-(torch.arange(-r, r + 1, dtype=torch.float32, device=device) ** 2) / (2 * sigma * sigma))
    return k / k.sum(), r


def blur(x, sigma):
    """Separable Gaussian blur of a (1,C,H,W) tensor (edge pixels repeated
    at the borders, which works for any size, even smaller than the kernel)."""
    k, r = _gauss(sigma, x.device)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def finish(x, sharpen=0.45, sigma=1.4, vibrance=1.08, contrast=1.04):
    """Unsharp mask, vibrance and contrast on a (1,3,H,W) 0..1 tensor."""
    if sharpen:
        x = x + sharpen * (x - blur(x, sigma))
    if vibrance != 1.0:
        luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        sat = (x.max(1, keepdim=True)[0] - x.min(1, keepdim=True)[0]).clamp(0, 1)
        x = luma + (x - luma) * (1 + (vibrance - 1) * (1 - sat))
    if contrast != 1.0:
        x = 0.5 + (x - 0.5) * contrast
    return x.clamp(0, 1)


def cinema_grade(x, curve=0.22, teal=0.02, warm=0.035, glow=0.10, vignette=0.20, lift=0.012):
    """Film-style grade on a whole (1,3,H,W) 0..1 frame.

    curve     S-curve: deeper shadows, rolled-off highlights
    teal      deep shadows pushed toward teal (mid-dark tones, e.g. dark hair, are spared)
    warm      highlights pushed toward warm orange
    glow      soft bloom from the brightest areas
    vignette  corners darker
    lift      blacks raised a little (film rather than pure digital black)
    """
    h, w = x.shape[2], x.shape[3]
    if glow:
        small = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False, antialias=True)
        bright = ((small - 0.75) / 0.25).clamp(0, 1) ** 2 * small
        bloom = F.interpolate(blur(bright, 6.0), size=(h, w), mode="bilinear", align_corners=False)
        x = 1 - (1 - x) * (1 - glow * bloom)                    # screen blend
    if curve:
        s = x * x * (3 - 2 * x)
        x = x + curve * (s - x)
    if teal or warm:
        luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        sh = (1 - luma / 0.3).clamp(0, 1) * (luma / 0.15).clamp(0, 1)
        hi = ((luma - 0.5) / 0.5).clamp(0, 1)
        tint_sh = torch.tensor([-1.0, 0.25, 0.75], device=x.device).view(1, 3, 1, 1)
        tint_hi = torch.tensor([1.0, 0.45, -0.6], device=x.device).view(1, 3, 1, 1)
        x = x + teal * sh * tint_sh + warm * hi * tint_hi
    if vignette:
        yy = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1)
        xx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w)
        d = ((xx ** 2 + yy ** 2) / 2).clamp(0, 1)
        x = x * (1 - vignette * d ** 1.5)
    if lift:
        x = lift + x * (1 - lift)
    return x.clamp(0, 1)


# named output resolutions: (long side, short side); the box turns with the
# image, so a portrait image gets a portrait 4K frame
RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440), "2k": (2560, 1440),
               "4k": (3840, 2160), "2160p": (3840, 2160), "5k": (5120, 2880), "8k": (7680, 4320)}


def parse_size(spec):
    """Output-size rule from a string -> function (w, h) -> (W, H).

      "1.5" / "2x"       scale factor
      "4k", "1440p" ...  fit inside that frame (turned to match the image's orientation)
      "3000x2000"        fit inside this box, keeping the aspect ratio
      "3000x2000!"       exactly this size (stretches)
      "2560x" / "x1440"  this width / height, the other side follows the aspect ratio
    """
    s = str(spec).strip().lower()
    if s in RESOLUTIONS:
        long_, short = RESOLUTIONS[s]

        def named(w, h):
            bw, bh = (long_, short) if w >= h else (short, long_)
            f = min(bw / w, bh / h)
            return round(w * f), round(h * f)
        return named
    m = re.fullmatch(r"(\d+(?:\.\d+)?)x?", s)
    if m and float(m.group(1)) <= 8:                    # "2", "2x", "1.5": a factor; "2560x" is a width
        f = float(m.group(1))
        if f <= 0:
            raise ValueError("scale factor must be above 0")
        return lambda w, h: (round(w * f), round(h * f))
    m = re.fullmatch(r"(\d+)?x(\d+)?(!?)", s)
    if m and (m.group(1) or m.group(2)):
        W, H, exact = int(m.group(1) or 0), int(m.group(2) or 0), bool(m.group(3))
        if exact:
            if not (W and H):
                raise ValueError("an exact size needs both width and height: WxH!")
            return lambda w, h: (W, H)
        if W and H:
            return lambda w, h: (round(w * min(W / w, H / h)), round(h * min(W / w, H / h)))
        if W:
            return lambda w, h: (W, round(h * W / w))
        return lambda w, h: (round(w * H / h), H)
    raise ValueError(f"can't read output size {spec!r} (try 1, 2x, 1.5, 4k, 1440p, 2560x, x1080, 3000x2000)")


class Processor:
    """frame (PIL image) -> finished frame.

    size: output size rule (parse_size string or function); default: the
    input's own size (the model upscales, the result is brought back down,
    which cleans and sharpens). The model runs on horizontal strips (tile
    rows + padding) so large frames fit in GPU memory; strips are cut after
    the finishing step, so there are no seams. `even=True` rounds output
    sizes to even numbers (needed by most video encoders).
    """

    def __init__(self, net, device, dtype, look=None, size="1", tile=None, pad=None, even=False):
        self.net, self.device, self.dtype = net, device, dtype
        self.look = look or PRESETS["enhance"]
        self.size = parse_size(size) if isinstance(size, (str, int, float)) else size
        self.native = getattr(net, "upscale", 4)
        # RRDB x2 / x1 models pixel-unshuffle their input: sizes must divide by 2 / 4
        self.multiple = {4: 1, 3: 1, 2: 2, 1: 4}.get(self.native, 1) if getattr(net, "arch", "") == "rrdb" else 1
        m = self.multiple
        self.tile = max(m, (tile or getattr(net, "tile", 540)) // m * m)
        self.pad = (pad if pad is not None else (24 if getattr(net, "arch", "") == "rrdb" else 16)) // m * m
        self.even = even

    def out_size(self, w, h):
        W, H = self.size(w, h)
        W, H = max(1, int(W)), max(1, int(H))
        if self.even:
            W, H = W + W % 2, H + H % 2
        return W, H

    @torch.no_grad()
    def tensor(self, rgb, grade=True):
        """(H,W,3) uint8 array -> (1,3,H_out,W_out) float tensor on the device."""
        lk, n = self.look, self.native
        a = torch.from_numpy(np.array(rgb, dtype=np.uint8, copy=True)).to(self.device).permute(2, 0, 1)[None]
        a = a.to(self.dtype) / 255.0
        h, w = a.shape[2], a.shape[3]
        W, H = self.out_size(w, h)
        m = self.multiple
        ph, pw = (-h) % m, (-w) % m
        ap = F.pad(a.float(), (0, pw, 0, ph), mode="replicate").to(self.dtype) if (ph or pw) else a
        k = W // w if (W % w == 0 and H == h * (W // w)) else None   # a whole-number factor
        outs = []
        for y in range(0, h, self.tile):
            y0, y1 = max(0, y - self.pad), min(h + ph, y + self.tile + self.pad)
            part = self.net(ap[:, :, y0:y1]).float()[:, :, :, : w * n]
            if k is None:                               # any other size: resized once, below
                outs.append(part[:, :, (y - y0) * n:(min(y + self.tile, h) - y0) * n])
                continue
            rows = (y1 - y0) * k
            if part.shape[2] != rows or part.shape[3] != w * k:
                part = F.interpolate(part, size=(rows, w * k), mode="bicubic", antialias=True, align_corners=False)
            if lk.keep:
                base = a[:, :, y0:min(y1, h)].float()
                if base.shape[2] < y1 - y0:             # the padded rows at the bottom
                    base = F.pad(base, (0, 0, 0, y1 - y0 - base.shape[2]), mode="replicate")
                if k != 1:
                    base = F.interpolate(base, size=(rows, w * k), mode="bicubic", align_corners=False)
                part = part * (1 - lk.keep) + base * lk.keep
            if lk.sharpen or lk.vibrance != 1.0 or lk.contrast != 1.0:
                part = finish(part.clamp(0, 1), lk.sharpen, lk.sigma, lk.vibrance, lk.contrast)
            outs.append(part[:, :, (y - y0) * k:(min(y + self.tile, h) - y0) * k])
        full = torch.cat(outs, 2)
        if k is None:
            full = F.interpolate(full, size=(H, W), mode="bicubic", antialias=True, align_corners=False)
            if lk.keep:
                base = F.interpolate(a.float(), size=(H, W), mode="bicubic", antialias=True, align_corners=False)
                full = full * (1 - lk.keep) + base * lk.keep
            if lk.sharpen or lk.vibrance != 1.0 or lk.contrast != 1.0:
                full = finish(full.clamp(0, 1), lk.sharpen, lk.sigma, lk.vibrance, lk.contrast)
        full = full.clamp(0, 1)
        if grade and lk.grade:
            full = cinema_grade(full, **lk.grade)
        return full

    def __call__(self, img, grade=True):
        """PIL image -> PIL image. Transparency is kept (resized with the image).
        grade=False skips the colour grade (e.g. for see-through layers that
        are composited over a graded background)."""
        alpha = img.getchannel("A") if img.mode in ("RGBA", "LA", "PA") else None
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
            alpha = img.getchannel("A")
        full = self.tensor(np.asarray(img.convert("RGB")), grade=grade)
        out = Image.fromarray((full[0] * 255 + 0.5).byte().permute(1, 2, 0).cpu().numpy())
        if alpha is not None:
            if alpha.size != out.size:
                alpha = alpha.resize(out.size, Image.LANCZOS)
            out.putalpha(alpha)
        return out


def is_layer(img):
    """True for an image that is see-through somewhere (an overlay / sprite)."""
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        return img.convert("RGBA").getchannel("A").getextrema()[0] < 255
    return False
