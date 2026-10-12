"""Runs on the CPU with a stand-in network (nearest-neighbour 4x), so no
weights or GPU are needed."""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from upscale_enhance import images
from upscale_enhance.look import PRESETS, Processor, cinema_grade, finish, is_layer, make_look, parse_size
from upscale_enhance.model import RRDBNet, SRVGGNetCompact, load_model


class Nearest4(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=4, mode="nearest")


def proc(preset="enhance", scale=1, tile=540):
    return Processor(Nearest4(), torch.device("cpu"), torch.float32, PRESETS[preset], size=str(scale), tile=tile)


def noise(w=97, h=61, mode="RGB", seed=0):
    rng = np.random.default_rng(seed)
    ch = 4 if mode == "RGBA" else 3
    return Image.fromarray(rng.integers(0, 255, (h, w, ch), dtype=np.uint8), mode)


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_output_size(scale):
    out = proc(scale=scale)(noise())
    assert out.size == (97 * scale, 61 * scale)


def test_upscale_preset_same_size_is_identity():
    # nearest 4x then antialiased bicubic back down, no blend / finishing:
    # flat colours survive exactly
    img = Image.new("RGB", (64, 40), (120, 30, 200))
    out = proc("upscale")(img)
    assert np.abs(np.asarray(out, int) - np.asarray(img, int)).max() <= 1


def test_strips_have_no_seams():
    img = noise(80, 70)
    whole = np.asarray(proc(tile=1000)(img), int)
    strips = np.asarray(proc(tile=16)(img), int)
    assert np.abs(whole - strips).max() <= 2


def test_alpha_kept_and_layer_detection():
    img = noise(50, 40, "RGBA")
    a = np.asarray(img)[:, :, 3].copy()
    out = proc(scale=2)(img)
    assert out.mode == "RGBA" and out.size == (100, 80)
    assert is_layer(img)
    opaque = img.copy()
    opaque.putalpha(255)
    assert not is_layer(opaque)
    assert np.asarray(proc()(img))[:, :, 3].tolist() == a.tolist()


def test_grade_and_finish_stay_in_range():
    x = torch.rand(1, 3, 64, 96)
    for y in (cinema_grade(x, **PRESETS["cinematic-strong"].grade), finish(x)):
        assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1


def test_vignette_darkens_corners():
    x = torch.full((1, 3, 64, 64), 0.5)
    y = cinema_grade(x, curve=0, teal=0, warm=0, glow=0, vignette=0.3, lift=0)
    assert y[0, 0, 0, 0] < y[0, 0, 32, 32]


def test_make_look_overrides():
    lk = make_look("cinematic", keep=0.2, sharpen=None)
    assert lk.keep == 0.2 and lk.sharpen == PRESETS["cinematic"].sharpen and lk.grade


def test_batch_run_resumes(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    (src / "a" / "b").mkdir(parents=True)
    noise(40, 30).save(src / "one.png")
    noise(40, 30, seed=1).save(src / "a" / "b" / "two.jpg", quality=95)
    noise(40, 30, "RGBA", seed=2).save(src / "a" / "layer.png")
    (src / "notes.txt").write_text("not an image")
    p = proc("cinematic")
    assert images.run(p, src, dst, savers=2) == 3
    assert (dst / "a" / "b" / "two.jpg").exists() and Image.open(dst / "a" / "b" / "two.jpg").format == "JPEG"
    assert Image.open(dst / "a" / "layer.png").mode == "RGBA"
    assert images.run(p, src, dst) == 0                     # everything already done
    with pytest.raises(ValueError):
        images.run(p, src, src)


@pytest.mark.parametrize("spec,expect", [
    ("1", (1920, 1080)), ("2x", (3840, 2160)), ("1.5", (2880, 1620)),
    ("4k", (3840, 2160)), ("1440p", (2560, 1440)), ("2560x", (2560, 1440)), ("x720", (1280, 720)),
    ("1000x1000", (1000, 562)), ("1000x1000!", (1000, 1000)),
])
def test_parse_size_landscape(spec, expect):
    assert parse_size(spec)(1920, 1080) == expect


def test_named_size_turns_with_portrait_images():
    assert parse_size("4k")(1080, 1920) == (2160, 3840)


def test_bad_size():
    with pytest.raises(ValueError):
        parse_size("huge")


@pytest.mark.parametrize("spec,expect", [("1.5", (146, 92)), ("x100", (159, 100)), ("90x90!", (90, 90))])
def test_any_output_size(spec, expect):
    p = Processor(Nearest4(), torch.device("cpu"), torch.float32, PRESETS["cinematic"], size=spec, tile=16)
    assert p(noise(97, 61)).size == expect


def test_even_sizes_for_video():
    p = Processor(Nearest4(), torch.device("cpu"), torch.float32, size="1", even=True)
    assert p.out_size(97, 61) == (98, 62)


@pytest.mark.parametrize("net", [
    SRVGGNetCompact(num_feat=8, num_conv=2, upscale=4),
    RRDBNet(scale=4, num_feat=8, num_block=1, num_grow_ch=4),
    RRDBNet(scale=2, num_feat=8, num_block=1, num_grow_ch=4),
])
def test_weights_detected_and_odd_sizes(tmp_path, net):
    path = tmp_path / "w.pth"
    torch.save({"params": net.state_dict()}, path)
    loaded, device, dtype = load_model(weights=path, device="cpu")
    assert type(loaded) is type(net) and loaded.upscale == net.upscale
    p = Processor(loaded, device, dtype, PRESETS["enhance"], size="2x", tile=8)
    assert p(noise(33, 21)).size == (66, 42)            # odd sizes work for x2 / pixel-unshuffle models too
    assert Processor(loaded, device, dtype, size="1", tile=8)(noise(33, 21)).size == (33, 21)


def test_collect_order_first_and_filters(tmp_path):
    for rel in ("x/10.png", "x/2.png", "y/1.png", "skip/1.png"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        noise(8, 8).save(tmp_path / rel)
    got = images.collect(tmp_path, exclude=["skip/*"], first=["y/*"])
    assert got == ["y/1.png", "x/2.png", "x/10.png"]
