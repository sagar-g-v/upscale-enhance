# upscale-enhance

Upscale, clean up and colour-grade folders of images and videos on the GPU.

- **Choice of model.** Six built-in Real-ESRGAN models: fast or detailed,
  photographic or drawn, 4x or 2x. Any compatible `.pth` (SRVGGNetCompact or
  RRDBNet) can also be loaded, with its architecture, size and scale detected
  from the file.
- **Any output resolution.** Keep the input's size (the default: the model
  upscales, then the result is brought back down, which cleans and sharpens).
  Or give a factor (`2x`, `1.5`), a named frame (`4k`, `1440p`, `1080p`, which
  turns for portrait images), a width or height (`2560x`, `x1440`), a box to
  fit in (`3000x2000`) or an exact size (`3000x2000!`).
- **Texture-preserving blend.** On clean, high-quality sources the model alone
  smooths fine texture (fabric, skin, wood grain) into a waxy look.
  `--keep 0.5` mixes half of the original back in. Use `0` for blurry or noisy sources.
- **Looks.** Light sharpening and vibrance, plus an optional film-style grade:
  S-curve, teal shadows and warm highlights, soft highlight glow, vignette and lifted blacks.
- **Built for big batches.** It reads ahead while the GPU works, writes with a
  thread pool and saves each file atomically. Finished files are skipped, so you
  can stop with Ctrl+C or a stop file and resume later. Priority lists, include
  and exclude globs, and before/after previews every N files are supported.
- **Transparency.** Images with alpha keep it. See-through layers (overlays,
  sprites) skip the colour grade by default, because they are usually composited
  over a background that is already graded.
- **Video.** Videos are processed frame by frame through ffmpeg pipes, keeping
  the frame rate, the sound and WebM alpha.

## Install

```bash
pip install -r requirements.txt     # install a CUDA build of torch first for GPU use
python -m upscale_enhance fetch-model                 # the default model
python -m upscale_enhance fetch-model --model all     # or every built-in model
```

Weights are stored in `~/.cache/upscale_enhance`. Set `UPSCALE_ENHANCE_HOME`
to store them somewhere else. Video processing needs `ffmpeg` and `ffprobe`,
either on `PATH` or given with `--ffmpeg-dir`.

## Use

```bash
# compare the looks on a few images first
python -m upscale_enhance compare shots/a.jpg shots/b.png -o compare/

# a folder of images -> a mirrored output folder (same size, cleaned + graded)
python -m upscale_enhance images photos/ photos_hd/ --preset cinematic --preview-every 1000

# to 4K with the large, most detailed model; important folders first
python -m upscale_enhance images in/ out/ --model x4plus --size 4k --first priority.txt

# drawn art, 2x bigger, only PNGs
python -m upscale_enhance images art/ art_2x/ --model anime --size 2x --include "*.png" --preset upscale

# videos: keeps container, frame rate, sound and WebM alpha
python -m upscale_enhance video clips/ clips_hd/ --preset cinematic --size 1440p --crf 20
python -m upscale_enhance video clips/ clips_hd/ --encoder "-c:v hevc_nvenc -cq 22"   # GPU encoder

# a community model
python -m upscale_enhance images in/ out/ --weights 4x-SomeModel.pth --size 2x
```

To stop a run, press Ctrl+C or create `DST/.stop`. Run the same command again to continue.

### Models (`--model`, default `general`)

| name | arch | scale | speed | for |
|---|---|---|---|---|
| `general` (default) | compact | 4x | fast | photos, 3D renders, mixed content |
| `general-denoise` | compact | 4x | fast | noisy or heavily compressed sources |
| `x4plus` | RRDB | 4x | ~10x slower | the most recovered detail |
| `x2plus` | RRDB | 2x | slow | 2x output, light cleanup |
| `anime` | RRDB (6 blocks) | 4x | medium | drawn art, illustrations |
| `anime-video` | compact | 4x | fast | anime video |

`list-models` shows which models are downloaded. Each model has a default
`--tile` (rows per pass) that suits a ~6 GB GPU. Lower it if memory runs out.

### Output size (`--size`, default `1` = same as the input)

| value | result for a 1920×1080 input |
|---|---|
| `1` | 1920×1080 (cleaned and sharpened, same size) |
| `2x`, `1.5` | 3840×2160, 2880×1620 |
| `4k` `1440p` `1080p` `720p` `8k` | fit in that frame (turned for portrait inputs) |
| `2560x` / `x1440` | 2560 wide / 1440 high, keeping the aspect ratio |
| `3000x2000` | fit inside the box, keeping the aspect ratio |
| `3000x2000!` | exactly 3000×2000 (stretched) |

Videos are rounded to even sizes, which most encoders need.

### Looks (`--preset`, default `enhance`)

| preset | keep | sharpen | colour | grade |
|---|---|---|---|---|
| `upscale` | 0 | – | – | – |
| `clean` | 0.5 | – | – | – |
| `enhance` (default) | 0.5 | 0.45 | vibrance 1.08, contrast 1.04 | – |
| `cinematic` | 0.5 | 0.45 | as enhance | soft film grade |
| `cinematic-strong` | 0.5 | 0.45 | as enhance | stronger contrast, glow and vignette |

`--keep`, `--sharpen`, `--vibrance` and `--contrast` override the preset's values.

### Other options

- `--first LIST`: a text file of relative paths or globs, one per line. These files are processed before everything else.
- `--min-width N`: leave images narrower than N pixels untouched, such as icons and UI pieces.
- `--grade-layers`: also grade see-through images.
- `--device cpu`: run without a GPU. This works but is slow.
- `--fp32`: use full precision on the GPU.
- `--quality`: JPEG and WebP quality. Each output keeps its source's format.

## Python API

```python
from PIL import Image
from upscale_enhance import Processor, load_model, make_look

net, device, dtype = load_model("general")           # or load_model(weights="model.pth")
proc = Processor(net, device, dtype, make_look("cinematic", keep=0.4), size="4k")
proc(Image.open("in.jpg")).save("out.jpg", quality=92)
```

## Tests

```bash
pip install pytest
python -m pytest -q          # CPU only, uses stand-in networks (no weights needed)
```

## License

The code is licensed under the Apache License 2.0 (see `LICENSE`). Model weights are not included.
`fetch-model` downloads them from the Real-ESRGAN project, which publishes them
under the BSD-3-Clause license. Check the license of any other model you load
with `--weights`.
