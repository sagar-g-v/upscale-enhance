"""Command line: python -m upscale_enhance <command> ...

  images   SRC DST     every image under SRC -> DST (same relative paths)
  video    SRC DST     every video under SRC -> DST (needs ffmpeg)
  compare  IMAGE...    one strip per image: original | each preset, to judge looks
  list-models          the built-in models (default marked *)
  fetch-model          download model weights (--model NAME or all)

Defaults: model "general", output size 1 (same as the input), preset "enhance".
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

from . import images, video
from .look import PRESETS, Processor, make_look, parse_size
from .model import DEFAULT_MODEL, MODELS, fetch_weights, load_model, weights_path


def add_common(p):
    p.add_argument("--preset", default="enhance", choices=sorted(PRESETS),
                   help="look to apply (default: enhance)")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS),
                   help=f"super-resolution model (default: {DEFAULT_MODEL}; see list-models)")
    p.add_argument("--size", "--scale", dest="size", default="1",
                   help="output size: 1 (same size: clean + sharpen, default), 2x, 1.5, 4k, 1440p, 1080p, "
                        "2560x, x1440, 3000x2000 (fit inside), 3000x2000! (exact)")
    p.add_argument("--keep", type=float, help="share of the original mixed back in, 0..1 (overrides the preset)")
    p.add_argument("--sharpen", type=float, help="unsharp amount (overrides the preset)")
    p.add_argument("--vibrance", type=float, help="colour boost, 1 = off (overrides the preset)")
    p.add_argument("--contrast", type=float, help="contrast, 1 = off (overrides the preset)")
    p.add_argument("--weights", default=None,
                   help="any compatible .pth (SRVGGNetCompact / RRDBNet, detected from the file); overrides --model")
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:1, cpu ...")
    p.add_argument("--fp32", action="store_true", help="full precision on the GPU (slower)")
    p.add_argument("--tile", type=int, default=None,
                   help="rows per model pass (default: the model's own; lower = less GPU memory)")
    p.add_argument("--log", default=None, help="also append progress to this file")
    p.add_argument("--stop-file", default=None, help="stop when this file appears (default: DST/.stop)")
    p.add_argument("--limit", type=int, default=None, help="only the first N files")
    p.add_argument("--force", action="store_true", help="redo files that already exist in DST")
    p.add_argument("--first", default=None,
                   help="list file of paths / globs (relative to SRC) to do before everything else")
    p.add_argument("--preview-every", type=int, default=0,
                   help="save a before/after preview every N images (videos: every N videos)")
    p.add_argument("--preview-dir", default=None, help="where previews go (default: DST/_previews)")


def processor_from(args):
    look = make_look(args.preset, keep=args.keep, sharpen=args.sharpen,
                     vibrance=args.vibrance, contrast=args.contrast)
    try:
        parse_size(args.size)                               # fail early on a bad size
    except ValueError as e:
        sys.exit(f"error: {e}")
    net, device, dtype = load_model(args.model, args.weights, args.device, half=False if args.fp32 else None)
    name = args.weights or args.model
    print(f"model {name} ({net.arch}, {net.upscale}x) on {device} ({'fp16' if str(dtype).endswith('16') else 'fp32'}), "
          f"preset {args.preset}, size {args.size}", flush=True)
    return Processor(net, device, dtype, look, size=args.size, tile=args.tile)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="upscale_enhance", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("images", help="process a folder (or one file) of images")
    p.add_argument("src")
    p.add_argument("dst")
    add_common(p)
    p.add_argument("--include", action="append", help="only paths matching this glob (repeatable)")
    p.add_argument("--exclude", action="append", help="skip paths matching this glob (repeatable)")
    p.add_argument("--min-width", type=int, default=0, help="leave images narrower than this alone")
    p.add_argument("--grade-layers", action="store_true",
                   help="also colour-grade see-through images (default: overlays get no grade)")
    p.add_argument("--quality", type=int, default=92, help="JPEG / WebP quality (default 92)")
    p.add_argument("--batch", type=int, default=16, help="images read ahead per batch")
    p.add_argument("--savers", type=int, default=4, help="threads encoding / writing results")

    p = sub.add_parser("video", help="process a folder (or one file) of videos (needs ffmpeg)")
    p.add_argument("src")
    p.add_argument("dst")
    add_common(p)
    p.add_argument("--ffmpeg-dir", default=None, help="folder with ffmpeg and ffprobe (default: PATH)")
    p.add_argument("--crf", type=int, default=22, help="encoder quality, lower = better / bigger (default 22)")
    p.add_argument("--encoder", default=None,
                   help='custom encoder arguments, e.g. "-c:v hevc_nvenc -cq 22" (replaces the default)')

    p = sub.add_parser("compare", help="before/after strips for some images, one column per preset")
    p.add_argument("images", nargs="+")
    p.add_argument("-o", "--out", default="compare", help="output folder (default: ./compare)")
    p.add_argument("--presets", default="upscale,enhance,cinematic,cinematic-strong")
    p.add_argument("--width", type=int, default=640, help="width of each column")
    add_common(p)

    p = sub.add_parser("fetch-model", help="download model weights")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"built-in model name, or 'all' (default: {DEFAULT_MODEL})")

    sub.add_parser("list-models", help="the built-in models and whether they are downloaded")

    args = ap.parse_args(argv)
    if args.cmd == "fetch-model":
        for name in (MODELS if args.model == "all" else [args.model]):
            if name not in MODELS:
                ap.error(f"unknown model {name!r}; built-in: {', '.join(MODELS)}")
            print(fetch_weights(name))
        return 0
    if args.cmd == "list-models":
        for name, info in MODELS.items():
            have = "downloaded" if weights_path(name).exists() else "not downloaded"
            star = "*" if name == DEFAULT_MODEL else " "
            print(f"{star} {name:16} {info.note}  [{have}]")
        print("\n* default. Any compatible .pth also works: --weights PATH")
        return 0

    first = images.read_list(args.first) if getattr(args, "first", None) else None
    log = images.Log(args.log)
    proc = processor_from(args)

    if args.cmd == "images":
        images.run(proc, args.src, args.dst, include=args.include, exclude=args.exclude, first=first,
                   min_width=args.min_width, grade_layers=args.grade_layers, quality=args.quality,
                   force=args.force, limit=args.limit, batch=args.batch, savers=args.savers,
                   preview_every=args.preview_every, preview_dir=args.preview_dir,
                   stop_file=args.stop_file, log=log)
    elif args.cmd == "video":
        enc = args.encoder.split() if args.encoder else None
        video.run(proc, args.src, args.dst, ffmpeg_dir=args.ffmpeg_dir, crf=args.crf, encoder_args=enc,
                  force=args.force, limit=args.limit, first=first, preview_every=args.preview_every,
                  preview_dir=args.preview_dir, stop_file=args.stop_file, log=log)
    elif args.cmd == "compare":
        compare(proc, args)
    return 0


def compare(proc, args):
    from PIL import ImageDraw
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    for path in args.images:
        img = Image.open(path)
        img.load()
        w = args.width
        h = int(w * img.size[1] / img.size[0])
        cols = [("original", img)]
        for name in presets:
            proc.look = make_look(name)
            cols.append((name, proc(img)))
        strip = Image.new("RGB", ((w + 10) * len(cols) - 10, h + 28), "white")
        draw = ImageDraw.Draw(strip)
        for i, (name, im) in enumerate(cols):
            strip.paste(im.convert("RGB").resize((w, h), Image.LANCZOS), (i * (w + 10), 28))
            draw.text((i * (w + 10) + 6, 6), name, fill="black")
        target = out / (Path(path).stem + "_compare.jpg")
        strip.save(target, quality=90)
        print(target, flush=True)


if __name__ == "__main__":
    sys.exit(main())
