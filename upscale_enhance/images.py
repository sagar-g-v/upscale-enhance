"""Batch-process a folder of images into a mirrored output folder.

The GPU never waits for the disk: one thread reads and decodes the next
batch while the model runs, and a pool of threads encodes and writes the
finished images. Each result is written to a temporary file and renamed, so
a reader never sees a half-written image. Images already in the output are
skipped, so a run can be stopped (Ctrl+C or the stop file) and resumed.
"""
import fnmatch
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .look import is_layer

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def collect(src, include=None, exclude=None, first=None):
    """Relative paths ('/'-separated) of the images under src, in order:
    entries matching a pattern in `first` (in that list's order), then the
    rest in natural order. include / exclude: glob patterns on the path."""
    src = Path(src)
    if src.is_file():
        return [src.name]
    found = []
    for d, _, files in os.walk(src):
        for f in files:
            if Path(f).suffix.lower() not in IMAGE_EXT:
                continue
            rel = Path(d, f).relative_to(src).as_posix()
            if include and not any(fnmatch.fnmatch(rel, p) for p in include):
                continue
            if exclude and any(fnmatch.fnmatch(rel, p) for p in exclude):
                continue
            found.append(rel)
    found.sort(key=natural)
    if first:
        def rank(rel):
            for i, p in enumerate(first):
                if rel == p or fnmatch.fnmatch(rel, p):
                    return i
            return len(first)
        found.sort(key=rank)                    # stable: natural order inside each group
    return found


def read_list(path):
    """A list file: one path or glob per line; '#' comments, blank lines ignored."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.replace("\\", "/"))
    return out


class Log:
    def __init__(self, path=None):
        self.path = path

    def __call__(self, msg):
        line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def save(img, out, target, quality=92, keep_format=True):
    """Write `out` like the source image `img` (same format), atomically."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fmt = (img.format or "").upper() if keep_format else ""
    if not fmt:
        fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP", ".bmp": "BMP",
               ".tif": "TIFF", ".tiff": "TIFF"}.get(target.suffix.lower(), "PNG")
    tmp = target.with_name(target.name + ".tmp")
    if fmt == "JPEG":
        out.convert("RGB").save(tmp, "JPEG", quality=quality)
    elif fmt == "WEBP":
        out.save(tmp, "WEBP", quality=quality)
    elif fmt in ("BMP", "TIFF"):
        out.save(tmp, fmt)
    else:
        out.save(tmp, "PNG", compress_level=6)
    os.replace(tmp, target)


def preview_strip(before, after, path, width=960):
    h = int(width * before.size[1] / before.size[0])
    strip = Image.new("RGB", (width * 2 + 10, h), "white")
    strip.paste(before.convert("RGB").resize((width, h), Image.LANCZOS), (0, 0))
    strip.paste(after.convert("RGB").resize((width, h), Image.LANCZOS), (width + 10, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    strip.save(path, quality=90)


def run(processor, src, dst, *, include=None, exclude=None, first=None, min_width=0,
        grade_layers=False, quality=92, force=False, limit=None, batch=16, savers=4,
        preview_every=0, preview_dir=None, stop_file=None, log=None):
    """Process every image under src into dst (same relative paths).
    Returns the number of images written."""
    log = log or Log()
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("output folder must differ from the input folder")
    base = src.parent if src.is_file() else src
    names = collect(src, include, exclude, first)
    todo = names if force else [n for n in names if not (dst / n).exists()]
    if limit:
        todo = todo[:limit]
    stop_file = Path(stop_file) if stop_file else dst / ".stop"
    if stop_file.exists():
        stop_file.unlink()
    log(f"{len(todo)} image(s) to do ({len(names) - len(todo)} already done)")
    if not todo:
        return 0

    def load(rels):
        items = []
        for rel in rels:
            try:
                img = Image.open(base / rel)
                img.load()
            except Exception as e:                      # unreadable: leave it
                log(f"skipped {rel} ({type(e).__name__})")
                continue
            if img.size[0] < min_width:
                continue
            items.append((rel, img))
        return items

    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    t0, done, written, next_preview = time.time(), 0, 0, 0
    with ThreadPoolExecutor(1) as loader, ThreadPoolExecutor(savers) as pool:
        nxt = loader.submit(load, batches[0])
        pending = []
        try:
            for i, rels in enumerate(batches):
                items = nxt.result()
                nxt = loader.submit(load, batches[i + 1]) if i + 1 < len(batches) else None
                for p in pending:
                    p.result()
                if stop_file.exists():
                    log(f"stopped ({stop_file} exists); run again to continue")
                    return written
                pending = []
                for k, (rel, img) in enumerate(items):
                    out = processor(img, grade=grade_layers or not is_layer(img))
                    if preview_every and done >= next_preview and k == len(items) // 2:
                        p = Path(preview_dir or dst / "_previews") / f"preview_{done:06d}.jpg"
                        preview_strip(img, out, p)
                        log(f"preview {p}")
                        next_preview = done + preview_every
                    pending.append(pool.submit(save, img, out, dst / rel, quality))
                    written += 1
                done += len(rels)
                rate = (time.time() - t0) / done
                log(f"{done}/{len(todo)} ({rate:.2f} s/image, ~{rate * (len(todo) - done) / 3600:.1f} h left) {rels[-1]}")
        except KeyboardInterrupt:
            log("interrupted; finishing the images in progress, run again to continue")
        finally:
            for p in pending:
                p.result()
    log(f"done: {written} image(s) written to {dst}")
    return written


if __name__ == "__main__":
    sys.exit("use: python -m upscale_enhance images ...")
