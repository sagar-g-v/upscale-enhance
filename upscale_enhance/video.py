"""Process videos frame by frame with ffmpeg.

Each video is decoded to raw frames (ffmpeg pipe), every frame goes through
the Processor, and the frames are piped into an encoder; the original sound
is copied. Output: same relative path and container in the output folder.
Videos with transparency (WebM VP8/VP9 alpha) keep it and skip the colour grade.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .images import Log, natural, preview_strip

VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".ogv"}

# default encoder per output container: (video args, pixel format, alpha pixel format)
ENCODERS = {
    ".webm": (["-c:v", "libvpx-vp9", "-crf", "{crf}", "-b:v", "0", "-deadline", "good",
               "-cpu-used", "4", "-row-mt", "1"], "yuv420p", "yuva420p"),
    ".mp4": (["-c:v", "libx264", "-crf", "{crf}", "-preset", "medium"], "yuv420p", None),
    ".m4v": (["-c:v", "libx264", "-crf", "{crf}", "-preset", "medium"], "yuv420p", None),
    ".mkv": (["-c:v", "libx264", "-crf", "{crf}", "-preset", "medium"], "yuv420p", None),
    ".mov": (["-c:v", "libx264", "-crf", "{crf}", "-preset", "medium"], "yuv420p", None),
    ".avi": (["-c:v", "libx264", "-crf", "{crf}", "-preset", "medium"], "yuv420p", None),
    ".ogv": (["-c:v", "libtheora", "-q:v", "8"], "yuv420p", None),
}


def find_tools(ffmpeg_dir=None):
    names = ("ffmpeg", "ffprobe")
    if ffmpeg_dir:
        paths = [Path(ffmpeg_dir) / (n + (".exe" if os.name == "nt" else "")) for n in names]
    else:
        paths = [shutil.which(n) for n in names]
    if not all(p and Path(p).exists() for p in paths):
        raise FileNotFoundError("ffmpeg / ffprobe not found: put them on PATH or pass --ffmpeg-dir")
    return [str(p) for p in paths]


def probe(ffprobe, path):
    r = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                        "stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,nb_frames:"
                        "stream_tags=alpha_mode:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    info = json.loads(r.stdout or "{}")
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = any(s.get("codec_type") == "audio" for s in streams)
    alpha = bool(v and (v.get("tags", {}).get("alpha_mode") == "1" or v.get("pix_fmt", "").startswith("yuva")))
    frames = None
    if v:
        try:
            frames = int(v.get("nb_frames") or 0) or None
            if not frames:
                num, den = (int(x) for x in v["r_frame_rate"].split("/"))
                frames = int(float(info.get("format", {}).get("duration", 0)) * num / max(den, 1)) or None
        except (ValueError, KeyError):
            frames = None
    return v, audio, alpha, frames


def collect(src):
    src = Path(src)
    if src.is_file():
        return [src.name]
    out = [Path(d, f).relative_to(src).as_posix() for d, _, fs in os.walk(src)
           for f in fs if Path(f).suffix.lower() in VIDEO_EXT]
    return sorted(out, key=natural)


def process_one(processor, tools, src, dst, *, crf=22, encoder_args=None, preview=None, log=None):
    """One video -> dst. Returns the number of frames written (0 = failed / skipped)."""
    ffmpeg, ffprobe = tools
    log = log or Log()
    v, audio, alpha, _ = probe(ffprobe, src)
    if not v:
        log(f"skipped {src} (no video stream)")
        return 0
    w, h, fps = int(v["width"]), int(v["height"]), v["r_frame_rate"]
    ext = Path(dst).suffix.lower()
    args, pix, apix = ENCODERS.get(ext, ENCODERS[".mp4"])
    if alpha and not apix:
        log(f"note: {ext} can't keep transparency; writing it without")
        alpha = False
    ch = 4 if alpha else 3
    dec = [ffmpeg, "-v", "error"]
    if alpha:                                   # libvpx decoders read the alpha channel; the built-in ones don't
        dec += ["-c:v", "libvpx-vp9" if v.get("codec_name") == "vp9" else "libvpx"]
    dec += ["-i", str(src), "-f", "rawvideo", "-pix_fmt", "rgba" if alpha else "rgb24", "-"]
    processor.even = True                       # yuv420 video needs even sizes
    W, H = processor.out_size(w, h)
    tmp = Path(dst).with_name(Path(dst).stem + ".tmp" + ext)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    enc = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba" if alpha else "rgb24",
           "-s", f"{W}x{H}", "-r", fps, "-i", "-"]
    if audio:
        enc += ["-i", str(src), "-map", "0:v", "-map", "1:a?", "-c:a", "copy"]
    enc += [a.format(crf=crf) for a in (encoder_args or args)]
    enc += ["-pix_fmt", apix if alpha else pix, str(tmp)]
    d = subprocess.Popen(dec, stdout=subprocess.PIPE)
    e = subprocess.Popen(enc, stdin=subprocess.PIPE)
    size, n = w * h * ch, 0
    try:
        while True:
            buf = d.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, ch)
            full = processor.tensor(frame[:, :, :3], grade=not alpha)
            arr = (full[0] * 255 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
            if alpha:
                a = frame[:, :, 3]
                if (W, H) != (w, h):
                    a = np.asarray(Image.fromarray(a).resize((W, H), Image.LANCZOS))
                arr = np.dstack([arr, a])
            e.stdin.write(arr.tobytes())
            if preview and n == 15:
                preview_strip(Image.fromarray(frame[:, :, :3]), Image.fromarray(arr[:, :, :3]), preview)
            n += 1
    finally:
        d.stdout.close()
        d.wait()
        e.stdin.close()
        ok = e.wait() == 0
    if not ok or n == 0:
        log(f"FAILED {src} (encoder error or no frames)")
        if tmp.exists():
            tmp.unlink()
        return 0
    os.replace(tmp, dst)
    return n


def run(processor, src, dst, *, ffmpeg_dir=None, crf=22, encoder_args=None, force=False, limit=None,
        first=None, preview_every=0, preview_dir=None, stop_file=None, log=None):
    log = log or Log()
    tools = find_tools(ffmpeg_dir)
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("output folder must differ from the input folder")
    base = src.parent if src.is_file() else src
    names = collect(src)
    if first:
        import fnmatch
        names.sort(key=lambda r: next((i for i, p in enumerate(first) if r == p or fnmatch.fnmatch(r, p)), len(first)))
    todo = names if force else [n for n in names if not (dst / n).exists()]
    if limit:
        todo = todo[:limit]
    stop_file = Path(stop_file) if stop_file else dst / ".stop"
    if stop_file.exists():
        stop_file.unlink()
    log(f"{len(todo)} video(s) to do ({len(names) - len(todo)} already done)")
    t0, frames = time.time(), 0
    for i, rel in enumerate(todo):
        if stop_file.exists():
            log(f"stopped ({stop_file} exists); run again to continue")
            break
        prev = None
        if preview_every and i % preview_every == 0:
            prev = Path(preview_dir or dst / "_previews") / f"video_{i:05d}.jpg"
        t1 = time.time()
        try:
            n = process_one(processor, tools, base / rel, dst / rel, crf=crf, encoder_args=encoder_args,
                            preview=prev, log=log)
        except KeyboardInterrupt:
            log("interrupted; run again to continue (the current video starts over)")
            break
        frames += n
        spent = time.time() - t1
        left = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
        log(f"{i + 1}/{len(todo)} {rel}: {n} frames, {spent:.0f} s ({spent / max(n, 1):.2f} s/frame, ~{left / 3600:.1f} h left)")
        if prev and prev.exists():
            log(f"preview {prev}")
    log(f"done: {frames} frame(s)")
    return frames
