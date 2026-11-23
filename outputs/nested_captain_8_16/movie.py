"""Render a continuous 1080p zoom through one nested PhotoQuilt mosaic."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from PIL import Image
from render import coord, parent, row, col, run_dir, viewport

SIDE = 1080
FPS = 30
Image.MAX_IMAGE_PIXELS = 16384 ** 2


def pyramid(path):
    with Image.open(path) as image:
        levels = [image.convert("RGB")]
    while levels[-1].width > 1024:
        levels.append(levels[-1].reduce(2))
    return sorted(levels, key=lambda image: image.width)


def interpolate_window(start, end, progress):
    # Travel along one line in canvas coordinates while the zoom speed varies smoothly.
    start_x, start_y, start_width = start
    end_x, end_y, end_width = end
    smooth = progress ** 3 * (10 - 15 * progress + 6 * progress ** 2)
    width = start_width * (end_width / start_width) ** smooth
    fraction = (width - start_width) / (end_width - start_width)
    return (start_x + (end_x - start_x) * fraction,
            start_y + (end_y - start_y) * fraction, width)


def movie(n):
    path = run_dir(n)
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()

    with (path / "example_selection.json").open() as source:
        selected = json.load(source)["ranked"][0]["tile"]
    ir, ic = coord(selected)
    outer = (0., 0., 1.)
    child = (col / 16, row / 16, 1 / 16)
    detail = ((col + ic/n) / 16, (row + ir/n) / 16, 1 / (16*n))
    layers = [(pyramid(parent / "mosaic.png"), outer),
              (pyramid(path / "mosaic.png"), child),
              (pyramid(path / "tiles" / (selected + ".png")), detail)]
    def frame(window):
        left, top, width = window
        rect = left, top, left + width, top + width
        out = Image.new("RGB", (SIDE, SIDE))
        for levels, box in layers:
            desired = SIDE * box[2] / width
            level = next((image for image in levels if image.width >= desired), levels[-1])
            part = viewport(level, rect, box, SIDE)
            if part:
                dest, patch = part
                out.paste(patch, dest[:2])
        return out.tobytes()

    sequence = ((outer, outer, 45), (outer, child, 135), (child, child, 90),
                (child, detail, 105), (detail, detail, 45), (detail, child, 75),
                (child, child, 60))
    count = sum(length for _, _, length in sequence)
    output = path / "zoom.mp4"
    temporary = path / "zoom.partial.mp4"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{SIDE}x{SIDE}", "-framerate",
               str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "3",
               "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags",
               "+faststart", str(temporary)]
    rendered = {}
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for start, end, length in sequence:
                for i in range(length):
                    if start == end:
                        if start not in rendered:
                            rendered[start] = frame(start)
                        data = rendered[start]
                    else:
                        data = frame(interpolate_window(start, end, (i+1)/length))
                    encoder.stdin.write(data)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("Video encoding failed")
        finally:
            if not encoder.stdin.closed:
                encoder.stdin.close()

    verify = subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(temporary),
        "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1"],
        capture_output=True, text=True, check=True)
    frames = [int(line.split("=")[1]) for line in verify.stdout.splitlines() if line.startswith("frame=")]
    if not frames or frames[-1] != count:
        raise RuntimeError(f"Expected {count} decoded video frames, got {frames[-1:]}")
    temporary.replace(output)
    print(f"Verified {output}, {count} frames, {SIDE}x{SIDE}, {FPS} fps, {count/FPS:.1f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=int, choices=(8,16))
    movie(parser.parse_args().grid)
