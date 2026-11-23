"""Play the original lap once, then pull back continuously through three more laps."""
import argparse
from contextlib import ExitStack
from itertools import accumulate
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageChops, ImageStat
from outputs.outward_captain_hulk_thor16.palette import scene_palettes
from outputs.outward_palette16.camera import project_layers

ROOT = Path(__file__).resolve().parent
GIF = ROOT / "spider_iron_captain_hulk_thor_zoom_out.gif"
MP4 = ROOT / "spider_iron_captain_hulk_thor_continuous_4_laps.mp4"
CONTINUOUS_LAPS = 3
LEG_FRAMES = 225  # 4.5 seconds at 50 fps
FFMPEG = os.environ.get("IMAGEIO_FFMPEG_EXE") or shutil.which("ffmpeg")


def store_frame(stack, frame):
    file = stack.enter_context(os.fdopen(os.memfd_create("mosaic-frame.bmp"), "w+b"))
    frame.convert("RGB").save(file, "BMP")
    file.flush()
    return file.fileno()


def prepare(stack):
    """Keep unique decoded frames in RAM so the export needs no large scratch disk."""
    frames, durations, stops, references = [], [], [], []
    with Image.open(GIF) as gif:
        for i in range(gif.n_frames):
            gif.seek(i)
            dt = gif.info["duration"] // 10
            if dt >= 100:
                stops.append(i)
                references.append(gif.convert("RGB").resize((88, 88), Image.Resampling.BOX))
            frames.append(store_frame(stack, gif))
            durations.append(dt)
        assert len(stops) == 5 and durations[-1] == 2
        assert gif.convert("RGB").resize((88, 88), Image.Resampling.BOX).tobytes() == references[0].tobytes()
    return frames, durations, stops, references


def continuous_frames(stack, frames):
    report = json.loads((ROOT / "render.json").read_text())
    edges = report["edges"]
    slots = [(edge["row"] - 1) * 16 + edge["column"] - 1 for edge in edges]
    views = []
    for edge in edges:
        with Image.open(ROOT / (edge["outer"] + "_overview.png")) as image:
            views.append(image.convert("RGB"))

    with Image.open(GIF) as source:
        approved = Image.new("P", (1, 1))
        approved.putpalette(source.getpalette())
    palette_at, palette_info = scene_palettes(approved, views, constant_zoom=True)
    assert palette_info == report["palette"]

    levels = []
    for j, edge in enumerate(edges):
        hero = edge["outer"]
        if hero in ("hulk", "thor"):
            path = ROOT / (hero + "_4096.jpg")
        else:
            folder = "outward_thor_iron16" if hero == "iron_man" else "outward_thor16"
            path = ROOT.parent / folder / (hero + "_16x16.jpg")
        with Image.open(path) as image:
            image.draft("RGB", (4096, 4096))
            board = image.convert("RGB")
        assert board.size == (4096, 4096)
        # Use this edition's recursive views in the shared native tiles.
        row, col = divmod(slots[j], 16)
        board.paste(views[j-1].reduce(4), (col * 256, row * 256))
        levels.append([views[j], board.reduce(2), board])

    steps = []
    for j in range(len(edges)):
        outer = (j + 1) % len(edges)
        row, col = divmod(slots[outer], 16)
        for k in range(LEG_FRAMES):
            # Equal change in log magnification per frame, including across scene boundaries.
            width = 16 ** (k / LEG_FRAMES - 1)
            bounds = (col * (1-width) / 15, row * (1-width) / 15, width)
            frame = project_layers(levels, slots, outer, bounds)
            steps.append(len(frames))
            frame = frame.quantize(palette=palette_at(j, k / LEG_FRAMES),
                                   dither=Image.Dither.FLOYDSTEINBERG)
            frames.append(store_frame(stack, frame))
    return steps


def timeline(durations, stops, steps):
    last = len(durations) - 1
    plan = list(enumerate(durations[:-1]))
    checks = {i: hero for hero, i in enumerate(stops)}
    assert sum(durations[:-1]) == 5000
    assert len(steps) == len(stops) * LEG_FRAMES
    for _ in range(CONTINUOUS_LAPS):
        for hero in range(len(stops)):
            checks[len(plan) + hero * LEG_FRAMES] = hero
        plan.extend((i, 2) for i in steps)
    checks[len(plan)] = 0
    # The final source sample contributes one more centisecond in the MP4 muxer.
    plan.extend([(last, 1), (last, 1)])
    return plan, checks


def verify(path, plan, checks, references):
    # Read presentation times from an actual full decode, and inspect all scene boundaries plus the ending.
    select = "+".join(f"eq(n,{i})" for i in checks)
    vf = f"settb=1/100,showinfo,select='{select}',scale=88:88:flags=area"
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-vf", vf,
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    result = subprocess.run(cmd, capture_output=True, check=True)
    pts = [int(x) for x in re.findall(rb"\bn:\s*\d+\s+pts:\s*(-?\d+)", result.stderr)]
    assert pts == [0, *accumulate(dt for _, dt in plan[:-1])], "Encoded timestamps differ from schedule"
    checksums = re.findall(rb"\bchecksum:([0-9A-F]+)", result.stderr)
    assert len(checksums) == len(pts)
    moving = [h for h, t in zip(checksums, pts) if 5000 <= t < 11750]
    assert len(moving) == 3 * 5 * LEG_FRAMES and all(a != b for a, b in zip(moving, moving[1:]))
    stride = 88 * 88 * 3
    assert len(result.stdout) == len(checks) * stride
    max_error = 0
    for j, hero in enumerate(checks.values()):
        frame = Image.frombytes("RGB", (88, 88), result.stdout[j*stride:(j+1)*stride])
        error = sum(ImageStat.Stat(ImageChops.difference(frame, references[hero])).mean) / 3
        max_error = max(error, max_error)
    assert max_error < 10, f"Stop images differ from the source: {max_error:.2f}"
    # Saturated magenta on the left of the mask exposes the missing GIF palette.
    # Sample both the first held image and each later departure at three offsets.
    spider_starts = [i for i, hero in checks.items() if hero == 0][:-1]
    samples = [i + k for i in spider_starts for k in (0, 5, 11)]
    select = "+".join(f"eq(n,{i})" for i in samples)
    probe = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf",
             f"select='{select}',crop=352:704:0:0", "-fps_mode", "passthrough",
             "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.check_output(probe)
    pixels = np.frombuffer(raw, np.uint8).reshape(len(samples), 704, 352, 3).astype(np.int16)
    red, green, blue = np.moveaxis(pixels, -1, 0)
    counts = ((red > 145) & (red-green > 55) & (blue-green > 35)).sum(axis=(1, 2))
    print("Spider-Man left-side magenta pixel counts by lap:", counts.reshape(-1, 3).tolist(), flush=True)
    assert counts.max() < 100, "Unexpected magenta at a Spider-Man departure"
    return max_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MP4)
    parser.add_argument("--max-bytes", type=int, default=28_000_000)
    args = parser.parse_args()
    if not FFMPEG:
        raise SystemExit("Set IMAGEIO_FFMPEG_EXE to an FFmpeg executable with libx264.")
    with tempfile.TemporaryDirectory(prefix="hero-laps-") as scratch, ExitStack() as stack:
        frames, durations, stops, refs = prepare(stack)
        steps = continuous_frames(stack, frames)
        plan, checks = timeline(durations, stops, steps)
        if args.output.is_file():
            error = verify(args.output, plan, checks, refs)
            print(f"Existing export verified, peak stop RGB error {error:.2f}")
            return
        scratch = Path(scratch)
        cuts = scratch / "timing.ffconcat"
        with cuts.open("w") as script:
            script.write("ffconcat version 1.0\n")
            for i, dt in plan:
                script.write(f"file '/proc/self/fd/{frames[i]}'\noption framerate 100\nduration {dt/100:.2f}\n")
        seconds = sum(dt for _, dt in plan) / 100
        print(f"{len(plan)} video samples, {seconds:.2f} seconds", flush=True)
        # Two-pass encoding budgets the available space for detailed motion.
        budget = min(args.max_bytes, shutil.disk_usage(args.output.parent).free - 2_800_000)
        if budget < 5_000_000:
            raise RuntimeError("Insufficient space to export the video")
        rate = round((budget * .96 - 150_000) * 8 / seconds / 1000)
        base = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                "-safe", "0", "-i", str(cuts), "-an", "-c:v", "libx264", "-preset", "slow",
                "-threads", "5", "-pix_fmt", "yuv420p", "-fps_mode", "vfr", "-enc_time_base", "1:100",
                "-bf", "0", "-g", "120", "-passlogfile", str(scratch / "x264")]
        print("Encoding first pass", flush=True)
        subprocess.run([*base, "-b:v", f"{rate}k", "-pass", "1", "-f", "null", "-"],
                       pass_fds=frames, check=True)
        video = stack.enter_context(os.fdopen(os.memfd_create("accelerated.mp4"), "w+b"))
        handle = f"/proc/self/fd/{video.fileno()}"
        for attempt in range(3):
            video.seek(0)
            video.truncate()
            print(f"Encoding final pass at {rate} kb/s", flush=True)
            subprocess.run([*base, "-b:v", f"{rate}k", "-pass", "2", "-video_track_timescale", "10000",
                            "-movflags", "+faststart", "-f", "mp4", handle],
                           pass_fds=[*frames, video.fileno()], check=True)
            size = os.fstat(video.fileno()).st_size
            print(f"Produced {size} bytes, limit {budget}", flush=True)
            if size <= budget:
                break
            rate = int(rate * budget / size * .96)
        else:
            raise RuntimeError("Encoded movie exceeds the export budget")
        # x264's two-pass motion statistics can occupy several MiB.
        for temporary in scratch.iterdir():
            temporary.unlink()
        print(f"Publishing {size} bytes with {shutil.disk_usage(args.output.parent).free} available", flush=True)
        with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=".accelerated-", suffix=".mp4") as stage:
            os.posix_fallocate(stage.fileno(), 0, size)
            video.seek(0)
            shutil.copyfileobj(video, stage)
            stage.flush()
            error = verify(stage.name, plan, checks, refs)
            # A hard link publishes the already verified file without an extra copy.
            os.link(stage.name, args.output)
        print(json.dumps(dict(laps=1+CONTINUOUS_LAPS, seconds=seconds, bytes=size,
                              frames=len(plan), verified_scene_boundaries=len(checks), max_rgb_error=error)))
if __name__ == "__main__":
    main()
