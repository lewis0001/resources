"""Locate ffmpeg (system install or the binary bundled with imageio-ffmpeg)
and provide the mux step."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found: install ffmpeg or `pip install imageio-ffmpeg`")


def mux(video: str | Path, audio_wav: str | Path, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-i", str(video), "-i", str(audio_wav),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", str(out)]
    subprocess.run(cmd, check=True)
    return out
