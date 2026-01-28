from pathlib import Path
import subprocess

from video_actions import _ffmpeg_exe


def stitch_clips(project_dir: Path):
    clips_dir = project_dir / "clips"
    output_video = project_dir / "final_video.mp4"

    clips = sorted(clips_dir.glob("clip_*.mp4"))
    if not clips:
        raise RuntimeError("No clips found to stitch.")

    print(f"Found {len(clips)} clips to stitch.")

    list_file = clips_dir / "clips.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for clip in clips:
            p = str(clip.resolve()).replace("'", r"'\''")
            f.write(f"file '{p}'\n")

    ffmpeg_path = _ffmpeg_exe()

    cmd_copy = [
        str(ffmpeg_path),
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_video),
    ]

    try:
        subprocess.run(cmd_copy, check=True)
    except subprocess.CalledProcessError:
        cmd_reencode = [
            str(ffmpeg_path),
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_video),
        ]
        subprocess.run(cmd_reencode, check=True)

    print(f"✅ Final video created: {output_video}")


if __name__ == "__main__":
    project_path = Path(
        input("Project folder path (e.g. projects\\7dad8d32): ").strip()
    )
    stitch_clips(project_path)
