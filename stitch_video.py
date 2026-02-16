from pathlib import Path
import subprocess
import sys

from video_actions import _ffmpeg_exe


def stitch_clips(project_dir: Path, run_audio_revision: bool = True) -> Path:
    clips_dir = project_dir / "clips"
    audio_dir = project_dir / "audio"
    output_video = project_dir / "final_video.mp4"

    clips = sorted(clips_dir.glob("clip_*.mp4"))
    if not clips:
        raise RuntimeError("No clips found to stitch.")

    print(f"Found {len(clips)} clips to stitch.")

    # Check if we have narration audio files
    audio_files = sorted(audio_dir.glob("narration_*.mp3")) if audio_dir.exists() else []
    has_audio = len(audio_files) > 0
    
    if has_audio:
        print(f"Found {len(audio_files)} narration audio files. Will combine with video.")
    else:
        print("No narration audio files found. Stitching video only.")

    ffmpeg_path = _ffmpeg_exe()

    # Step 1: Combine video clips
    video_list_file = clips_dir / "clips.txt"
    with video_list_file.open("w", encoding="utf-8") as f:
        for clip in clips:
            p = str(clip.resolve()).replace("'", r"'\''")
            f.write(f"file '{p}'\n")

    # Temporary combined video (without audio)
    temp_video = project_dir / "temp_combined_video.mp4"
    
    cmd_video = [
        str(ffmpeg_path),
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(video_list_file),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        str(temp_video),
    ]

    try:
        subprocess.run(cmd_video, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to combine video clips: {e}")

    # Step 2: Combine audio files if available (with crossfading for smooth transitions)
    if has_audio:
        temp_audio = project_dir / "temp_combined_audio.mp3"
        
        try:
            from tts_client import combine_audio_with_crossfade
            
            print("[AUDIO] Combining audio segments with smooth crossfades...")
            combine_audio_with_crossfade(
                audio_files=audio_files,
                output_path=temp_audio,
                crossfade_duration_s=0.3,  # 300ms crossfade for smooth transitions
            )
            print(f"[OK] Audio combined with crossfades: {temp_audio.name}")
        except Exception as e:
            print(f"[WARN] Failed to combine audio with crossfade: {e}")
            print("   Falling back to simple concatenation...")
            
            # Fallback to simple concatenation
            audio_list_file = audio_dir / "audio.txt"
            with audio_list_file.open("w", encoding="utf-8") as f:
                for audio_file in audio_files:
                    p = str(audio_file.resolve()).replace("'", r"'\''")
                    f.write(f"file '{p}'\n")
            
            cmd_audio = [
                str(ffmpeg_path),
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(audio_list_file),
                "-c:a", "libmp3lame",
                "-b:a", "128k",
                str(temp_audio),
            ]
            
            try:
                subprocess.run(cmd_audio, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Failed to combine audio files: {e}")
                print("   Continuing with video-only output...")
                has_audio = False

    # Step 3: Combine video and audio
    if has_audio and temp_audio.exists():
        cmd_final = [
            str(ffmpeg_path),
            "-y",
            "-i", str(temp_video),
            "-i", str(temp_audio),
            "-c:v", "copy",  # Copy video stream (no re-encoding)
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",  # Match the shorter of video/audio
            str(output_video),
        ]
        
        try:
            subprocess.run(cmd_final, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[OK] Final video with narration created: {output_video}")
        except subprocess.CalledProcessError:
            # Fallback: re-encode video if copy fails
            cmd_final_reencode = [
                str(ffmpeg_path),
                "-y",
                "-i", str(temp_video),
                "-i", str(temp_audio),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
                str(output_video),
            ]
            subprocess.run(cmd_final_reencode, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[OK] Final video with narration created (re-encoded): {output_video}")
        
        # Cleanup temp files
        if temp_audio.exists():
            temp_audio.unlink()
    else:
        # No audio, just rename temp video
        temp_video.rename(output_video)
        print(f"[OK] Final video created (no audio): {output_video}")
    
    # Cleanup temp video
    if temp_video.exists():
        temp_video.unlink()
    
    # Cleanup list files
    if video_list_file.exists():
        video_list_file.unlink()
    if audio_dir.exists() and (audio_dir / "audio.txt").exists():
        (audio_dir / "audio.txt").unlink()

    # Post-stitch: add audio. If POST_STITCH_AUDIO_ONLY=1 use GPT+ElevenLabs (video frames + main script);
    # else use per-shot narration revision (revise_and_remix_audio).
    if run_audio_revision:
        try:
            import os
            if os.getenv("POST_STITCH_AUDIO_ONLY", "").strip().lower() in ("1", "true", "yes", "y", "on"):
                from post_stitch_audio_from_video import generate_audio_from_video_and_script
                generate_audio_from_video_and_script(project_dir)
            else:
                from audio_revision import revise_and_remix_audio
                revise_and_remix_audio(project_dir)
        except Exception as e:
            print(f"[WARN] Post-stitch audio skipped: {e}")
            print("   Final video has stitched audio only (or video-only if no narration files).")

    return project_dir / "final_video.mp4"


if __name__ == "__main__":
    run_revision = True
    args = [a for a in sys.argv[1:] if a != "--no-revise-audio"]
    if "--no-revise-audio" in sys.argv:
        run_revision = False
    if len(args) >= 1:
        project_path = Path(args[0])
    else:
        project_path = Path(
            input("Project folder path (e.g. projects\\7dad8d32): ").strip()
        )
    stitch_clips(project_path, run_audio_revision=run_revision)
