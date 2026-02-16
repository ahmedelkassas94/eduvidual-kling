"""
Post-stitch audio revision: revise voiceover from main script, add sound design (SFX),
mix voice + SFX, and mux with the final video.
"""
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from video_actions import _ffmpeg_exe
from tts_client import (
    generate_speech,
    process_audio_for_consistency,
    DEFAULT_VOICE,
    DEFAULT_MODEL,
    DEFAULT_SPEED,
)
from llm_client import generate_audio_revision_json
from sound_effects import generate_sound_effect


def _strip_json_markdown(raw: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` wrapper if present."""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    return s


def _load_state(project_dir: Path) -> Dict[str, Any]:
    state_path = project_dir / "project_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Project state not found: {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _concat_audio_files(ffmpeg: Path, audio_files: List[Path], output_path: Path) -> None:
    if not audio_files:
        raise ValueError("No audio files to concat")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_path.parent / "concat_list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for p in audio_files:
            path_str = str(p.resolve()).replace("'", r"'\''")
            f.write(f"file '{path_str}'\n")
    cmd = [
        str(ffmpeg),
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:a", "libmp3lame",
        "-b:a", "128k",
        "-ar", "44100",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if list_file.exists():
        list_file.unlink()


def _create_silence(ffmpeg: Path, duration_s: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg),
        "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=stereo",
        "-t", str(max(0.1, duration_s)),
        "-q:a", "9",
        "-acodec", "libmp3lame",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _mix_voice_and_sfx(ffmpeg: Path, voice_path: Path, sfx_path: Path, output_path: Path) -> None:
    """Mix voice (full level) and SFX (reduced level) into one track."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Voice at 1.0, SFX at 0.35 so narration is clear
    filter_complex = (
        "[0:a]volume=1.0[a0];"
        "[1:a]volume=0.35[a1];"
        "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0"
    )
    cmd = [
        str(ffmpeg),
        "-y",
        "-i", str(voice_path),
        "-i", str(sfx_path),
        "-filter_complex", filter_complex,
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _mux_video_and_audio(ffmpeg: Path, video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Replace video's audio with the new mixed audio track."""
    # -map 0:v = video from first input, -map 1:a = audio from second input
    cmd = [
        str(ffmpeg),
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-map", "0:v",
        "-map", "1:a",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def revise_and_remix_audio(project_dir: Path) -> Path:
    """
    After stitching: revise voiceover from main script, generate sound-design track,
    mix voice + SFX, and mux with final_video.mp4. Overwrites final_video.mp4.
    Returns path to final_video.mp4.
    """
    state = _load_state(project_dir)
    main_script = (state.get("main_script_15s") or "").strip()
    shots = state.get("shots") or []
    if not main_script or not shots:
        raise ValueError("Project state must contain main_script_15s and shots (ingredients pipeline).")

    final_video = project_dir / "final_video.mp4"
    if not final_video.exists():
        raise FileNotFoundError(f"Stitched video not found: {final_video}. Run stitch_video first.")

    ffmpeg = _ffmpeg_exe()
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    revised_dir = audio_dir / "revised"
    revised_dir.mkdir(parents=True, exist_ok=True)
    sfx_dir = audio_dir / "sfx"
    sfx_dir.mkdir(parents=True, exist_ok=True)

    print("[AUDIO REVISION] Revising voiceover and sound design from main script...")
    raw = generate_audio_revision_json(main_script, shots)
    raw = _strip_json_markdown(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}") from e

    revised_narration = {int(x["shot_id"]): x["revised_text"] for x in data.get("revised_narration", [])}
    sound_design_list = data.get("sound_design", [])
    sound_design_by_shot = {int(x["shot_id"]): x for x in sound_design_list}

    # Build shot order and durations from state (sorted by shot_id)
    shot_order = sorted(
        [(s["shot_id"], int(s.get("duration_s", 3))) for s in shots],
        key=lambda x: x[0],
    )
    voice_files: List[Path] = []
    sfx_files: List[Path] = []

    for shot_id, duration_s in shot_order:
        # 1. Revised voiceover
        text = revised_narration.get(shot_id) or (next((s.get("narration_text") or "" for s in shots if s.get("shot_id") == shot_id), ""))
        if not text:
            text = " "
        vo_path = revised_dir / f"voiceover_{shot_id:03d}.mp3"
        generate_speech(text=text, output_path=vo_path, voice=DEFAULT_VOICE, model=DEFAULT_MODEL, speed=DEFAULT_SPEED)
        process_audio_for_consistency(vo_path, float(duration_s))
        voice_files.append(vo_path)

        # 2. Sound design for this shot
        sd = sound_design_by_shot.get(shot_id)
        desc = (sd.get("description") or "ambient silence") if sd else "ambient silence"
        sfx_path = sfx_dir / f"sfx_{shot_id:03d}.mp3"
        result = generate_sound_effect(desc, float(duration_s), sfx_path)
        if result is None:
            _create_silence(ffmpeg, float(duration_s), sfx_path)
        sfx_files.append(sfx_path)

    print("[AUDIO REVISION] Combining voiceover and SFX tracks...")
    voice_track = project_dir / "temp_voice_track.mp3"
    sfx_track = project_dir / "temp_sfx_track.mp3"
    _concat_audio_files(ffmpeg, voice_files, voice_track)
    _concat_audio_files(ffmpeg, sfx_files, sfx_track)

    mixed_audio = project_dir / "temp_mixed_audio.m4a"
    _mix_voice_and_sfx(ffmpeg, voice_track, sfx_track, mixed_audio)

    print("[AUDIO REVISION] Muxing mixed audio with video...")
    final_new = project_dir / "final_video_new.mp4"
    _mux_video_and_audio(ffmpeg, final_video, mixed_audio, final_new)
    final_new.replace(final_video)

    # Cleanup temp files
    for f in (voice_track, sfx_track, mixed_audio):
        if f.exists():
            f.unlink()

    print(f"[OK] Final video with revised voiceover + sound design: {final_video}")
    return final_video
