"""
Post-stitch audio from video: send stitched video (as key frames) + main script to ChatGPT,
get narration + sound-design prompts, generate audio via ElevenLabs, then mux with video.
No per-shot narration; one analysis of the full video and one coherent audio track.
"""
import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from video_actions import _ffmpeg_exe

load_dotenv()

# ElevenLabs default voice (Rachel); override with ELEVENLABS_VOICE_ID
DEFAULT_ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


def _load_state(project_dir: Path) -> Dict[str, Any]:
    state_path = project_dir / "project_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Project state not found: {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _strip_json_markdown(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    return s


def extract_key_frames(video_path: Path, out_dir: Path, num_frames: int = 8) -> List[Path]:
    """Extract N key frames at regular intervals. Returns list of image paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_exe()
    # One frame every (15/num_frames) seconds for ~15s video; scale down for smaller upload
    interval = max(1.5, 15.0 / num_frames)
    cmd = [
        str(ffmpeg),
        "-y",
        "-i", str(video_path),
        "-vf", f"fps=1/{interval},scale=640:-1",
        "-vframes", str(num_frames),
        str(out_dir / "frame_%03d.png"),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = sorted(out_dir.glob("frame_*.png"))
    return frames[:num_frames]


def gpt_analyze_video_and_script(frame_paths: List[Path], main_script: str, video_duration_s: float = 15) -> Dict[str, Any]:
    """
    Send key frames + main script to GPT-4 Vision. Returns { narration_script, sound_design }.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL") or "gpt-4o"

    content: List[Any] = [
        {
            "type": "text",
            "text": (
                f"You are an expert scriptwriter for educational videos. You are given:\n"
                f"1. The MAIN SCRIPT (shot-by-shot description) that was used to create this video.\n"
                f"2. Key frames extracted from the actual stitched video (about {video_duration_s}s total).\n\n"
                "Compare what the video shows (from the frames) with the main script. "
                "Your task is to produce:\n\n"
                "A) narration_script: A single, coherent narration script that EXPLAINS THE CONCEPT for university students. "
                "Do NOT describe what we see (e.g. 'we see a leaf'). Do explain the educational content (e.g. 'photosynthesis begins when light hits the chloroplasts'). "
                "Write in a clear, academic tone. Length: roughly 2-3 words per second so it fits the video duration.\n\n"
                "B) sound_design: Optional list of segments for ambient/SFX. Each item: { \"start_s\": 0, \"end_s\": 3, \"description\": \"gentle wind, leaves\" }. "
                "Keep 3-6 segments max; descriptions should be short.\n\n"
                "Output ONLY valid JSON, no markdown:\n"
                "{\n  \"narration_script\": \"Full narration text here.\",\n  \"sound_design\": [ { \"start_s\": 0, \"end_s\": 3, \"description\": \"...\" } ]\n}"
            ),
        },
    ]
    for i, fp in enumerate(frame_paths):
        if not fp.exists():
            continue
        with open(fp, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
        })

    content.append({
        "type": "text",
        "text": f"MAIN SCRIPT (reference for the video):\n\n{main_script}",
    })

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=2000,
    )
    raw = resp.choices[0].message.content or "{}"
    raw = _strip_json_markdown(raw)
    return json.loads(raw)


def elevenlabs_tts(text: str, output_path: Path, voice_id: Optional[str] = None, model_id: str = "eleven_multilingual_v2") -> Path:
    """Generate speech from text using ElevenLabs TTS. Returns output_path."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not found in .env")
    voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID") or DEFAULT_ELEVENLABS_VOICE_ID
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import requests
    except ImportError:
        raise RuntimeError("Install requests to use ElevenLabs TTS")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {"text": text, "model_id": model_id}
    # Chunk if too long (e.g. 5000 chars limit per request)
    if len(text) > 4500:
        parts = []
        for i in range(0, len(text), 4500):
            chunk = text[i : i + 4500]
            r = requests.post(url, json={"text": chunk, "model_id": model_id}, headers=headers, timeout=120)
            r.raise_for_status()
            parts.append(r.content)
        # Concatenate MP3s with ffmpeg
        ffmpeg = _ffmpeg_exe()
        list_f = output_path.parent / "tts_concat_list.txt"
        for idx, audio_bytes in enumerate(parts):
            p = output_path.parent / f"tts_part_{idx}.mp3"
            p.write_bytes(audio_bytes)
            with open(list_f, "a", encoding="utf-8") as f:
                f.write(f"file '{p.resolve()}'\n")
        cmd = [str(ffmpeg), "-y", "-f", "concat", "-safe", "0", "-i", str(list_f), "-c", "copy", str(output_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for p in output_path.parent.glob("tts_part_*.mp3"):
            p.unlink(missing_ok=True)
        if list_f.exists():
            list_f.unlink(missing_ok=True)
        return output_path

    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    output_path.write_bytes(r.content)
    return output_path


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
        str(ffmpeg), "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if list_file.exists():
        list_file.unlink()


def _create_silence(ffmpeg: Path, duration_s: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg), "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(max(0.1, duration_s)), "-q:a", "9", "-acodec", "libmp3lame", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _mix_voice_and_sfx(ffmpeg: Path, voice_path: Path, sfx_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = "[0:a]volume=1.0[a0];[1:a]volume=0.35[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=0"
    cmd = [
        str(ffmpeg), "-y", "-i", str(voice_path), "-i", str(sfx_path),
        "-filter_complex", filter_complex, "-c:a", "aac", "-b:a", "128k", "-ar", "44100", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _mux_video_and_audio(ffmpeg: Path, video_path: Path, audio_path: Path, output_path: Path) -> None:
    cmd = [
        str(ffmpeg), "-y", "-i", str(video_path), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-map", "0:v", "-map", "1:a", "-shortest", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_audio_from_video_and_script(project_dir: Path) -> Path:
    """
    After stitching: extract key frames from final_video.mp4, send frames + main_script to GPT,
    get narration_script + sound_design, generate audio via ElevenLabs TTS (and optional SFX),
    then mux with video. Overwrites final_video.mp4.
    """
    state = _load_state(project_dir)
    main_script = (state.get("main_script_15s") or "").strip()
    if not main_script:
        raise ValueError("Project state must contain main_script_15s")

    final_video = project_dir / "final_video.mp4"
    if not final_video.exists():
        raise FileNotFoundError(f"Stitched video not found: {final_video}. Run stitch_video first.")

    ffmpeg = _ffmpeg_exe()
    frames_dir = project_dir / "audio" / "key_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("[POST-STITCH AUDIO] Extracting key frames from stitched video...")
    frame_paths = extract_key_frames(final_video, frames_dir, num_frames=8)
    if not frame_paths:
        raise RuntimeError("No frames extracted from video")

    print("[POST-STITCH AUDIO] Sending video frames + main script to ChatGPT for analysis...")
    video_duration = 15.0  # could probe video; default 15
    data = gpt_analyze_video_and_script(frame_paths, main_script, video_duration)
    narration_script = (data.get("narration_script") or "").strip()
    if not narration_script:
        raise ValueError("GPT did not return narration_script")

    voice_path = project_dir / "audio" / "post_stitch_narration.mp3"
    print("[POST-STITCH AUDIO] Generating narration with ElevenLabs TTS...")
    elevenlabs_tts(narration_script, voice_path)

    sound_design = data.get("sound_design") or []
    sfx_path = project_dir / "temp_sfx_track.mp3"
    if sound_design:
        try:
            from sound_effects import generate_sound_effect
            sfx_files = []
            for seg in sound_design:
                start_s = float(seg.get("start_s", 0))
                end_s = float(seg.get("end_s", start_s + 3))
                desc = (seg.get("description") or "ambient").strip()
                dur = max(0.5, end_s - start_s)
                p = project_dir / "audio" / "sfx" / f"sfx_{len(sfx_files):03d}.mp3"
                p.parent.mkdir(parents=True, exist_ok=True)
                if generate_sound_effect(desc, dur, p):
                    sfx_files.append(p)
                else:
                    _create_silence(ffmpeg, dur, p)
                    sfx_files.append(p)
            if sfx_files:
                _concat_audio_files(ffmpeg, sfx_files, sfx_path)
            else:
                _create_silence(ffmpeg, video_duration, sfx_path)
        except Exception as e:
            print(f"[WARN] Sound design failed: {e}")
            _create_silence(ffmpeg, video_duration, sfx_path)
    else:
        _create_silence(ffmpeg, video_duration, sfx_path)

    mixed_audio = project_dir / "temp_mixed_audio.m4a"
    _mix_voice_and_sfx(ffmpeg, voice_path, sfx_path, mixed_audio)
    if sfx_path.exists():
        sfx_path.unlink(missing_ok=True)

    print("[POST-STITCH AUDIO] Muxing audio with video...")
    final_new = project_dir / "final_video_new.mp4"
    _mux_video_and_audio(ffmpeg, final_video, mixed_audio, final_new)
    final_new.replace(final_video)
    if mixed_audio.exists():
        mixed_audio.unlink(missing_ok=True)

    print(f"[OK] Final video with GPT+ElevenLabs audio: {final_video}")
    return final_video
