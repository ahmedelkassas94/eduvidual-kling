"""
Stitch audio pipeline (steps 2–3):
- Send stitched video + original script to Gemini; Gemini analyzes the video, compares to the plan,
  and outputs a detailed prompt/spec for ElevenLabs (narration + SFX with timing and delivery).
- Generate audio from that spec via ElevenLabs, then mux with the stitched video.
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from video_actions import _ffmpeg_exe

load_dotenv()

# ElevenLabs default voice
DEFAULT_ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

GEMINI_AUDIO_SPEC_VFX_ONLY_SYSTEM = """You are an expert sound designer for educational videos. You are given:
1. The ACTUAL stitched video (what was really produced).
2. The ORIGINAL script (for reference to understand what is shown).

Your task is to ANALYZE THE VIDEO and produce a precise audio specification for sound effects ONLY. NO narration, NO voiceover.
Output ONLY sound effects and VFX that match the visuals: camera movements (whoosh on zoom, subtle rumble on pan), object motion (swoosh, flow, creak), ambient (soft hum, nature, space), and movement-specific sounds.

Output a JSON object with this exact structure (no markdown):
{
  "narration_segments": [],
  "sound_design": [
    {
      "start_s": 0.0,
      "end_s": 5.0,
      "description": "Detailed SFX description for ElevenLabs, e.g. soft whoosh as camera zooms in, subtle ambient hum, gentle crackle."
    }
  ]
}

Rules:
- narration_segments: MUST be an empty array. No narration.
- sound_design: 4–10 segments covering the full video. Describe VFX: whooshes, rumbles, swooshes, ambient, movement sounds. Match timing to visual events (zooms, pans, object motion, scene changes).
- Each description should be specific for ElevenLabs sound generation (e.g. "subtle whoosh as diagram zooms in", "low rumble as magma rises", "soft ambient sci-fi hum")."""


GEMINI_AUDIO_SPEC_SYSTEM = """You are an expert audio director for educational videos. You are given:
1. The ACTUAL stitched video (what was really produced).
2. The ORIGINAL long-form script (what was planned) — for reference only.

Your task is to ANALYZE THE VIDEO ITSELF and produce a precise audio specification for ElevenLabs that will be used to generate narration and sound effects. The output must be based on what you actually see in the video, not on the script alone. Compare the original plan to what appears in the video; if something differs (e.g. a shot was shortened, a visual is different), adjust your narration and timing to match the video.

Output a JSON object with this exact structure (no markdown, no code fence):

{
  "narration_segments": [
    {
      "start_s": 0.0,
      "end_s": 3.5,
      "text": "Exact words to be spoken in this segment.",
      "delivery_notes": "How to say it: e.g. calm, emphasis on key term, slightly faster, pause after X."
    }
  ],
  "sound_design": [
    {
      "start_s": 0.0,
      "end_s": 15.0,
      "description": "Short SFX/ambient description for ElevenLabs sound generation, e.g. soft ambient, subtle whoosh, light nature."
    }
  ]
}

Rules:
- narration_segments: Cover the full video duration. Each segment has start_s, end_s (in seconds), the exact text to speak, and delivery_notes for tone/pace/emphasis.
- Base timing and content on what you SEE in the video (scene changes, visuals). If the video differs from the script, describe what the video shows and write narration that matches it.
- sound_design: 2–6 segments; short descriptions for ambient or SFX. Can span the whole video or key moments.
- Total duration of the video in seconds: use the actual length (you can infer from the number of shots/scenes). Ensure narration segments do not exceed that duration."""


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


def _get_video_duration_s(video_path: Path) -> float:
    path = Path(video_path).resolve()
    if not path.exists():
        return 15.0
    ffmpeg = _ffmpeg_exe()
    cmd = [
        str(ffmpeg), "-i", str(path),
        "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=path.parent)
    if result.returncode != 0:
        return 15.0
    try:
        return float((result.stdout or "").strip() or "15.0")
    except ValueError:
        return 15.0


def gemini_analyze_video_and_script(
    video_path: Path,
    main_script: str,
    video_duration_s: float,
    vfx_only: bool = False,
) -> Dict[str, Any]:
    """
    Send the stitched video (as key frames) and original script to Gemini.
    Returns JSON with narration_segments and sound_design, based on analysis of the video.
    When vfx_only=True, produces sound_design only (no narration).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")
    client = genai.Client(api_key=api_key)
    model = (os.getenv("REVISER_MODEL") or os.getenv("PLANNER_MODEL") or "gemini-2.5-flash").strip()

    # Prefer native video upload when available; else key frames
    try:
        uploaded = client.files.upload(file=str(video_path))
        system = GEMINI_AUDIO_SPEC_VFX_ONLY_SYSTEM if vfx_only else GEMINI_AUDIO_SPEC_SYSTEM
        user_content = (
            f"The attached file is the full stitched video (duration ~{video_duration_s:.1f}s).\n\n"
            "ORIGINAL SCRIPT (for reference only — the video may differ):\n"
            f"\"\"\"\n{main_script[:4000]}\n\"\"\"\n\n"
            + (
                "Analyze the video. Output the JSON with sound_design ONLY (narration_segments must be empty). "
                "Describe VFX/sound effects for camera movements, zooms, object motion, ambient."
                if vfx_only
                else "Analyze the video and compare it to this script. Output the JSON audio spec "
                "that matches what is actually shown in the video (timing, content, tone)."
            )
        )
        response = client.models.generate_content(
            model=model,
            contents=[uploaded, user_content],
            config=GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
            ),
        )
        text = getattr(response, "text", None)
        if not text and getattr(response, "candidates", None) and response.candidates:
            c = response.candidates[0]
            parts = getattr(getattr(c, "content", None), "parts", None) or []
            if parts:
                text = getattr(parts[0], "text", None)
        if text and text.strip():
            return json.loads(_strip_json_markdown(text))
    except Exception as e:
        print(f"[WARN] Gemini with video file failed: {e}. Using key frames...")

    return _gemini_analyze_key_frames(video_path, main_script, video_duration_s, client, model, vfx_only=vfx_only)


def _gemini_analyze_key_frames(
    video_path: Path,
    main_script: str,
    video_duration_s: float,
    client: genai.Client,
    model: str,
    vfx_only: bool = False,
) -> Dict[str, Any]:
    """Fallback: extract key frames and send images to Gemini."""
    from PIL import Image
    frames_dir = video_path.parent / "audio" / "key_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_exe()
    num_frames = 8
    interval = max(1.0, video_duration_s / num_frames)
    cmd = [
        str(ffmpeg), "-y", "-i", str(video_path),
        "-vf", f"fps=1/{interval},scale=640:-1", "-vframes", str(num_frames),
        str(frames_dir / "frame_%03d.png"),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frame_paths = sorted(frames_dir.glob("frame_*.png"))[:num_frames]
    if not frame_paths:
        raise RuntimeError("No key frames extracted")

    system = GEMINI_AUDIO_SPEC_VFX_ONLY_SYSTEM if vfx_only else GEMINI_AUDIO_SPEC_SYSTEM
    content_parts = [
        f"The following {len(frame_paths)} images are key frames from the video (duration ~{video_duration_s:.1f}s).\n\n"
        "ORIGINAL SCRIPT (reference only):\n\"\"\"\n" + main_script[:4000] + "\n\"\"\"\n\n"
        + (
            "Analyze what the video shows. Output JSON with sound_design ONLY (narration_segments empty). Describe VFX for movements."
            if vfx_only
            else "Analyze what the video shows and output the JSON audio spec (narration_segments + sound_design) that matches the video."
        ),
    ]
    for fp in frame_paths:
        content_parts.append(Image.open(fp))

    response = client.models.generate_content(
        model=model,
        contents=content_parts,
        config=GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
        ),
    )
    text = getattr(response, "text", None)
    if not text and getattr(response, "candidates", None) and response.candidates:
        c = response.candidates[0]
        parts = getattr(getattr(c, "content", None), "parts", None) or []
        if parts:
            text = getattr(parts[0], "text", None)
    if not text or not text.strip():
        raise RuntimeError("Gemini returned empty response")
    return json.loads(_strip_json_markdown(text))


def elevenlabs_tts(text: str, output_path: Path, voice_id: Optional[str] = None) -> Path:
    """Generate speech with ElevenLabs. Returns output_path."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not found in .env")
    voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID") or DEFAULT_ELEVENLABS_VOICE_ID
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import requests
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {"text": text, "model_id": "eleven_multilingual_v2"}
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    output_path.write_bytes(r.content)
    return output_path


def _create_silence(ffmpeg: Path, duration_s: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg), "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(max(0.1, duration_s)), "-q:a", "9", "-acodec", "libmp3lame", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _concat_audio_files(ffmpeg: Path, audio_files: List[Path], output_path: Path) -> None:
    if not audio_files:
        raise ValueError("No audio files to concat")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_path.parent / "concat_audio_list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for p in audio_files:
            f.write(f"file '{p.resolve()}'\n")
    cmd = [
        str(ffmpeg), "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if list_file.exists():
        list_file.unlink()


def _mix_voice_and_sfx(ffmpeg: Path, voice_path: Path, sfx_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = "[0:a]volume=1.0[a0];[1:a]volume=0.35[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=0"
    cmd = [
        str(ffmpeg), "-y", "-i", str(voice_path), "-i", str(sfx_path),
        "-filter_complex", filter_complex, "-c:a", "aac", "-b:a", "128k", "-ar", "44100", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_audio_from_gemini_spec(
    spec: Dict[str, Any],
    video_duration_s: float,
    project_dir: Path,
) -> Path:
    """
    Turn Gemini's JSON spec into one combined audio file: narration (with optional silence padding for timing) + SFX.
    """
    ffmpeg = _ffmpeg_exe()
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    narration_segments = spec.get("narration_segments") or []
    sound_design = spec.get("sound_design") or []

    # Build narration track: for each segment, generate TTS then pad with silence to align start_s
    voice_segments = []
    current_time = 0.0
    for seg in narration_segments:
        start_s = float(seg.get("start_s", 0))
        end_s = float(seg.get("end_s", start_s + 3))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        # Silence from current_time to start_s
        if start_s > current_time:
            silence_path = audio_dir / f"silence_{len(voice_segments):03d}.mp3"
            _create_silence(ffmpeg, start_s - current_time, silence_path)
            voice_segments.append(silence_path)
        # TTS for this segment
        seg_path = audio_dir / f"narration_seg_{len(voice_segments):03d}.mp3"
        try:
            elevenlabs_tts(text, seg_path)
            voice_segments.append(seg_path)
        except Exception as e:
            print(f"[WARN] TTS failed for segment: {e}")
            _create_silence(ffmpeg, end_s - start_s, seg_path)
            voice_segments.append(seg_path)
        current_time = end_s

    # Trailing silence if narration ends before video end
    if current_time < video_duration_s and voice_segments:
        silence_path = audio_dir / "silence_trailing.mp3"
        _create_silence(ffmpeg, video_duration_s - current_time, silence_path)
        voice_segments.append(silence_path)

    if not voice_segments:
        # No narration: full silence for voice track
        voice_path = audio_dir / "stitch_voice_only.mp3"
        _create_silence(ffmpeg, video_duration_s, voice_path)
    else:
        voice_path = audio_dir / "stitch_voice_only.mp3"
        _concat_audio_files(ffmpeg, voice_segments, voice_path)
        # Trim or pad to video duration
        probe_cmd = [str(ffmpeg), "-i", str(voice_path), "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"]
        r = subprocess.run(probe_cmd, capture_output=True, text=True)
        try:
            dur = float(r.stdout.strip() or 0)
        except Exception:
            dur = video_duration_s
        if dur > video_duration_s:
            trimmed = audio_dir / "stitch_voice_trimmed.mp3"
            cmd = [str(ffmpeg), "-y", "-i", str(voice_path), "-t", str(video_duration_s), "-c", "copy", str(trimmed)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            voice_path = trimmed
        elif dur < video_duration_s:
            padded = audio_dir / "stitch_voice_padded.mp3"
            silence_tail = audio_dir / "silence_tail.mp3"
            _create_silence(ffmpeg, video_duration_s - dur, silence_tail)
            _concat_audio_files(ffmpeg, [voice_path, silence_tail], padded)
            voice_path = padded

    # SFX track
    sfx_dir = audio_dir / "sfx_stitch"
    sfx_dir.mkdir(parents=True, exist_ok=True)
    sfx_files = []
    if sound_design:
        try:
            from sound_effects import generate_sound_effect
            for i, seg in enumerate(sound_design):
                start_s = float(seg.get("start_s", 0))
                end_s = float(seg.get("end_s", start_s + 3))
                desc = (seg.get("description") or "ambient").strip()
                dur = max(0.5, end_s - start_s)
                p = sfx_dir / f"sfx_{i:03d}.mp3"
                if generate_sound_effect(desc, dur, p):
                    sfx_files.append(p)
                else:
                    _create_silence(ffmpeg, dur, p)
                    sfx_files.append(p)
        except Exception as e:
            print(f"[WARN] SFX generation failed: {e}")
    if not sfx_files:
        sfx_path = audio_dir / "stitch_sfx_silent.mp3"
        _create_silence(ffmpeg, video_duration_s, sfx_path)
    else:
        sfx_path = audio_dir / "stitch_sfx_combined.mp3"
        _concat_audio_files(ffmpeg, sfx_files, sfx_path)
        # Pad/trim SFX to video duration if needed (simplified: just use as-is or pad)
        probe_cmd = [str(ffmpeg), "-i", str(sfx_path), "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"]
        r = subprocess.run(probe_cmd, capture_output=True, text=True)
        try:
            dur = float(r.stdout.strip() or 0)
        except Exception:
            dur = video_duration_s
        if dur < video_duration_s:
            silence_tail = audio_dir / "sfx_silence_tail.mp3"
            _create_silence(ffmpeg, video_duration_s - dur, silence_tail)
            padded_sfx = audio_dir / "stitch_sfx_padded.mp3"
            _concat_audio_files(ffmpeg, [sfx_path, silence_tail], padded_sfx)
            sfx_path = padded_sfx

    # VFX only: use SFX track as final audio (no narration)
    if not narration_segments:
        return sfx_path

    # Mix voice + SFX
    combined = project_dir / "temp_stitch_combined_audio.m4a"
    _mix_voice_and_sfx(ffmpeg, voice_path, sfx_path, combined)
    return combined


def analyze_video_and_generate_audio(
    project_dir: Path,
    stitched_video_path: Path,
    vfx_only: bool = False,
) -> Path:
    """
    Step 2 + 3: Send stitched video + script to Gemini → get spec → generate audio with ElevenLabs → return path to combined audio.
    When vfx_only=True, produces VFX/sound effects only (no narration).
    """
    state = _load_state(project_dir)
    main_script = (state.get("main_script_15s") or "").strip()
    if not main_script:
        raise ValueError("project_state.json must contain main_script_15s")

    video_duration_s = _get_video_duration_s(stitched_video_path)
    print(f"[STITCH AUDIO] Video duration: {video_duration_s:.1f}s")
    print(f"[STITCH AUDIO] Sending video + script to Gemini ({'VFX only' if vfx_only else 'narration + SFX'})...")
    spec = gemini_analyze_video_and_script(stitched_video_path, main_script, video_duration_s, vfx_only=vfx_only)
    spec_path = project_dir / "audio" / "gemini_audio_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with spec_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(f"[STITCH AUDIO] Spec saved to {spec_path.name}")

    print(f"[STITCH AUDIO] Generating {'VFX/SFX' if vfx_only else 'narration and SFX'} with ElevenLabs...")
    combined_audio = generate_audio_from_gemini_spec(spec, video_duration_s, project_dir)
    print(f"[OK] Combined audio: {combined_audio}")
    return combined_audio
