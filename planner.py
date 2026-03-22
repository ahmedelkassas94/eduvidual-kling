import sys
import uuid
import json
from pathlib import Path
from typing import Any

import env_loader  # noqa: F401 - load .env from project root first

try:
    from rich import print
except ImportError:
    pass  # use built-in print

from llm_client import (
    generate_clip_plan_json,
    generate_ingredients_plan_json,
    generate_narration_script_24s,
    generate_main_script_from_transcription,
    generate_shots_and_ingredients_from_main_script,
    transcribe_audio_with_timestamps,
)
from schemes import ProjectState, StyleBible, ImageFrame, Ingredient, ScriptShot
from tts_client import generate_speech, DEFAULT_VOICE, DEFAULT_MODEL, DEFAULT_SPEED


OUT_DIR = Path("projects")


def save_state(state: ProjectState) -> Path:
    project_dir = OUT_DIR / state.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    out_path = project_dir / "project_state.json"
    out_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return out_path


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


def ai_plan_ingredients(prompt: str, target_duration_s: int = 24) -> ProjectState:
    """
    Narration-first pipeline:
    1. Generate narration script (max 24s when spoken).
    2. Generate audio from script (TTS).
    3. Transcribe audio with exact start/end times per sentence.
    4. Generate main script (video description) aligned to transcription timeline.
    5. Generate shots + ingredients from main script + transcription.
    """
    env_loader.require_env("ANTHROPIC_API_KEY", "Claude required for narration and script generation.")
    env_loader.require_env("OPENAI_API_KEY", "OpenAI required for TTS and Whisper transcription.")

    project_id = str(uuid.uuid4())[:8]
    project_dir = OUT_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Narration script (strictly 24 seconds or less when spoken)
    if getattr(print, "__module__", None) == "rich":
        print("[cyan]Step 1:[/cyan] Generating narration script (max 24s when spoken)...")
    else:
        print("Step 1: Generating narration script (max 24s when spoken)...")
    narration_text = generate_narration_script_24s(prompt)
    (audio_dir / "narration_script.txt").write_text(narration_text, encoding="utf-8")

    # Step 2: TTS → audio file
    if getattr(print, "__module__", None) == "rich":
        print("[cyan]Step 2:[/cyan] Generating audio from script (TTS)...")
    else:
        print("Step 2: Generating audio from script (TTS)...")
    audio_path = audio_dir / "narration_script_audio.mp3"
    generate_speech(
        text=narration_text,
        output_path=audio_path,
        voice=DEFAULT_VOICE,
        model=DEFAULT_MODEL,
        speed=DEFAULT_SPEED,
    )

    # Step 3: Transcribe with timestamps
    if getattr(print, "__module__", None) == "rich":
        print("[cyan]Step 3:[/cyan] Transcribing audio (exact start/end per sentence)...")
    else:
        print("Step 3: Transcribing audio (exact start/end per sentence)...")
    segments = transcribe_audio_with_timestamps(audio_path)
    if not segments:
        raise ValueError("Transcription returned no segments. Check audio file and Whisper API.")
    (audio_dir / "transcription.json").write_text(json.dumps(segments, indent=2), encoding="utf-8")
    total_duration = max(s["end_s"] for s in segments) if segments else float(target_duration_s)

    # Step 4: Main script from transcription (video description aligned to timeline)
    if getattr(print, "__module__", None) == "rich":
        print("[cyan]Step 4:[/cyan] Generating main script (video aligned to transcription timeline)...")
    else:
        print("Step 4: Generating main script (video aligned to transcription timeline)...")
    main_script_15s = generate_main_script_from_transcription(prompt, segments)

    # Step 5: Shots + ingredients from main script + transcription
    if getattr(print, "__module__", None) == "rich":
        print("[cyan]Step 5:[/cyan] Generating shots and ingredients from main script...")
    else:
        print("Step 5: Generating shots and ingredients from main script...")
    data = generate_shots_and_ingredients_from_main_script(main_script_15s, segments, prompt)
    style = StyleBible(**data.get("style_bible", {}))
    ingredients = [Ingredient(**ing) for ing in data.get("ingredients", [])]
    shots_data = data.get("shots", [])
    shot_defaults = {
        "movement_prompt": "",
        "last_frame_t2i_prompt": "",
        "reference_element_names": [],
        "ingredient_names": [],
        "new_ingredient_names": [],
        "narration_text": "",
        "on_screen_text_overlay": "none",
        "assisting_visual_aids": "",
        "i2i_spatial_prompt": "",
        "camera_path": None,
    }

    def _to_str(v: Any, default: str = "") -> str:
        """Coerce LLM output to string (schema expects string, LLM sometimes returns list)."""
        if isinstance(v, str):
            return v
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v) if v else default
        return str(v) if v is not None else default

    def _normalize_shot(s: dict, idx: int) -> dict:
        out = {**shot_defaults, **s, "movement_prompt": s.get("movement_prompt", "")}
        # Keep planner-provided time_range + duration_s (new requirement: 3–15 seconds).
        shot_id = int(s.get("shot_id", idx + 1))
        out["shot_id"] = shot_id
        out["on_screen_text_overlay"] = _to_str(out.get("on_screen_text_overlay"), "none")
        out["assisting_visual_aids"] = _to_str(out.get("assisting_visual_aids"), "")
        # camera_path: preserve dict if valid; else None
        cp = out.get("camera_path")
        if not isinstance(cp, dict) or not cp.get("type"):
            out["camera_path"] = None
        return out

    shots = [ScriptShot(**_normalize_shot(s, idx)) for idx, s in enumerate(shots_data)]
    # Total video duration is the sum of planned shot durations.
    total_video_s = int(sum(s.duration_s for s in shots))

    state = ProjectState(
        project_id=project_id,
        user_prompt=prompt,
        target_duration_s=total_video_s,
        style_bible=style,
        clips=[],
        frames=[],
        full_script_20s="",
        main_script_15s=main_script_15s,
        first_frame_t2i_prompt=data.get("first_frame_t2i_prompt", ""),
        ingredients=ingredients,
        shots=shots,
    )
    return state


def ai_plan(prompt: str, target_duration_s: int) -> ProjectState:
    """Legacy: 20s frames-based plan."""
    _ = min(int(target_duration_s), 20)

    raw = generate_clip_plan_json(prompt, 20)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("LLM did not return valid JSON. Run again.")

    style = StyleBible(**data["style_bible"])
    frames = [ImageFrame(**frame) for frame in data["frames"]]

    state = ProjectState(
        project_id=str(uuid.uuid4())[:8],
        user_prompt=prompt,
        target_duration_s=20,
        style_bible=style,
        clips=[],
        frames=frames,
        full_script_20s=data["full_script_20s"],
    )
    return state


def main():
    env_loader.check_planner_env()  # fail fast with clear message if .env missing key

    if len(sys.argv) < 2:
        print("[red]Usage:[/red] python planner.py \"your prompt\" [target_seconds]")
        print("  target_seconds: 24 = narration-first pipeline (video up to 24s), 20 = legacy frames pipeline")
        raise SystemExit(1)

    prompt = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) >= 3 else 24

    if target == 24 or target == 15:
        # Narration-first pipeline: audio max 24s → transcribe → main script → shots (video follows audio, up to 24s)
        state = ai_plan_ingredients(prompt, target_duration_s=24)
        out_path = save_state(state)
        total = sum(s.duration_s for s in state.shots)
        print(f"[green]Saved:[/green] {out_path}")
        print(f"[cyan]Shots:[/cyan] {len(state.shots)} | [cyan]Ingredients:[/cyan] {len(state.ingredients)} | [cyan]Duration:[/cyan] {total}s")
        print(f"[magenta]Pipeline:[/magenta] narration-first → ingredients + shots (video up to 24s)")
    else:
        if target != 20:
            print(f"[yellow]Legacy:[/yellow] forcing target duration to 20s (you gave {target}s).")
        state = ai_plan(prompt, 20)
        out_path = save_state(state)
        total = sum(f.duration_s for f in state.frames)
        print(f"[green]Saved:[/green] {out_path}")
        print(f"[cyan]Frames:[/cyan] {len(state.frames)} | [cyan]Planned duration:[/cyan] {total}s")
        print(f"[magenta]Pipeline:[/magenta] legacy frames (20s)")


if __name__ == "__main__":
    main()
