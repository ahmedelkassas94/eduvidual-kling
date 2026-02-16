import sys
import uuid
import json
from pathlib import Path

from rich import print

from llm_client import generate_clip_plan_json, generate_ingredients_plan_json
from schemes import ProjectState, StyleBible, ImageFrame, Ingredient, ScriptShot


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


def ai_plan_ingredients(prompt: str, target_duration_s: int = 15) -> ProjectState:
    """New architecture: 15s shot-by-shot script + named ingredients."""
    raw = generate_ingredients_plan_json(prompt, target_duration_s)
    raw = _strip_json_markdown(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("LLM did not return valid JSON. Run again.")

    style = StyleBible(**data["style_bible"])
    ingredients = [Ingredient(**ing) for ing in data["ingredients"]]
    shots = [ScriptShot(**s) for s in data["shots"]]

    state = ProjectState(
        project_id=str(uuid.uuid4())[:8],
        user_prompt=prompt,
        target_duration_s=15,
        style_bible=style,
        clips=[],
        frames=[],
        full_script_20s="",
        main_script_15s=data.get("main_script_15s", ""),
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
    if len(sys.argv) < 2:
        print("[red]Usage:[/red] python planner.py \"your prompt\" [target_seconds]")
        print("  target_seconds: 15 = new ingredients pipeline, 20 = legacy frames pipeline")
        raise SystemExit(1)

    prompt = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) >= 3 else 15

    if target == 15:
        state = ai_plan_ingredients(prompt, 15)
        out_path = save_state(state)
        total = sum(s.duration_s for s in state.shots)
        print(f"[green]Saved:[/green] {out_path}")
        print(f"[cyan]Shots:[/cyan] {len(state.shots)} | [cyan]Ingredients:[/cyan] {len(state.ingredients)} | [cyan]Duration:[/cyan] {total}s")
        print(f"[magenta]Pipeline:[/magenta] ingredients + shots (15s)")
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
