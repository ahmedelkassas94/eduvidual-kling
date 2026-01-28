import sys
import uuid
import json
from pathlib import Path

from rich import print

from llm_client import generate_clip_plan_json
from schemes import ProjectState, StyleBible, ImageFrame


OUT_DIR = Path("projects")


def save_state(state: ProjectState) -> Path:
    project_dir = OUT_DIR / state.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    out_path = project_dir / "project_state.json"
    out_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def ai_plan(prompt: str, target_duration_s: int) -> ProjectState:
    # Trial mode: hard-cap total to 20 seconds and force 2 clips.
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
        raise SystemExit(1)

    prompt = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) >= 3 else 20

    if target != 20:
        print(f"[yellow]Trial mode:[/yellow] forcing target duration to 20s (you gave {target}s).")
    target = 20

    state = ai_plan(prompt, target)
    out_path = save_state(state)

    total = sum(f.duration_s for f in state.frames)
    print(f"[green]Saved:[/green] {out_path}")
    print(f"[cyan]Frames:[/cyan] {len(state.frames)} | [cyan]Planned duration:[/cyan] {total}s")
    print(f"[magenta]Target (trial):[/magenta] 20s")


if __name__ == "__main__":
    main()
