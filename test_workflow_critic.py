"""
Test the Critic node in isolation (no full workflow).
Provide a state with image_paths, scene_blueprints, and extracted_facts.
Usage: python test_workflow_critic.py --scene-id scene_1 --image path/to/scene_1.png --description "..." [--facts '[...]']
Or pass a state JSON file: python test_workflow_critic.py --state workflow_output/state.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodes.critic import critic_node

if __name__ == "__main__":
    state = {}
    if "--state" in sys.argv:
        i = sys.argv.index("--state")
        if i + 1 < len(sys.argv):
            with open(sys.argv[i + 1]) as f:
                state = json.load(f)
    else:
        # Minimal state from CLI
        scene_id = "scene_1"
        image_path = ""
        description = "A diagram showing photosynthesis."
        facts = []
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--scene-id" and i + 1 < len(sys.argv):
                scene_id = sys.argv[i + 1]
                i += 2
                continue
            if sys.argv[i] == "--image" and i + 1 < len(sys.argv):
                image_path = sys.argv[i + 1]
                i += 2
                continue
            if sys.argv[i] == "--description" and i + 1 < len(sys.argv):
                description = sys.argv[i + 1]
                i += 2
                continue
            if sys.argv[i] == "--facts" and i + 1 < len(sys.argv):
                facts = json.loads(sys.argv[i + 1])
                i += 2
                continue
            i += 1
        if not image_path or not Path(image_path).exists():
            print("Usage: python test_workflow_critic.py --image path/to/image.png --description \"...\" [--facts '[]']")
            print("   or: python test_workflow_critic.py --state path/to/state.json")
            sys.exit(1)
        state = {
            "image_paths": {scene_id: image_path},
            "scene_blueprints": [{"scene_id": scene_id, "description": description, "visual_prompt": description}],
            "extracted_facts": facts,
        }

    out = critic_node(state)
    print("critic_feedback:", json.dumps(out.get("critic_feedback"), indent=2))
