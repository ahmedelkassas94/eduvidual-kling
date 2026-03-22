"""
Shared state for the LangGraph video production pipeline.
TypedDict allows incremental updates and is compatible with LangGraph.
"""
from typing import TypedDict, List, Dict, Any, Optional


class VideoProductionState(TypedDict, total=False):
    """State passed through the video production graph."""

    # Input: raw document or script text
    raw_text: str
    # Optional: path to document (file or dir) for LlamaIndex; used by Retriever when raw_text is empty
    document_path: str

    # Retriever output: key scientific entities and relationships
    extracted_facts: List[Dict[str, Any]]

    # Planner output: scene-by-scene visual script (JSON-serializable)
    scene_blueprints: List[Dict[str, Any]]

    # Stylist output: global visual style guide
    style_config: Dict[str, Any]

    # Visualizer output: scene_id -> image file path
    image_paths: Dict[str, str]

    # Critic output: peer review result
    critic_feedback: Dict[str, Any]
    # critic_feedback must include:
    #   is_accurate: bool, failing_scenes: List[str], feedback: str,
    #   revisions: Dict[str, List[str]] (scene_id -> list of prompt adjustments)

    # Optional: output directory for images (defaults to workflow output dir)
    output_dir: str

    # Optional: limit Critic -> Visualizer regeneration loops
    critic_loop_count: int
    max_critic_loops: int
