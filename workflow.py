"""
LangGraph workflow for video production: Retriever -> Planner -> Stylist -> Visualizer -> Critic
with a conditional edge from Critic back to Visualizer when is_accurate is False.
All API keys (OpenAI, Gemini) are read from .env.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from workflow_state import VideoProductionState
from nodes.retriever import retriever_node
from nodes.planner import planner_node
from nodes.stylist import stylist_node
from nodes.visualizer import visualizer_node
from nodes.critic import critic_node

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError("langgraph required. pip install langgraph")


def _critic_route(state: VideoProductionState) -> str:
    """Route after Critic: back to Visualizer to regenerate failing scenes, or END."""
    feedback = state.get("critic_feedback") or {}
    if feedback.get("is_accurate", True):
        return END
    max_loops = state.get("max_critic_loops") or 3
    loop_count = state.get("critic_loop_count") or 0
    if loop_count >= max_loops:
        return END
    return "visualizer"


def build_video_production_graph(output_dir: str = "workflow_output") -> StateGraph:
    """
    Build the 5-agent LangGraph:
    Retriever -> Planner -> Stylist -> Visualizer -> Critic -> (conditional) Visualizer | END
    """
    graph = StateGraph(VideoProductionState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("planner", planner_node)
    graph.add_node("stylist", stylist_node)
    graph.add_node("visualizer", visualizer_node)
    graph.add_node("critic", critic_node)

    graph.add_edge("retriever", "planner")
    graph.add_edge("planner", "stylist")
    graph.add_edge("stylist", "visualizer")
    graph.add_edge("visualizer", "critic")
    graph.add_conditional_edges("critic", _critic_route)

    graph.set_entry_point("retriever")
    return graph


def run_workflow(
    raw_text: str = "",
    document_path: str = "",
    output_dir: str = "workflow_output",
    max_critic_loops: int = 3,
) -> VideoProductionState:
    """
    Run the full pipeline. Provide either raw_text or document_path (file/dir for LlamaIndex).
    If document_path is set and raw_text is empty, Retriever will load the document via LlamaIndex.
    """
    graph = build_video_production_graph(output_dir=output_dir)
    app = graph.compile()

    initial: VideoProductionState = {
        "output_dir": output_dir,
        "max_critic_loops": max_critic_loops,
    }
    if raw_text:
        initial["raw_text"] = raw_text
    if document_path:
        initial["document_path"] = document_path

    if not initial.get("raw_text") and not initial.get("document_path"):
        raise ValueError("Provide either raw_text or document_path")

    state = initial
    for event in app.stream(state):
        for node_name, node_state in event.items():
            state = dict(node_state)
    return state


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run video production LangGraph workflow")
    parser.add_argument("--text", type=str, help="Raw scientific/educational text input")
    parser.add_argument("--document", type=str, help="Path to document (file or dir) for LlamaIndex")
    parser.add_argument("--output", type=str, default="workflow_output", help="Output directory for images")
    parser.add_argument("--max-critic-loops", type=int, default=3, help="Max regeneration loops after critic")
    args = parser.parse_args()

    if not args.text and not args.document:
        parser.error("Provide either --text or --document")

    final = run_workflow(
        raw_text=args.text or "",
        document_path=args.document or "",
        output_dir=args.output,
        max_critic_loops=args.max_critic_loops,
    )
    print("Final state keys:", list(final.keys()))
    print("critic_feedback:", json.dumps(final.get("critic_feedback"), indent=2))
    print("image_paths:", final.get("image_paths"))
