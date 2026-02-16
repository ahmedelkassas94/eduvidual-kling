"""
Retriever Node: Use LlamaIndex to parse the input document and extract
key scientific entities and relationships. Can be run in isolation for testing.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


def _extract_facts_via_llm(raw_text: str) -> List[Dict[str, Any]]:
    """Use OpenAI (from .env) to extract structured facts from text."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for retriever. pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()

    prompt = f"""Analyze the following scientific or educational document and extract key facts as structured data.

For each important fact, identify:
1. entities (concepts, objects, processes, quantities),
2. relationships between them (causes, describes, part-of, etc.),
3. short label and one-sentence description.

Return a JSON array of objects. Each object must have: "entities" (list of strings), "relationship" (string), "description" (string).
Example: [{{"entities": ["photosynthesis", "chlorophyll"], "relationship": "requires", "description": "Photosynthesis requires chlorophyll to capture light."}}]

Document:
---
{raw_text[:12000]}
---

Return ONLY the JSON array, no markdown or extra text."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = (response.choices[0].message.content or "").strip()
    # Strip markdown code block if present
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [{"raw": content, "entities": [], "relationship": "", "description": content[:200]}]


def _load_document_llamaindex(file_path: str) -> str:
    """Load and parse document with LlamaIndex; return full text."""
    try:
        from llama_index.core import SimpleDirectoryReader
    except ImportError:
        raise ImportError("llama-index package required. pip install llama-index")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document path not found: {file_path}")

    if path.is_file():
        reader = SimpleDirectoryReader(input_files=[str(path)])
    else:
        reader = SimpleDirectoryReader(input_dir=file_path)
    docs = reader.load_data()
    return "\n\n".join(doc.get_content() for doc in docs)


def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retriever node: parse input document (LlamaIndex) and extract scientific facts (LLM).
    Expects state["raw_text"] to be set, OR state["document_path"] to load via LlamaIndex.
    Updates state with extracted_facts.
    """
    raw_text = state.get("raw_text", "")
    document_path = state.get("document_path")

    if document_path and not raw_text:
        raw_text = _load_document_llamaindex(document_path)

    if not raw_text or not raw_text.strip():
        return {"raw_text": raw_text, "extracted_facts": []}

    facts = _extract_facts_via_llm(raw_text)
    return {"raw_text": raw_text, "extracted_facts": facts}
