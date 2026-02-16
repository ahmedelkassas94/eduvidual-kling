"""
Test the Retriever node in isolation (no full workflow).
Usage: python test_workflow_retriever.py "Your scientific text here"
   or: python test_workflow_retriever.py --document path/to/file.pdf
"""
import json
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodes.retriever import retriever_node

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python test_workflow_retriever.py "Scientific text..."')
        print("   or: python test_workflow_retriever.py --document path/to/file.pdf")
        sys.exit(1)

    raw = ""
    doc_path = ""
    if sys.argv[1] == "--document" and len(sys.argv) >= 3:
        doc_path = sys.argv[2]
    else:
        raw = " ".join(sys.argv[1:])

    state = {"raw_text": raw} if raw else {"document_path": doc_path}
    out = retriever_node(state)
    print("extracted_facts:", json.dumps(out.get("extracted_facts", []), indent=2))
