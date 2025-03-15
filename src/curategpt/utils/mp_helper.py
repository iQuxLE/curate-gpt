# -----------------------------------------------------------------------------
# Helper function to process all HP term objects concurrently and save them as JSONL.
# -----------------------------------------------------------------------------
import json
import multiprocessing
from pathlib import Path


def save_enhanced_descriptions_jsonl(view, adapter, num_workers=4):
    """
    Process all objects from the ontology view that are HP terms concurrently
    and save the results in a JSONL file in a directory called 'jsonl_enhanced_description'
    (created in the current working directory).
    Each line in the output file contains a JSON object with the term's original_id,
    label, and enhanced_description.
    """
    # Get all objects and filter to HP terms.
    all_objects = list(view.filtered_o())
    hp_objects = [obj for obj in all_objects if obj.get("original_id", "").startswith("HP:")]

    def process_term(obj):
        try:
            # Use the adapter's _text() method to obtain the enhanced description.
            enhanced_text = adapter._text(obj, adapter.text_lookup)
        except Exception as e:
            enhanced_text = ""
        return {
            "original_id": obj.get("original_id", ""),
            "label": obj.get("label", ""),
            "enhanced_description": enhanced_text,
        }

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_term, hp_objects)

    # Create the output directory if it doesn't exist.
    output_dir = Path.cwd() / "jsonl_enhanced_description"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "enhanced_descriptions.jsonl"

    with output_file.open("w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

