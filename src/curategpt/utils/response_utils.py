from pathlib import Path
from typing import Dict, Optional, List

from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message

class Body(BaseModel):
    choices: List[Choice]

class Response(BaseModel):
    body: Body

class LLMResponse(BaseModel):
    term_id: str = Field(..., alias="custom_id")
    response: Optional[Response] = None
    error: Optional[str] = None

    @property
    def definition(self) -> Optional[str]:
        if self.response and self.response.body.choices:
            return self.response.body.choices[0].message.content
        return None


def load_cache(
    directory: Path,
    file_limit: int = None,
    line_limit: int = None,
    specific_file: Optional[Path] = None
) -> Dict[str, str]:
    """
    Load cached enhanced descriptions from JSONL files.

    Parameters:
      - directory: the directory containing JSONL files.
      - file_limit: process at most this many files (if specific_file is not provided).
      - line_limit: process at most this many lines per file.
      - specific_file: if provided, process only this file.

    Returns:
      A dictionary mapping term_id to definition.
    """
    enhanced_cache = {}
    if specific_file:
        files = [specific_file]
    else:
        files = sorted(directory.glob("*.jsonl"))
        if file_limit is not None:
            files = files[:file_limit]

    for cache_file in files:
        print(f"Processing file: {cache_file}")
        with open(cache_file, "r") as f:
            for i, line in enumerate(f):
                if line_limit is not None and i >= line_limit:
                    break
                try:
                    # Use model_validate_json (or parse_raw in older versions)
                    item = LLMResponse.model_validate_json(line)
                    if item.definition is not None:
                        enhanced_cache[item.term_id] = item.definition
                except Exception as e:
                    print(f"Skipping line due to error: {e}")
                    continue

    return enhanced_cache
