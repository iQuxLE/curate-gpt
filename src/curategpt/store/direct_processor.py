"""Enhanced ChromaDB adapter with direct API processing for CBORG."""
import json
import logging
import os
import re
import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, Iterable, List, Iterator, Any, Optional

from openai import OpenAI, AsyncOpenAI
from curategpt.utils.response_utils import load_cache

logger = logging.getLogger()

class CborgAsyncEnhancementProcessor:
    """Process ontology terms in batches using direct API calls to CBORG."""

    def __init__(
            self,
            batch_size: int = 5,
            model: str = "openai/gpt-4o",
            cache_dir: Path = Path("batch_output"),
            file_limit: int = None,
            line_limit: int = None,
            specific_file: Path = None,
            max_concurrency: int = 20,
            cborg_api_key=None,
    ):
        """
        Initialize the direct processor.

        Args:
            batch_size: Number of items to process in each output file
            model: OpenAI model to use for enhancement
            cache_dir: Directory to store and load cache from
            file_limit: Limit number of cache files to load
            line_limit: Limit number of cache lines to load per file
            specific_file: Load cache from a specific file only
            max_concurrency: Maximum number of concurrent API requests
        """
        self.batch_size = batch_size
        self.model = model
        self.max_concurrency = max_concurrency
        self.cborg_api_key = cborg_api_key

        self.client = OpenAI(
            api_key=cborg_api_key,
            base_url="https://api.cborg.lbl.gov/v1"
        )

        self.async_client = AsyncOpenAI(
            api_key=cborg_api_key,
            base_url="https://api.cborg.lbl.gov/v1"
        )

        self.enhanced_cache = {}
        self.cache_dir = Path(cache_dir)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.enhanced_cache = load_cache(
                directory=self.cache_dir,
                file_limit=file_limit,
                line_limit=line_limit,
                specific_file=specific_file
            )

        # Validate API key
        if not os.environ.get("CBORG_API_KEY"):
            raise ValueError("Either CBORG_API_KEY or OPENAI_API_KEY environment variable must be set")

    def _create_enhancement_prompt(
            self,
            term_id: str,
            label: str,
            definition: str,
            relationships: List[Dict] = None,
            aliases: List[str] = None
    ) -> str:
        """
        Create a prompt for enhancing an HP term description.

        Args:
            term_id: The ID of the term
            label: The label of the term
            definition: The original definition
            relationships: Optional list of relationships
            aliases: Optional list of aliases

        Returns:
            Prompt for the OpenAI model
        """
        return f"""
You are an expert in human phenotypes and medical terminology. 
Create a comprehensive and detailed description of the phenotypic term: "{label}" (ID: {term_id}).

Your description should:
1. Be more detailed and descriptive than typical ontology definitions
2. Include clinically relevant details about etiology, prevalence, and associated conditions
3. Describe anatomical structures and physiological processes involved
4. Explain how this phenotype may present across different severities and contexts
5. Make the term comparable both to related terms and as a standalone concept
6. Include important distinguishing features that differentiate it from similar phenotypes
7. Be clear to medical professionals while remaining precise
8. Use your general knowledge about human phenotypes to enrich the description

The description should not exceed 8000 tokens.
Provide the enhanced description only, without any additional formatting or meta-information.
"""

    async def _process_term(
            self,
            term_id: str,
            label: str,
            definition: str,
            semaphore: asyncio.Semaphore,
            relationships: List[Dict] = None,
            aliases: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single term using direct API call with CBORG.

        Args:
            term_id: The ID of the term
            label: The label of the term
            definition: The original definition
            semaphore: Semaphore for concurrency control
            relationships: Optional list of relationships
            aliases: Optional list of aliases

        Returns:
            Dict with the result in batch API format
        """
        if term_id in self.enhanced_cache:
            print(f"CACHE: Term {term_id} already in cache, returning cached version")
            return {
                "custom_id": term_id,
                "cached": True,
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": self.enhanced_cache[term_id]
                                }
                            }
                        ]
                    }
                }
            }

        prompt = self._create_enhancement_prompt(term_id, label, definition, relationships, aliases)

        async with semaphore:
            try:
                print(f"CBORG: Processing term {term_id} with direct API call")

                response = await self.async_client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=4096
                )

                content = response.choices[0].text.strip()

                result = {
                    "custom_id": term_id,
                    "cached": False,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "id": response.id,
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": content
                                    }
                                }
                            ]
                        }
                    }
                }

                self.enhanced_cache[term_id] = content
                return result

            except Exception as e:
                logger.error(f"Error processing term {term_id}: {str(e)}")
                return {
                    "custom_id": term_id,
                    "error": str(e)
                }

    async def _process_term_batch(self, term_batch: List[Dict]) -> List[Dict[str, Any]]:
        """
        Process a small batch of terms using the semaphore to limit concurrency.
        Doesn't write to any files - just returns the results.

        Args:
            term_batch: List of terms to process

        Returns:
            List of processed results
        """
        if not term_batch:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrency)

        tasks = []
        for obj in term_batch:
            term_id = obj.get("original_id", "")
            label = obj.get("label", "")
            definition = obj.get("definition", "")
            relationships = obj.get("relationships", [])
            aliases = obj.get("aliases", []) if "aliases" in obj else []

            task = self._process_term(
                term_id,
                label,
                definition,
                semaphore,
                relationships,
                aliases
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        for result in results:
            if "response" in result and "body" in result["response"] and "choices" in result["response"]["body"]:
                choices = result["response"]["body"]["choices"]
                for choice in choices:
                    if "message" in choice and "index" not in choice:
                        choice["index"] = 0

        success_count = sum(1 for r in results if "error" not in r)
        error_count = sum(1 for r in results if "error" in r)
        print(f"Processing batch completed: {success_count} successful, {error_count} failed")

        return results

    def _write_batch_results(self, results: List[Dict], output_dir: Path, batch_num: int) -> None:
        """
        Write collected results to a batch output file and update cache.

        Args:
            results: List of API results
            output_dir: Directory to write to
            batch_num: Batch number for file naming
        """
        new_results = [r for r in results if not r.get("cached", False)]

        if not new_results:
            print(f"No new results to write for batch {batch_num}, skipping file creation")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        batch_output_file = output_dir / f"batch_{batch_num}_output.jsonl"
        with open(batch_output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        result_file = output_dir / "enhanced_descriptions.jsonl"
        with open(result_file, "a") as f:
            for result in results:
                term_id = result.get("custom_id")
                if term_id and "error" not in result and "response" in result:
                    response_body = result["response"].get("body", {})
                    choices = response_body.get("choices", [])
                    if choices and "message" in choices[0]:
                        content = choices[0]["message"].get("content", "")
                        if content:
                            self.enhanced_cache[term_id] = content
                            write_enhanced_descriptions(output_dir, term_id, content)

    def process_ontology_in_batches(
            self,
            ontology_objects: Iterable[Dict],
            output_dir: Path
    ) -> Iterator[Dict]:
        """
        Process ontology objects in batches, enhancing HP terms.
        Uses direct API calls instead of the batch API.

        Args:
            ontology_objects: Iterator of ontology objects
            output_dir: Directory to store batch files and results

        Returns:
            Iterator of enhanced ontology objects
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        # Configure batch sizes
        processing_batch_size = 20  # Number of concurrent API calls
        output_batch_size = 200  # Number of results per output file

        self.enhanced_cache = load_cache_jsonl_batch_files(output_dir)
        print(f"Loaded {len(self.enhanced_cache)} enhanced descriptions from cache")

        highest_existing_batch = -1
        for file_path in output_dir.glob("batch_*_output.jsonl"):
            match = re.search(r"batch_(\d+)_output\.jsonl", str(file_path))
            if match:
                batch_num = int(match.group(1))
                if batch_num > highest_existing_batch:
                    highest_existing_batch = batch_num

        batch_count = highest_existing_batch + 1
        print(f"Starting with batch #{batch_count}")

        hp_term_indices = {}
        all_objects = []

        logger.info("First pass: collecting objects and identifying HP terms")
        for i, obj in enumerate(ontology_objects):
            all_objects.append(obj)

            term_id = obj.get("original_id", "")
            if term_id.startswith("HP:"):
                hp_term_indices[i] = term_id

        logger.info(f"Found {len(all_objects)} total objects with {len(hp_term_indices)} HP terms")

        cached_hp_terms = [term_id for term_id in hp_term_indices.values() if term_id in self.enhanced_cache]
        logger.info(f"{len(cached_hp_terms)} out of {len(hp_term_indices)} HP terms already in cache")

        if hp_term_indices:
            logger.info("Second pass: Processing HP terms in batches")

            terms_to_process = []

            for i, term_id in hp_term_indices.items():
                if term_id not in self.enhanced_cache:
                    terms_to_process.append(all_objects[i])

            logger.info(f"Need to process {len(terms_to_process)} HP terms that aren't in cache")
            collected_results = []
            process_count = 0

            for i in range(0, len(terms_to_process), processing_batch_size):
                process_batch = terms_to_process[i:i + processing_batch_size]
                process_count += len(process_batch)
                process_results = asyncio.run(self._process_term_batch(process_batch))
                collected_results.extend(process_results)
                if len(collected_results) >= output_batch_size or process_count == len(terms_to_process):
                    if collected_results:
                        logger.info(f"Writing batch {batch_count} with {len(collected_results)} results")
                        self._write_batch_results(collected_results, output_dir, batch_count)
                        batch_count += 1
                        collected_results = []

                logger.info(f"Processed {process_count}/{len(terms_to_process)} terms")

        enhanced_count = 0
        print("Third pass: Adding enhanced descriptions to objects")

        for obj in all_objects:
            term_id = obj.get("original_id", "")
            if term_id.startswith("HP:") and term_id in self.enhanced_cache:
                enhanced_description = self.enhanced_cache[term_id]
                obj["enhanced_description"] = enhanced_description
                enhanced_count += 1

            yield obj

        print(f"Added {enhanced_count} enhanced descriptions to {len(hp_term_indices)} HP terms")

        #verification
        if enhanced_count < len(hp_term_indices):
            logger.warning(
                f"WARNING: Not all HP terms were enhanced. {len(hp_term_indices) - enhanced_count} terms missing enhancements.")
            missing_terms = [term_id for term_id in hp_term_indices.values() if term_id not in self.enhanced_cache]
            if missing_terms:
                logger.warning(f"Examples of missing terms: {missing_terms[:5]}")
                with open(os.path.join(output_dir, "terms_missing_response.txt"), "w") as f:
                    f.write("\n".join(missing_terms))

def write_enhanced_descriptions(output_dir: Path, term_id: str, content: str) -> None:
    """
    Write enhanced descriptions to file in a format that's consistent with batch output.

    Args:
        output_dir: Directory to write to
        term_id: Term ID (e.g., HP:0000123)
        content: Enhanced description content
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    enhanced_file = output_dir / "enhanced_descriptions.jsonl"

    response_record = {
        "custom_id": term_id,
        "response": {
            "body": {
                "choices": [
                    {
                        "index": 0,  # Add the required index field
                        "message": {
                            "role": "assistant",
                            "content": content
                        }
                    }
                ]
            }
        }
    }

    with open(enhanced_file, "a") as f:
        f.write(json.dumps(response_record) + "\n")
        logger.info(f"Written enhanced description for {term_id} to JSONL file")


def load_cache_jsonl_batch_files(directory: Path) -> Dict[str, str]:
    """
    Direct parser specifically for your batch format.

    Args:
        directory: Directory containing batch files

    Returns:
        Dictionary mapping term IDs to enhanced descriptions
    """
    cache = {}
    directory = Path(directory)

    if not directory.exists():
        print(f"Cache directory {directory} does not exist")
        return cache

    batch_files = list(directory.glob("batch_*_output.jsonl"))

    for file_path in batch_files:
        print(f"Parsing batch file: {file_path}")
        file_cache_count = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())

                        if "custom_id" not in data:
                            continue

                        term_id = data["custom_id"]
                        if not term_id.startswith("HP:"):
                            continue

                        if "response" not in data or "body" not in data["response"]:
                            continue

                        body = data["response"]["body"]
                        if "choices" not in body or not body["choices"]:
                            continue

                        choices = body["choices"]
                        if not choices or "message" not in choices[0]:
                            continue

                        message = choices[0]["message"]
                        if "content" not in message:
                            continue

                        content = message["content"]
                        if not content:
                            continue

                        print(f"Found term {term_id} at line {line_num}")
                        cache[term_id] = content
                        file_cache_count += 1

                    except json.JSONDecodeError:
                        print(f"Invalid JSON at line {line_num}")
                    except Exception as e:
                        print(f"Error processing line {line_num}: {str(e)}")

            print(f"Loaded {file_cache_count} terms from {file_path}")

        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")

    enhanced_file = directory / "enhanced_descriptions.jsonl"
    if enhanced_file.exists():
        try:
            file_cache_count = 0
            print(f"Parsing enhanced descriptions: {enhanced_file}")

            with open(enhanced_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())

                        if "custom_id" in data and "response" in data and "body" in data["response"]:
                            term_id = data["custom_id"]
                            body = data["response"]["body"]

                            if "choices" in body and body["choices"] and "message" in body["choices"][0]:
                                content = body["choices"][0]["message"].get("content", "")
                                if content:
                                    cache[term_id] = content
                                    file_cache_count += 1
                    except Exception as e:
                        print(f"Error at line {line_num}: {str(e)}")

            print(f"Loaded {file_cache_count} terms from enhanced descriptions")

        except Exception as e:
            print(f"Error reading enhanced descriptions: {str(e)}")

    print(f"Total loaded: {len(cache)} terms")

    sample_keys = list(cache.keys())[:10]
    print(f"Sample terms: {sample_keys}")

    if sample_keys:
        first_term = sample_keys[0]
        content = cache[first_term]
        print(f"Sample content for {first_term}: {content[:100]}...")

    return cache

