"""Enhanced ChromaDB adapter with custom description generation for ontology terms."""
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Iterator

from openai import OpenAI
from curategpt.utils.custom_definition_utils import load_cache, load_existing_batch_outputs

logger = logging.getLogger(__name__)

class BatchEnhancementProcessor:
    """Process ontology terms in batches using OpenAI's Batch API."""

    def __init__(
            self,
            batch_size: int = 1000,
            model: str = "gpt-4o",
            completion_window: str = "24h",
            cache_dir: Path = "batch_output",
            file_limit: int = None,
            line_limit: int = None,
            specific_file: Path = None

    ):
        """
        Initialize the batch processor.

        Args:
            batch_size: Number of items to process in each batch
            model: OpenAI model to use for enhancement
            completion_window: Completion window for batch API
        """
        self.batch_size = batch_size
        self.model = model
        self.completion_window = completion_window
        self.client = OpenAI()
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

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable must be set")


    def prepare_batch_file(self, objects: List[Dict], output_file: str) -> str:
        """
        Prepare a JSONL batch file for OpenAI Batch API.

        Args:
            objects: List of ontology objects to enhance
            output_file: Path to write the batch file

        Returns:
            Path to the created batch file
        """
        logger.info(f"Preparing batch file with {len(objects)} objects")

        with open(output_file, 'w') as f:
            for i, obj in enumerate(objects):
                term_id = obj.get("original_id", "")

                if not term_id.startswith("HP:"):
                    continue

                label = obj.get("label", "")
                definition = obj.get("definition", "")
                relationships = obj.get("relationships", [])
                aliases = obj.get("aliases", []) if "aliases" in obj else []

                prompt = self._create_enhancement_prompt(term_id, label, definition, relationships, aliases)

                request = {
                    "custom_id": term_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": prompt}
                        ],
                        "max_completion_tokens": 8000
                    }
                }

                # Write the request to the batch file
                f.write(json.dumps(request) + '\n')

        logger.info(f"Batch file created at {output_file}")
        return output_file

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

    # The existing definition is: "{definition}"
    #
    # {f"The term has these aliases: {', '.join(aliases)}" if aliases else ""}
    #
    # {f"The term has these relationships: {relationships}" if relationships else ""}

    def process_ontology_in_batches(
            self,
            ontology_objects: Iterable[Dict],
            output_dir: Path
    ) -> Iterator[Dict]:
        """
        Process ontology objects in batches, enhancing HP terms.
        Sores these objects in a list (all_objects) and maps their original indices in a dictionary (hp_term_indices).

        Args:
            ontology_objects: Iterator of ontology objects from filtered_o
            output_dir: Directory to store batch files and results

        Returns:
            Iterator of enhanced ontology objects
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        highest_existing_batch = load_existing_batch_outputs(output_dir)
        batch_count = highest_existing_batch + 1
        batch = []

        hp_term_indices = {}
        all_objects = []

        logger.info("First pass: collecting objects and identifying HP terms")
        for i, obj in enumerate(ontology_objects):
            all_objects.append(obj)

            term_id = obj.get("original_id", "")
            if term_id.startswith("HP:"):
                hp_term_indices[i] = term_id

        logger.info(f"Found {len(all_objects)} total objects with {len(hp_term_indices)} HP terms")

        if hp_term_indices:
            logger.info("Second pass: Processing HP terms in batches")
            batch_hp_indices = {}
            hp_terms_batch = []

            #take from cache if existing
            for i, term_id in hp_term_indices.items():
                if term_id in self.enhanced_cache:
                    logger.info(f"Using cached response for {term_id}")
                    continue
                # add to temp hp_terms_batch and record position in batch_hp_indices
                hp_terms_batch.append(all_objects[i])
                batch_hp_indices[len(hp_terms_batch) - 1] = i
                # when nr of items in hp_terms_batch reaches batch_size -> process_batch
                if len(hp_terms_batch) >= self.batch_size:
                    self._process_batch(hp_terms_batch, batch_hp_indices, all_objects, output_dir, batch_count)
                    batch_count += 1
                    hp_terms_batch = []
                    batch_hp_indices = {}

            if hp_terms_batch:
                self._process_batch(hp_terms_batch, batch_hp_indices, all_objects, output_dir, batch_count)

        for obj in all_objects:
            term_id = obj.get("original_id", "")
            if term_id.startswith("HP:") and term_id in self.enhanced_cache:
                enhanced_description = self.enhanced_cache[term_id]
                obj["enhanced_description"] = enhanced_description
            yield obj

    def _process_batch(
            self,
            hp_terms_batch: List[Dict],
            batch_indices: Dict[int, int],
            all_objects: List[Dict],
            output_dir: Path,
            batch_num: int
    ):
        """
        Process a batch of HP terms. Creates Jsonl as request file. Uploads and creates job to OpenAI API.

        Args:
            hp_terms_batch: Batch of HP terms to process
            batch_indices: Mapping from batch index to original index
            all_objects: List of all ontology objects
            output_dir: Directory to store batch files and results
            batch_num: Batch number for file naming
        """
        if not hp_terms_batch:
            return

        batch_file = output_dir / f"batch_{batch_num}.jsonl"
        logger.info(f"Processing batch {batch_num} with {len(hp_terms_batch)} HP terms")

        self.prepare_batch_file(hp_terms_batch, str(batch_file))

        batch_file_upload = self.client.files.create(
            file=open(batch_file, "rb"),
            purpose="batch"
        )

        print(f"Uploaded batch file with ID: {batch_file_upload.id}")

        batch = self.client.batches.create(
            input_file_id=batch_file_upload.id,
            endpoint="/v1/chat/completions",
            completion_window=self.completion_window
        )

        print(f"Created batch with ID: {batch.id}")

        completed = False
        while not completed:
            batch_status = self.client.batches.retrieve(batch.id)
            status = batch_status.status

            logger.info(f"Batch {batch.id} status: {status}")

            if status == "completed":
                completed = True
            elif status in ["failed", "expired", "cancelled"]:
                logger.error(f"Batch {batch.id} {status}")
                return
            else:
                logger.info(f"Waiting for batch to complete. Current status: {status}")
                time.sleep(30)

        output_file_id = batch_status.output_file_id
        if output_file_id:
            results_response = self.client.files.content(output_file_id)
            results_content = results_response.text

            # Write the complete batch response to a file
            batch_output_file = output_dir / f"batch_{batch_num}_output.jsonl"
            with open(batch_output_file, "w") as f:
                f.write(results_content)

            results = [json.loads(line) for line in results_content.strip().split('\n')]

            for result in results:
                term_id = result.get("custom_id")
                if term_id and "error" not in result:
                    response_body = result.get("response", {}).get("body", {})
                    choices = response_body.get("choices", [])

                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        if content:
                            self.enhanced_cache[term_id] = content
                            logger.info(f"Cached enhanced description for {term_id}")

                            # Write to output /enhanced_descriptions.jsonl
                            result_file = output_dir / f"enhanced_descriptions.jsonl"
                            with open(result_file, "a") as f:
                                json_record = json.dumps({"term_id": term_id, "enhanced_description": content})
                                f.write(json_record + "\n")
                                logger.info(f"Written enhanced description for {term_id} to JSONL file")
