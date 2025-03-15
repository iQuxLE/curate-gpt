"""Enhanced ChromaDB adapter with custom description generation for ontology terms."""
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Dict, Iterable, List, Mapping, Optional, Union, Iterator

import openai
from chromadb.api import EmbeddingFunction
from openai import OpenAI

from curategpt.store.chromadb_adapter import ChromaDBAdapter
from curategpt.store.vocab import OBJECT

logger = logging.getLogger(__name__)


class HPTermEnhancedEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for HP (Human Phenotype) terms that generates enhanced 
    descriptions using OpenAI's o1 model before embedding.
    """

    def __init__(self, base_embedding_function: EmbeddingFunction):
        """
        Initialize with a base embedding function that will perform the actual embedding.
        
        :param base_embedding_function: The underlying embedding function to use
        """
        self.base_embedding_function = base_embedding_function
        
        # Ensure OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        self.client = openai.OpenAI()
        self.enhanced_descriptions_cache = {}

    def _generate_enhanced_description(self, term_id: str, label: str, definition: str,
                                      relationships: List[Dict] = None, aliases: List[str] = None) -> str:
        """
        Generate an enhanced description for an HP term using the OpenAI o1 model.
        
        :param term_id: The ID of the term
        :param label: The label of the term
        :param definition: The original definition
        :param relationships: Optional list of relationships
        :param aliases: Optional list of aliases
        :return: Enhanced description
        """
        # Check if we already have a cached description
        cache_key = term_id
        if cache_key in self.enhanced_descriptions_cache:
            return self.enhanced_descriptions_cache[cache_key]
        
        # Create a detailed prompt
        prompt = f"""
You are an expert in human phenotypes and medical terminology. 
Create a comprehensive and detailed description of the phenotypic term: "{label}" (ID: {term_id}).

The existing definition is: "{definition}"

{f"The term has these aliases: {', '.join(aliases)}" if aliases else ""}

{f"The term has these relationships: {relationships}" if relationships else ""}

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

        response = self.client.chat.completions.create(
            model="o1",
            messages=[{"role": "system", "content": prompt}],
            max_completion_tokens=8000,
        )

        enhanced_description = response.choices[0].message.content.strip()
        print("LOGGING API")
        print(enhanced_description)

        self.enhanced_descriptions_cache[cache_key] = enhanced_description

        return enhanced_description


    def _get_document_for_embedding(self, obj: Dict) -> str:
        """
        Process an object to create text for embedding, enhancing HP term descriptions.
        
        :param obj: The object containing term information
        :return: Text prepared for embedding
        """
        is_hp_term = False
        term_id = obj.get("original_id", "")
        
        if term_id.startswith("HP:"):
            is_hp_term = True
        
        if not is_hp_term:
            parts = []
            for field in ["label", "definition", "relationships"]:
                if field in obj and obj[field]:
                    if field == "relationships" and isinstance(obj[field], list):
                        rel_texts = []
                        for rel in obj[field]:
                            if isinstance(rel, dict):
                                rel_texts.append(f"{rel.get('predicate', '')}: {rel.get('target', '')}")
                        parts.append(" ".join(rel_texts))
                    else:
                        parts.append(str(obj[field]))
            
            if "aliases" in obj and obj["aliases"]:
                parts.append("Aliases: " + ", ".join(obj["aliases"]))
                
            return " ".join(parts)
        
        label = obj.get("label", "")
        definition = obj.get("definition", "")
        relationships = obj.get("relationships", [])
        aliases = obj.get("aliases", [])
        
        enhanced_description = self._generate_enhanced_description(
            term_id, label, definition, relationships, aliases
        )
        if enhanced_description is None:
            enhanced_description = ""
        
        parts = [
            label,
            term_id,
            enhanced_description
        ]
        
        if aliases:
            parts.append("Aliases: " + ", ".join(aliases))

        return " ".join(parts)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Process and embed a list of texts.
        
        This function will be called by ChromaDB when embedding documents.
        
        :param texts: List of texts or JSON strings
        :return: List of embedding vectors
        """
        return self.base_embedding_function(texts)


@dataclass
class EnhancedChromaDBAdapter(ChromaDBAdapter):
    """
    Enhanced ChromaDB adapter that uses custom description generation for HP ontology terms.
    """
    name: ClassVar[str] = "enhanced_chromadb"
    
    def _embedding_function(self, model: str = None) -> EmbeddingFunction:
        """
        Get the embedding function for a given model, enhancing it with HP term descriptions.

        :param model: The embedding model to use. Supports multiple formats:
                      - OpenAI models: "openai:model-name" or shorthands like "ada", "small3", "large3"
                      - Ollama models: "ollama:model-name"
                      - Hugging Face models: Shorthands like "bge-m3", "nomic", "mxbai-l"
                      - SentenceTransformer models: Direct model names like "all-MiniLM-L6-v2"
        :return: An enhanced embedding function for HP terms
        """
        base_ef = super()._embedding_function(model)
        
        return HPTermEnhancedEmbeddingFunction(base_ef)
    
    def _text(self, obj: OBJECT, text_field: Union[str, Callable]) -> str:
        """
        Override the text extraction method to use our enhanced descriptions.
        
        :param obj: The object to extract text from
        :param text_field: The field or function to use for extraction
        :return: Text for embedding
        """

        print("Inside _text method")
        print("Object original_id:", obj.get("original_id", ""))
        if isinstance(obj, dict) and obj.get("original_id", "").startswith("HP:"):
            print("HP term detected, using enhanced description")
            ef = self._embedding_function("large3")
            if isinstance(ef, HPTermEnhancedEmbeddingFunction):
                return ef._get_document_for_embedding(obj)
        print("Falling back to super() _text")
        return super()._text(obj, text_field)


class BatchEnhancementProcessor:
    """Process ontology terms in batches using OpenAI's Batch API."""

    def __init__(
            self,
            batch_size: int = 100,
            model: str = "o1",
            completion_window: str = "24h"
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
                        "max_tokens": 8000
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

The existing definition is: "{definition}"

{f"The term has these aliases: {', '.join(aliases)}" if aliases else ""}

{f"The term has these relationships: {relationships}" if relationships else ""}

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

    def process_ontology_in_batches(
            self,
            ontology_objects: Iterable[Dict],
            output_dir: Path
    ) -> Iterator[Dict]:
        """
        Process ontology objects in batches, enhancing HP terms.

        Args:
            ontology_objects: Iterator of ontology objects from filtered_o
            output_dir: Directory to store batch files and results

        Returns:
            Iterator of enhanced ontology objects
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_count = 0
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

            for i, term_id in hp_term_indices.items():
                if term_id in self.enhanced_cache:
                    logger.info(f"Using cached enhanced description for {term_id}")
                    continue

                hp_terms_batch.append(all_objects[i])
                batch_hp_indices[len(hp_terms_batch) - 1] = i

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
        Process a batch of HP terms.

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
