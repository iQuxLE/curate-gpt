"""Enhanced ChromaDB adapter with custom description generation for ontology terms."""

import logging
import os
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, Iterable, List, Mapping, Optional, Union

import openai
from chromadb.api import EmbeddingFunction

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

        try:
            response = self.client.chat.completions.create(
                model="o1",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=8000,
                temperature=0.2
            )
            
            enhanced_description = response.choices[0].message.content.strip()
            
            # Cache the result
            self.enhanced_descriptions_cache[cache_key] = enhanced_description
            
            return enhanced_description
        except Exception as e:
            logger.error(f"Error generating enhanced description for {term_id}: {e}")
            # Fallback to original definition if there's an error
            return definition

    def _get_document_for_embedding(self, obj: Dict) -> str:
        """
        Process an object to create text for embedding, enhancing HP term descriptions.
        
        :param obj: The object containing term information
        :return: Text prepared for embedding
        """
        # Check if this is an HP term
        is_hp_term = False
        term_id = obj.get("original_id", "")
        
        if term_id.startswith("HP:"):
            is_hp_term = True
        
        if not is_hp_term:
            # For non-HP terms, just return the regular fields
            parts = []
            for field in ["label", "definition", "relationships"]:
                if field in obj and obj[field]:
                    if field == "relationships" and isinstance(obj[field], list):
                        # Flatten relationships
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
        
        # For HP terms, generate enhanced description
        label = obj.get("label", "")
        definition = obj.get("definition", "")
        relationships = obj.get("relationships", [])
        aliases = obj.get("aliases", [])
        
        enhanced_description = self._generate_enhanced_description(
            term_id, label, definition, relationships, aliases
        )
        
        # Combine the original fields with the enhanced description for embedding
        parts = [
            f"Term: {label}",
            f"ID: {term_id}",
            f"Description: {enhanced_description}"
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
        # Process the texts through the base embedding function
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

        :param model: The embedding model to use
        :return: An embedding function
        """
        # Get the base embedding function from the parent class
        base_ef = super()._embedding_function(model)
        
        # Wrap it with our enhanced embedding function
        return HPTermEnhancedEmbeddingFunction(base_ef)
    
    def _text(self, obj: OBJECT, text_field: Union[str, Callable]) -> str:
        """
        Override the text extraction method to use our enhanced descriptions.
        
        :param obj: The object to extract text from
        :param text_field: The field or function to use for extraction
        :return: Text for embedding
        """
        # For HP terms, use our enhanced embedding function's document processing
        if isinstance(obj, dict) and obj.get("original_id", "").startswith("HP:"):
            ef = self._embedding_function(self.default_model)
            if isinstance(ef, HPTermEnhancedEmbeddingFunction):
                return ef._get_document_for_embedding(obj)
        
        # For all other objects, fall back to the parent implementation
        return super()._text(obj, text_field)