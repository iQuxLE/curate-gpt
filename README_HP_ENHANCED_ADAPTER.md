# Enhanced ChromaDB Adapter for HP Terms

This document explains how to use the `EnhancedChromaDBAdapter`, a custom adapter for CurateGPT that generates detailed descriptions for Human Phenotype (HP) terms using OpenAI's o1 model before embedding them.

## Overview

The `EnhancedChromaDBAdapter` extends the standard `ChromaDBAdapter` by:

1. Generating rich, detailed descriptions for HP terms using OpenAI's o1 model
2. Incorporating these enhanced descriptions into the embedding process
3. Caching generated descriptions to avoid redundant API calls
4. Preserving all standard ChromaDB adapter functionality

This results in higher quality embeddings for HP terms that better capture clinical knowledge and context beyond what's provided in the original definitions.

## Prerequisites

- CurateGPT installed
- An OpenAI API key with access to the o1 model
- The `OPENAI_API_KEY` environment variable set

## Usage

### Command Line Usage

To use the adapter from the command line:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Index an HP ontology with enhanced descriptions
curategpt ontology index -p ./my_db_path -c hp_enhanced \
    -D enhanced_chromadb \
    --index-fields label,definition,relationships \
    sqlite:obo:hp
```

### Python Usage

```python
import os
from curategpt.store import EnhancedChromaDBAdapter
from curategpt.wrappers.ontology import OntologyWrapper
from oaklib import get_adapter

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# Initialize the enhanced adapter
db = EnhancedChromaDBAdapter(path="./my_db_path")

# Use with an ontology wrapper
oak_adapter = get_adapter("sqlite:obo:hp")
view = OntologyWrapper(oak_adapter=oak_adapter)

# Index the ontology terms with enhanced descriptions
db.insert(
    view.objects(),
    collection="hp_enhanced",
    model="all-MiniLM-L6-v2",  # The base embedding model to use
    batch_size=100
)
```

### In Other Code Bases

To use the enhanced adapter in another codebase:

1. Install CurateGPT:
```bash
pip install curategpt
```

2. Import and use the adapter:
```python
import os
from curategpt.store import EnhancedChromaDBAdapter

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# Initialize the adapter
db = EnhancedChromaDBAdapter(path="./my_db_path")

# Now use it as you would use a standard ChromaDB adapter
# ...
```

## How it Works

The adapter uses a special embedding function that:

1. Detects HP terms by checking if the original_id starts with "HP:"
2. For HP terms, constructs a detailed prompt for the o1 model
3. Generates an enhanced description that includes:
   - Clinical details about etiology and associated conditions
   - Anatomical structures and physiological processes
   - Presentation across different severities
   - Distinguishing features from similar phenotypes
4. Combines the enhanced description with the original term data
5. Passes this enhanced text to the base embedding model

This process results in embeddings that better capture the semantic meaning and clinical context of HP terms.

## Performance Considerations

- The adapter caches generated descriptions to minimize API calls
- For large ontologies, the first indexing run may be slow due to API calls
- Subsequent runs will be faster if using the same database with cached descriptions

## Example Enhancement

**Original HP Term Definition:**
```
Abnormality of finger morphology: A structural anomaly of the finger.
```

**Enhanced Description (example):**
```
Abnormality of finger morphology refers to any structural deviation from the typical anatomical configuration of the fingers. This broad phenotypic category encompasses a wide spectrum of morphological variations affecting the digits of the hand.

Anatomically, fingers consist of phalanges (proximal, middle, and distal), joints (interphalangeal and metacarpophalangeal), tendons, ligaments, nerves, blood vessels, and skin. Abnormalities can affect any of these structures individually or in combination.

These abnormalities may manifest in various ways, including:
- Polydactyly: The presence of extra fingers
- Syndactyly: Fusion of two or more fingers
- Brachydactyly: Abnormally short fingers
- Arachnodactyly: Abnormally long and slender fingers
- Clinodactyly: Curvature of a finger, typically the fifth digit
- Camptodactyly: Permanent flexion of one or more fingers
- Macrodactyly: Abnormal enlargement of fingers
- Microdactyly: Abnormally small fingers
- Ectrodactyly: Absence of one or more central digits
- Symphalangism: Fusion of finger joints
...
```

## Notes

- The adapter checks for HP terms specifically, but the approach could be extended to other ontologies
- The quality of enhanced descriptions depends on the capabilities of the o1 model
- Descriptions are constrained to 8000 tokens maximum