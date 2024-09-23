from dataclasses import Field
from typing import Dict, Optional, List

from pydantic import ConfigDict, BaseModel
from venomx.model.venomx import Index,Prefix, Dataset, Embedding, NamedObject, MetadataObject, ModelInputMethod, Model


class CollectionMetadata(BaseModel):
    """
    Metadata about a collection.
    Useful for all even when no uniqye ID in objects. (e.g. MaxO Ontology)
    This is an open class, so additional metadata can be added.
    """

    model_config = ConfigDict(protected_namespaces=())

    name: Optional[str] = None
    """Name of the collection"""

    description: Optional[str] = None
    """Description of the collection"""

    model: Optional[str] = None
    """Name of any ML model"""

    object_type: Optional[str] = None
    """Type of object in the collection"""

    source: Optional[str] = None
    """Source of the collection"""

    # DEPRECATED
    annotations: Optional[Dict] = None
    """Additional metadata"""

    object_count: Optional[int] = None
    """Number of objects in the collection"""

    hnsw_space: Optional[str] = None
    """Space used for hnsw index (e.g. 'cosine')"""


# use this when Ontology with ID or at least unique identifier
# e.g in hf_adapter return this when OntologyWrapper, which means we need to bea ble to read it from store
# or use the get_raw_objects method to have a understanding and then decide!
class VenomXMetadata(BaseModel):
    """
    Metadata about a collection in VenomX style.
    """

    description: Optional[str]
    """Description of the collection"""
    prefixes: Optional[List[Prefix]]
    """Index prefixes"""
    model: Optional[Model]
    """Model used in the collection"""
    model_input_method: Optional[ModelInputMethod]
    """embedded fields"""
    dataset: Optional[Dataset] = Field(None, description="""Dataset for the collection""")
    """Dataset for the collection"""
    objects: Optional[List[NamedObject]]
    """Objects inside collection"""

    object_type: Optional[str]
    """Type of object in the collection"""
    object_count: Optional[int]
    """Number of objects in the collection"""
    hnsw_space: Optional[str]
    """Space used for hnsw index (e.g., 'cosine')"""
    index: Optional[Index]
    """Index object"""
    embedding: Optional[Embedding]
    """Embedding Object"""
    # Any other VenomX-specific fields can be added here as needed