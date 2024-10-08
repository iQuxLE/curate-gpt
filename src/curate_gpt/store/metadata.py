import json
from dataclasses import Field
from typing import Dict, Optional, List, Any

from pydantic import ConfigDict, BaseModel
from venomx.model.venomx import Index, Model, Dataset, Embedding, ModelInputMethod


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



"""
ChromaDB Constraints:

    Metadata Must Be Scalar: ChromaDB only accepts metadata values that are scalar types (str, int, float, bool).
    No None Values: Metadata fields cannot have None as a value.
    
DuckDB Capabilities:

    Nested Objects Supported: DuckDB can handle nested objects directly within metadata.
"""


class Metadata(Index):
    model_config = ConfigDict(protected_namespaces=())

    # Application-level field for adapters that support nested objects (e.g., DuckDB)
    venomx: Optional[Index] = None
    """
    Retains the complex venomx Index object for internal application use.
    https://github.com/cmungall/venomx
    """

    # Serialized field for adapters that require scalar metadata (e.g., ChromaDB)
    _venomx: Optional[str] = None
    """Stores the serialized JSON string of the venomx object for ChromaDB."""

    hnsw_space: Optional[str] = None
    """Space used for hnsw index (e.g. 'cosine')"""

    object_type: Optional[str] = None
    """Type of object in the collection"""

    @classmethod
    def from_adapter_metadata(cls, metadata_dict: dict, adapter: str):
        """
        Create a Metadata instance from adapter-specific metadata dictionary.

        :param metadata_dict: Metadata dictionary from the adapter.
        :param adapter: Adapter name (e.g., 'chroma', 'duckdb').
        :return: Metadata instance.
        """
        if adapter == 'chroma':
            # Deserialize '_venomx' back into 'venomx'
            venomx_json = metadata_dict.pop("_venomx")
            if venomx_json:
                metadata_dict["venomx"] = Index(**json.loads(venomx_json))
        # For other adapters like 'duckdb', 'venomx' remains as is

        return cls(**metadata_dict)

    def to_adapter_metadata(self, adapter: str) -> dict:
        """
        Convert the Metadata instance to a dictionary suitable for the specified adapter.

        :param adapter: Adapter name (e.g., 'chroma', 'duckdb').
        :return: Metadata dictionary.
        """
        if adapter == 'chromadb':
            # Serialize 'venomx' into '_venomx' and exclude 'venomx'
            metadata_dict = self.model_dump(
                exclude={"venomx"},
                exclude_unset=True,
                exclude_none=True
            )
            if self.venomx:
                metadata_dict["_venomx"] = json.dumps(self.venomx.model_dump())
            return metadata_dict
        elif adapter == 'duckdb':
            # Use 'venomx' directly and exclude '_venomx'
            metadata_dict = self.model_dump(
                exclude={"_venomx"},
                exclude_unset=True,
                exclude_none=True
            )
            return metadata_dict
        else:
            raise ValueError(f"Unsupported adapter: {adapter}")



