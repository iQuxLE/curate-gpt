from typing import Dict, Optional

from pydantic import ConfigDict
from venomx.model.venomx import Index,Prefix, Dataset, Embedding, NamedObject, MetadataObject, ModelInputMethod, Model


class CollectionMetadata(Index, Embedding, Dataset, Prefix, NamedObject, MetadataObject, ModelInputMethod, Model):
    """
    Metadata about a collection.
    Inherits from venomx.
    This is an open class, so additional metadata can be added.
    """

    description: Optional[str] = None
    """Description of the collection"""

    object_type: Optional[str] = None
    """Type of object in the collection"""

    object_count: Optional[int] = None
    """Number of objects in the collection"""

    hnsw_space: Optional[str] = None
    """Space used for hnsw index (e.g. 'cosine')"""