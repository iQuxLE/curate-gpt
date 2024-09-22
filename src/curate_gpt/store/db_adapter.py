"""Abstract DB adapter."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    TextIO,
    Union,
)

import pandas as pd
import yaml
from click.utils import LazyFile
from jsonlines import jsonlines
from linkml_runtime.dumpers import json_dumper
from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel

from curate_gpt.store.metadata import CollectionMetadata
from curate_gpt.store.schema_proxy import SchemaProxy
from curate_gpt.store.vocab import (
    DEFAULT_COLLECTION,
    DOCUMENTS,
    EMBEDDINGS,
    FILE_LIKE,
    METADATAS,
    OBJECT,
    PROJECTION,
    QUERY,
    SEARCH_RESULT,
)


logger = logging.getLogger(__name__)


def _get_file(file: Optional[FILE_LIKE] = None, mode="r") -> Optional[TextIO]:
    if file is None:
        return None
    if isinstance(file, Path):
        file = str(file)
    if isinstance(file, str):
        return open(file, mode)
    elif isinstance(file, TextIO):
        return file
    elif isinstance(file, LazyFile):
        return file
    else:
        raise TypeError(f"Unknown file type: {type(file)}")


@dataclass
class DBAdapter(ABC):
    """
    Base class for stores.

    This base class provides a common interface for a wide variety of document or object stores.
    The interface is intended to closely mimic the kind of interface found for document stores
    such as mongoDB or vector databases such as ChromaDB, but the intention is that can be
    used for SQL databases, SPARQL endpoints, or even file systems.

    The store allows for storage and retrieval of *objects* which are arbitrary dictionary
    objects, equivalient to a JSON object.

    Objects are partitioned into *collections*, which maps to the equivalent concept in
    MongoDB and ChromaDB.

    >>> from curate_gpt.store import get_store
    >>> store = get_store("in_memory")
    >>> store.insert({"name": "John", "age": 42}, collection="people")

    If you are used to working with MongoDB and ChromaDB APIs directly, one difference is that
    here we do not provide a separate Collection object, everything is handled through the
    store object. You can optionally bind a store object to a collection, which effectively
    gives you a collection object:

    >>> from curate_gpt.store import get_store
    >>> store = get_store("in_memory")
    >>> store.set_collection("people")
    >>> store.insert({"name": "John", "age": 42})

    TODO: decide if this is the final interface
    """

    path: str = None
    """Path to a location where the database is stored or disk or the network."""

    name: ClassVar[str] = "base"

    schema_proxy: Optional[SchemaProxy] = None
    """Schema manager"""

    collection: Optional[str] = None
    """Default collection"""

    # _field_names_by_collection: Dict[str, Optional[List[str]]] = field(default_factory=dict)
    _field_names_by_collection: Dict[str, Optional[List[str]]] = None

    distance_metric: str = "cosine"
    """Default distance metric"""

    default_model = "all-MiniLM-L6-v2"
    """Default model"""

    id_field: str = field(default="id")
    """ID to search for in object to insert"""

    text_lookup: Optional[Union[str, Callable]] = field(default="text")
    """Text to embed"""

    id_to_object: Mapping[str, OBJECT] = field(default_factory=dict)
    """if no id full obj is id"""

    default_max_document_length: ClassVar[int] = 6000  # TODO: use tiktoken
    """max document length for text to embed (differently handled in duckdb - optimised with tokenizer)"""

    # CUD operations

    @abstractmethod
    def insert(self, objs: Union[OBJECT, Iterable[OBJECT]], collection: str = None, **kwargs):
        """
        Insert an object or list of objects into the store.

        >>> from curate_gpt.store import get_store
        >>> store = get_store("in_memory")
        >>> store.insert([{"name": "John", "age": 42}], collection="people")

        :param objs:
        :param collection:
        :return:
        """

    def update(self, objs: Union[OBJECT, List[OBJECT]], collection: str = None, **kwargs):
        """
        Update an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        raise NotImplementedError

    def upsert(self, objs: Union[OBJECT, List[OBJECT]], collection: str = None, **kwargs):
        """
        Upsert an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        raise NotImplementedError

    def delete(self, id: str, collection: str = None, **kwargs):
        """
        Delete an object by its ID.

        :param id:
        :param collection:
        :return:
        """
        raise NotImplementedError

    # View operations

    def create_view(self, view_name: str, collection: str, expression: QUERY, **kwargs):
        """
        Create a view in the database.

        Todo:
        ----
        :param view:
        :return:
        """
        raise NotImplementedError

    # Collection operations

    def set_collection(self, collection: str):
        """
        Set the current collection.

        If this is set, then all subsequent operations will be performed on this collection, unless
        overridden.

        This allows the following

        >>> from curate_gpt.store import get_store
        >>> store = get_store("in_memory")
        >>> store.set_collection("people")
        >>> store.insert([{"name": "John", "age": 42}])

        to be written in place of

        >>> from curate_gpt.store import get_store
        >>> store = get_store("in_memory")
        >>> store.insert([{"name": "John", "age": 42}], collection="people")

        :param collection:
        :return:
        """
        self.collection = collection

    def _get_collection(self, collection: Optional[str] = None):
        if collection is not None:
            return collection
        elif self.collection is not None:
            return self.collection
        else:
            return DEFAULT_COLLECTION

    @abstractmethod
    def create_collection(self, collection: str = None, **kwargs):
        """
        Create a collection on the current DB instance
        :param collection:
        :return:
        """
        raise NotImplementedError

    def remove_collection(self, collection: str = None, exists_ok=False, **kwargs):
        """
        Remove a collection from the database.

        :param collection:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def list_collection_names(self) -> List[str]:
        """
        List all collections in the database.

        :return: names of collections
        """

    @abstractmethod
    def collection_metadata(
        self, collection_name: Optional[str] = None, include_derived=False, **kwargs
    ) -> Optional[CollectionMetadata]:
        """
        Get the metadata for a collection.

        :param collection_name:
        :param include_derived: Include derived metadata, e.g. counts
        :return:
        """

    def set_collection_metadata(
        self, collection_name: Optional[str], metadata: CollectionMetadata, **kwargs
    ):
        """
        Set the metadata for a collection.

        >>> from curate_gpt.store import get_store
        >>> from curate_gpt.store import CollectionMetadata
        >>> store = get_store("in_memory")
        >>> cm = CollectionMetadata(name="People", description="People in the database")
        >>> store.set_collection_metadata("people", cm)

        :param collection_name:
        :return:
        """
        raise NotImplementedError

    def update_collection_metadata(self, collection_name: str, **kwargs) -> CollectionMetadata:
        """
        Update the metadata for a collection.

        :param collection_name:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self, text: str, where: QUERY = None, collection: str = None, **kwargs
    ) -> Iterator[SEARCH_RESULT]:
        """
        Query the database for a text string.

        >>> from curate_gpt.store import get_store
        >>> store = get_store("chromadb", "db")
        >>> for obj, distance, info in store.search("forebrain neurons", collection="ont_cl"):
        ...     obj_id = obj["id"]
        ...     # print at precision of 2 decimal places
        ...     print(f"{obj_id} {distance:.2f}")
        <BLANKLINE>
        ...
        NeuronOfTheForebrain 0.28
        ...

        :param text:
        :param collection:
        :param where:
        :param kwargs:
        :return: tuple of object, distance, metadata
        """

    def find(
        self,
        where: QUERY = None,
        projection: PROJECTION = None,
        collection: str = None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        """
        Query the database.

        >>> from curate_gpt.store import get_store
        >>> store = get_store("chromadb", "db")
        >>> objs = list(store.find({"name": "NeuronOfTheForebrain"}, collection="ont_cl"))

        :param collection:
        :param where:
        :param projection:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def matches(self, obj: OBJECT, **kwargs) -> Iterator[SEARCH_RESULT]:
        """
        Query the database for matches to an object.

        :param obj:
        :param kwargs:
        :return:
        """

    @abstractmethod
    def lookup(self, id: str, collection: str = None, **kwargs) -> OBJECT:
        """
        Lookup an object by its ID.

        :param id:
        :param collection:
        :return:
        """

    def lookup_multiple(self, ids: List[str], **kwargs) -> Iterator[OBJECT]:
        """
        Lookup an object by its ID.

        :param id:
        :param collection:
        :return:
        """
        yield from [self.lookup(id, **kwargs) for id in ids]

    @abstractmethod
    def peek(self, collection: str = None, limit=5, **kwargs) -> Iterator[OBJECT]:
        """
        Peek at first N objects in a collection.

        :param collection:
        :param limit:
        :return:
        """
        raise NotImplementedError

    # Schema operations
    # thats basically get() from chroma
    @abstractmethod
    def fetch_all_objects_memory_safe(
        self, collection: str = None, batch_size: int = 100, **kwargs
    ) -> Iterator[OBJECT]:
        """
        Fetch all objects from a collection, in batches to avoid memory overload.
        """
        raise NotImplementedError

    def identifier_field(self, collection: str = None) -> str:
        # TODO: use collection
        if self.schema_proxy and self.schema_proxy.schemaview:
            fields = []
            for s in self.schema_proxy.schemaview.all_slots(attributes=True).values():
                if s.identifier:
                    fields.append(s.name)
            if fields:
                if len(fields) > 1:
                    raise ValueError(f"Multiple identifier fields: {fields}")
                return fields[0]
        return "id"

    def label_field(self, collection: str = None) -> str:
        field_names = self.field_names(collection)
        preferred = ["label", "name", "title"]
        for candidate in preferred:
            if candidate in field_names:
                return candidate
        return field_names[0] if field_names else "label"

    # TODO: Might be abstract as currently gives not same values for DuckDBVSSAdapter and ChromaDBAdapter
    def field_names(self, collection: str = None) -> List[str]:
        """
        Return the names of all top level fields in the database for a collection.

        :param collection:
        :return:
        """
        # TODO: use schema proxy if set
        if not self._field_names_by_collection:
            self._field_names_by_collection = {}
        collection = self._get_collection(collection)
        if collection not in self._field_names_by_collection:
            logger.debug(f"Getting field names for {collection}")
            objs = self.peek(collection=collection, limit=1)
            if not objs:
                raise ValueError(f"Collection {collection} is empty")
            fields = []
            for obj in objs:
                for f in obj:
                    if f not in fields:
                        fields.append(f)
            self._field_names_by_collection[collection] = fields
        else:
            logger.debug(f"Using cached field names for {collection}")
        return self._field_names_by_collection[collection]

    def _text(self, obj: OBJECT, text_field: Union[str, Callable]):
        if isinstance(obj, list):
            raise ValueError(f"Cannot handle list of text fields: {obj}")
        if text_field is None or (isinstance(text_field, str) and text_field not in obj):
            obj = {k: v for k, v in obj.items() if v}
            t = yaml.safe_dump(obj, sort_keys=False)
        elif isinstance(text_field, Callable):
            t = text_field(obj)
        elif isinstance(obj, dict):
            t = obj[text_field]
        else:
            t = getattr(obj, text_field)
        t = t.strip()
        if not t:
            raise ValueError(f"Text field {text_field} is empty for {type(obj)} : {obj}")
        if len(t) > self.default_max_document_length:
            logger.warning(f"Truncating text field {text_field} for {str(obj)[0:100]}...")
            t = t[: self.default_max_document_length]
        return t

    def _id(self, obj: OBJECT, id_field: str):
        if isinstance(obj, dict):
            id = obj.get(id_field, None)
        else:
            id = getattr(obj, id_field, None)
        if not id:
            id = str(obj)
        self.id_to_object[id] = obj
        return id

    def _dict(self, obj: OBJECT):
        if isinstance(obj, dict):
            return obj
        elif isinstance(obj, BaseModel):
            return obj.dict(exclude_unset=True)
        elif isinstance(obj, YAMLRoot):
            return json_dumper.to_dict(obj)
        else:
            raise ValueError(f"Cannot convert {obj} to dict")

    # Loading and dumping
    def dump(
        self,
        collection: str = None,
        to_file: FILE_LIKE = None,
        metadata_to_file: FILE_LIKE = None,
        format=None,
        include=None,
        **kwargs,
    ):
        """
        Dump the database to a file.

        :param collection:
        :param kwargs:
        :return:
        """
        collection_name = collection
        collection = self._get_collection(collection)
        logger.debug(f"Dumping collection {self.collection_metadata(collection) }")
        metadata = self.collection_metadata(collection).model_dump(exclude_none=True)
        if format is None:
            format = "json"
        if not include:
            include = [EMBEDDINGS, DOCUMENTS, METADATAS]
            # include = ["embeddings", "documents", "metadatas"]
        if not isinstance(include, list):
            include = list(include)
        objects = list(self.find(collection=collection, include=include, **kwargs))
        if format.startswith("venomx"):
            import venomx as vx
            from venomx.tools.file_io import save_index

            cm = self.collection_metadata(collection_name, include_derived=False)
            label_field = self.label_field(collection_name)
            id_field = self.identifier_field(collection_name)

            def _label(obj):
                return obj.get(label_field, None)

            def _id(obj):
                original_id = obj.get("original_id", None)
                if original_id:
                    return original_id
                return obj.get(id_field, None)

            rows = [
                {"id": _id(obj), "text": meta["document"], "values": meta["_embeddings"]}
                for obj, _, meta in objects
            ]
            vx_objects = [{"id": _id(obj), "label": _label(obj)} for obj, _, _ in objects]
            logger.info(f"Num vx objects: {len(vx_objects)}")
            vx_index = vx.Index(
                embedding_model=vx.model.venomx.Model(name=cm.model),
                dataset=vx.model.venomx.Dataset(name=collection_name),
                embeddings_frame=pd.DataFrame(rows),
                objects=vx_objects,
            )
            save_index(vx_index, to_file)
            return
        to_file = _get_file(to_file, "w")
        metadata_to_file = _get_file(metadata_to_file, "w")
        streaming = True
        if format == "jsonl":
            writer = jsonlines.Writer(to_file)
            writer.write_all(objects)
            writer.close()
        elif format == "yamlblock":
            for obj in objects:
                to_file.write("---\n")
                yaml.dump(obj, to_file)
        else:
            streaming = False
            database = {"metadata": metadata, "objects": list(objects)}
            if format == "json":
                json.dump(database, to_file)
            elif format == "yaml":
                yaml.dump(database, to_file)
            else:
                raise ValueError(f"Unknown format {format}")
        if streaming:
            if not metadata_to_file:
                raise ValueError("Streaming dump requires metadata_to_file")
            if format == "jsonl":
                metadata_to_file.write(json.dumps(metadata))
            elif format == "yamlblock":
                metadata_to_file.write(yaml.dump(metadata))

    def dump_then_load(self, collection: str = None, target: "DBAdapter" = None):
        """
        Dump a collection to a file, then load it into another database.

        :param collection:
        :param target:
        :return:
        """
        raise NotImplementedError

    # def dump_file_to_db(self, collection, file, **kwargs):
    #     with open(file, "r") as f:
    #         yaml.dump(f)
    #     # the way it works now: cause it uses Metadata object from store wich is i think not correct
    #     db = get_store()
    #     # need FileWrapper or needs to understand from structure what it is, maybe JsonWrapper ? maybe YamlWrapper
    #     # or OntologyWrapper, so maybe just BaseWrapper
    #     filewrapper = FilesystemWrapper(local_store=db, extractor=)



