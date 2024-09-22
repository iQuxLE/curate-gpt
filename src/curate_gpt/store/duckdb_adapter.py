"""
This is a DuckDB adapter for the Vector Similarity Search (VSS) extension
using the experimental persistence feature
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterable, Iterator, List, Optional, Union

import duckdb
import llm
import numpy as np
import openai
import psutil
from oaklib.utilities.iterator_utils import chunk
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from curate_gpt.store.db_adapter import DBAdapter
from curate_gpt.store.duckdb_result import DuckDBSearchResult
from curate_gpt.store.metadata import CollectionMetadata
from curate_gpt.store.vocab import (
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    DISTANCES,
    DOCUMENTS,
    EMBEDDINGS,
    IDS,
    METADATAS,
    MODEL_MAP,
    OBJECT,
    PROJECTION,
    QUERY,
    SEARCH_RESULT,
)
from curate_gpt.utils.vector_algorithms import mmr_diversified_search

logger = logging.getLogger(__name__)


@dataclass
class DuckDBAdapter(DBAdapter):
    name: ClassVar[str] = "duckdb"
    conn: duckdb.DuckDBPyConnection = field(init=False)
    vec_dimension: int = field(init=False)
    ef_construction: int = 128
    ef_search: int = 64
    M: int = 16
    openai_client: OpenAI = field(default=None)

    def __post_init__(self):
        if not self.path:
            self.path = "./db/db_file.duckdb"
        if os.path.isdir(self.path):
            self.path = os.path.join("./db", self.path, "db_file.duckdb")
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            logger.info(
                f"Path {self.path} is a directory. Using {self.path} as the database path\n\
            as duckdb needs a file path"
            )
        logger.info(f"Using DuckDB at {self.path}")
        # handling concurrency
        try:
            self.conn = duckdb.connect(self.path, read_only=False)
        except duckdb.IOException as e:
            match = re.search(r"PID (\d+)", str(e))
            if match:
                pid = int(match.group(1))
                logger.info(f"Got {e}.Attempting to kill process with PID: {pid}")
                self.kill_process(pid)
                self.conn = duckdb.connect(self.path, read_only=False)
            else:
                logger.error(f"{e} without PID information.")
                raise
        self.conn.execute("INSTALL vss;")
        self.conn.execute("LOAD vss;")
        self.conn.execute("SET hnsw_enable_experimental_persistence=true;")
        if self.default_model is None:
            self.model = self.default_model
        self.vec_dimension = self._get_embedding_dimension(self.default_model)

    def _initialize_openai_client(self):
        if self.openai_client is None:
            from dotenv import load_dotenv

            load_dotenv()
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
            else:
                raise openai.OpenAIError(
                    "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
                )

    def _get_collection_name(self, collection: Optional[str] = None) -> str:
        """
        Get the collection name or the default collection name
        :param collection:
        :return:
        """
        return self._get_collection(collection)

    def _create_table_if_not_exists(
        self, collection: str, vec_dimension: int, distance: str, model: str = None
    ):
        """
        Create a table for the given collection if it does not exist
        :param collection:
        :return:
        """
        logger.info(
            f"Table {collection} does not exist, and is created with the following table metadata: model: {model}, distance: {distance},\
        vec_dimension: {vec_dimension}"
        )
        if model is None:
            model = self.default_model
            logger.debug(f"Model in create_table_if_not_exists: {model}")
        if distance is None:
            distance = self.distance_metric

        self.create_collection(
            collection=collection,
            model=model,
            vec_dimension=vec_dimension,
            distance=distance,
        )

    def create_index(self, collection: str):
        """
        Create an index for the given collection
        Parameters
        ----------
        collection

        Returns
        -------

        """
        cm = self.collection_metadata(collection)
        sql_safe_collection_name = f'"{collection}"'
        index_name = f"{collection}_index"
        create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS "{index_name}" ON {sql_safe_collection_name}
            USING HNSW (embeddings) WITH (
                metric='{cm.hnsw_space}',
                ef_construction={self.ef_construction},
                ef_search={self.ef_search},
                M={self.M}
            )
        """
        self.conn.execute(create_index_sql)

    def _embedding_function(
        self, texts: Union[str, List[str], List[List[str]]], model: str = None
    ) -> list:
        """
        Get the embeddings for the given texts using the specified model
        :param texts: A single text or a list of texts to embed
        :param model: Model to use for embedding
        :return: A single embedding or a list of embeddings
        """
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True

        if model is None:
            model = self.model

        if model.startswith("openai:"):
            self._initialize_openai_client()
            openai_model = model.split(":", 1)[1]
            if openai_model == "" or openai_model not in MODEL_MAP.keys():
                logger.info(
                    f"The model {openai_model} is not "
                    f"one of {[MODEL_MAP.keys()]}. Defaulting to {DEFAULT_OPENAI_MODEL}"
                )
                openai_model = DEFAULT_OPENAI_MODEL

            responses = [
                self.openai_client.embeddings.create(input=text, model=openai_model)
                .data[0]
                .embedding
                for text in texts
            ]
            return responses[0] if single_text else responses

        model = SentenceTransformer(model)
        embeddings = model.encode(texts, convert_to_tensor=False).tolist()
        return embeddings[0] if single_text else embeddings

    def insert(self, objs: Union[OBJECT, Iterable[OBJECT]], **kwargs):
        """
        Insert objects into the collection
        :param objs:
        :param kwargs:
        :return:
        """
        logger.info(f"\n\nIn insert duckdb, {kwargs.get('model')}\n\n")
        self._process_objects(objs, method="insert", **kwargs)

    # DELETE first to ensure primary key  constraint https://duckdb.org/docs/sql/indexes
    def update(self, objs: Union[OBJECT, Iterable[OBJECT]], **kwargs):
        """
        Update objects in the collection.
        :param objs:
        :param kwargs:
        :return:
        """
        collection = kwargs.get("collection")
        ids = [self._id(o, self.id_field) for o in objs]
        sql_safe_collection_name = f'"{collection}"'
        delete_sql = f"DELETE FROM {sql_safe_collection_name} WHERE id = ?"
        logger.info("DELETED collection: {collection}")
        self.conn.executemany(delete_sql, [(id_,) for id_ in ids])
        logger.info(f"INSERTING collection: {collection}")
        self.insert(objs, **kwargs)

    def upsert(self, objs: Union[OBJECT, Iterable[OBJECT]], **kwargs):
        """
        Upsert objects into the collection
        :param objs:
        :param kwargs:
        :return:
        """
        collection = kwargs.get("collection")
        logger.info(f"\n\nUpserting objects into collection {collection}\n\n")
        logger.info(f"model in upsert: {kwargs.get('model')}, distance: {self.distance_metric}")
        if collection not in self.list_collection_names():
            vec_dimension = self._get_embedding_dimension(kwargs.get("model"))
            self._create_table_if_not_exists(
                collection, vec_dimension, model=kwargs.get("model"), distance=self.distance_metric
            )
        ids = [self._id(o, self.id_field) for o in objs]
        existing_ids = set()
        for id_ in ids:
            sql_safe_collection_name = f'"{collection}"'
            result = self.conn.execute(
                f"SELECT id FROM {sql_safe_collection_name} WHERE id = ?", [id_]
            ).fetchall()
            if result:
                existing_ids.add(id_)
        objs_to_update = [o for o in objs if self._id(o, self.id_field) in existing_ids]
        objs_to_insert = [o for o in objs if self._id(o, self.id_field) not in existing_ids]
        if objs_to_update:
            logger.info(f"in Upsert and updating now in collection: {collection}")
            self.update(objs_to_update, **kwargs)

        if objs_to_insert:
            logger.info(f"in Upsert and inserting now in collection: {collection}")
            self.insert(objs_to_insert, **kwargs)

    def _process_objects(
        self,
        objs: Union[OBJECT, Iterable[OBJECT]],
        collection: str = None,
        batch_size: int = None,
        object_type: str = None,
        model: str = None,
        distance: str = None,
        text_field: Union[str, Callable] = None,
        method: str = "insert",
        **kwargs,
    ):
        """
        Process objects by inserting, updating or upserting them into the collection
        :param objs:
        :param collection:
        :param batch_size:
        :param object_type:
        :param model:
        :param text_field:
        :param method:
        :param kwargs:
        :return:
        """
        collection = self._get_collection_name(collection)
        logger.debug(f"Processing objects for collection {collection}")
        self.vec_dimension = self._get_embedding_dimension(model)
        logger.debug(f"Model: {model}, vec_dimension: {self.vec_dimension}")
        if collection not in self.list_collection_names():
            logger.debug(f"Creating table for collection {collection}")
            self._create_table_if_not_exists(
                collection, self.vec_dimension, model=model, distance=distance
            )
        if isinstance(objs, Iterable) and not isinstance(objs, str):
            objs = list(objs)
        else:
            objs = [objs]
        obj_count = len(objs)
        kwargs.update({"object_count": obj_count})
        cm = self.collection_metadata(collection)
        if batch_size is None:
            batch_size = 100000
        if text_field is None:
            text_field = self.text_lookup
        id_field = self.id_field
        sql_command = self._generate_sql_command(collection, method)
        sql_command = sql_command.format(collection=collection)
        if not self._is_openai(collection):
            for next_objs in chunk(objs, batch_size):
                next_objs = list(next_objs)
                docs = [self._text(o, text_field) for o in next_objs]
                metadatas = [self._dict(o) for o in next_objs]
                ids = [self._id(o, id_field) for o in next_objs]
                embeddings = self._embedding_function(docs, cm.model)
                try:
                    self.conn.execute("BEGIN TRANSACTION;")
                    self.conn.executemany(
                        sql_command, list(zip(ids, metadatas, embeddings, docs))  # noqa: B905
                    )
                    self.conn.execute("COMMIT;")
                except Exception as e:
                    self.conn.execute("ROLLBACK;")
                    logger.error(
                        f"Transaction failed: {e}, default model: {self.default_model}, model used: {model}, len(embeddings): {len(embeddings[0])}"
                    )
                    raise
                finally:
                    self.create_index(collection)
        else:
            if model.startswith("openai:"):
                openai_model = model.split(":", 1)[1]
                if openai_model == "" or openai_model not in MODEL_MAP.keys():
                    logger.info(
                        f"The model {openai_model} is not "
                        f"one of {MODEL_MAP.keys()}. Defaulting to {DEFAULT_OPENAI_MODEL}"
                    )
                    openai_model = DEFAULT_OPENAI_MODEL  # ada 002
                else:
                    logger.error(f"Something went wonky ## model: {model}")
            from transformers import GPT2Tokenizer

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            for next_objs in chunk(objs, batch_size):  # Existing chunking
                next_objs = list(next_objs)
                docs = [self._text(o, text_field) for o in next_objs]
                metadatas = [self._dict(o) for o in next_objs]
                ids = [self._id(o, id_field) for o in next_objs]

                tokenized_docs = [tokenizer.encode(doc) for doc in docs]
                current_batch = []
                current_token_count = 0
                batch_embeddings = []

                i = 0
                while i < len(tokenized_docs):
                    doc_tokens = tokenized_docs[i]
                    # peek
                    if current_token_count + len(doc_tokens) <= 8192:
                        current_batch.append(doc_tokens)
                        current_token_count += len(doc_tokens)
                        i += 1
                    else:
                        if current_batch:
                            logger.info(f"Tokens: {current_token_count}")
                            texts = [tokenizer.decode(tokens) for tokens in current_batch]
                            short_name, _ = MODEL_MAP[openai_model]
                            embedding_model = llm.get_embedding_model(short_name)
                            logger.info(f"Number of texts/docs to embed in batch: {len(texts)}")
                            embeddings = list(embedding_model.embed_multi(texts, len(texts)))
                            logger.info(f"Number of Documents in batch: {len(embeddings)}")
                            batch_embeddings.extend(embeddings)

                        if len(doc_tokens) > 8192:
                            logger.warning(
                                f"Document with ID {ids[i]} exceeds the token limit alone and will be skipped."
                            )
                            # try:
                            #     embeddings = OpenAIEmbeddings(model=model, tiktoken_model_name=model).embed_query(texts,
                            #     embeddings.average                                                                                model)
                            #     batch_embeddings.extend(embeddings)
                            # skipping
                            i += 1
                            continue
                        else:
                            current_batch = []
                            current_token_count = 0

                if current_batch:
                    logger.info(f"Last batch, token count: {current_token_count}")
                    texts = [tokenizer.decode(tokens) for tokens in current_batch]
                    short_name, _ = MODEL_MAP[openai_model]
                    embedding_model = llm.get_embedding_model(short_name)
                    embeddings = list(embedding_model.embed_multi(texts))
                    batch_embeddings.extend(embeddings)
                logger.info(
                    f"Trying to insert: {len(ids)} IDS, {len(metadatas)} METADATAS, {len(batch_embeddings)} EMBEDDINGS"
                )
                try:
                    self.conn.execute("BEGIN TRANSACTION;")
                    self.conn.executemany(
                        sql_command, list(zip(ids, metadatas, batch_embeddings, docs, strict=False))
                    )
                    self.conn.execute("COMMIT;")
                except Exception as e:
                    self.conn.execute("ROLLBACK;")
                    logger.error(
                        f"Transaction failed: {e}, default model: {self.default_model}, model used: {model}, len(embeddings): {len(embeddings[0])}"
                    )
                    raise
                finally:
                    self.create_index(collection)


    def create_collection(self, collection: str = None, model: str = None, vec_dimension: int = None, distance: str = None):
        """
        Create a table with the given name. Metadata will be filled with default values from class DuckDBAdapter if not specified.
        :param collection:
        :param model:
        :param vec_dimension:
        :param distance:
        :return:
        """
        if model is None:
            model = self.default_model
        if distance is None:
            distance = self.distance_metric
        if vec_dimension is None:
            vec_dimension = self.vec_dimension
        sql_safe_collection_name = f'"{collection}"'
        # using default embedding function as in chroma "all-MiniLM-L6-v2"
        self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {sql_safe_collection_name} (
                id VARCHAR PRIMARY KEY,
                metadata JSON,
                embeddings FLOAT[{vec_dimension}],
                documents TEXT
            )
        """)

        metadata = CollectionMetadata(name=collection, model=model, hnsw_space=distance)
        metadata_json = json.dumps(metadata.dict(exclude_none=True))
        sql_safe_collection_name = f'"{collection}"'
        self.conn.execute(
            f"""
                            INSERT INTO {sql_safe_collection_name} (id, metadata) VALUES ('__metadata__', ?)
                            ON CONFLICT (id) DO NOTHING
                            """,
            [metadata_json],
        )

    def get(self, collection: str, include: List[str] = None, limit = None) -> Iterator[SEARCH_RESULT]:
        if include is None:
            include = {IDS, METADATAS}
        else:
            include = set(include)

        sql_safe_collection_name = f"'{collection}'"
        sql = f"""
            SELECT * FROM {collection} LIMIT ?;   
            """, [limit].fetchall()
        
    def remove_collection(self, collection: str = None, exists_ok=False, **kwargs):
        """
        Remove the collection from the database
        :param collection:
        :param exists_ok:
        :param kwargs:
        :return:
        """
        collection = self._get_collection(collection)
        if not exists_ok:
            if collection not in self.list_collection_names():
                raise ValueError(f"Collection {collection} does not exist")
        # duckdb, requires that identifiers containing special characters ("-") must be enclosed in double quotes.
        sql_safe_collection_name = f'"{collection}"'
        self.conn.execute(f"DROP TABLE IF EXISTS {sql_safe_collection_name}")

    def search(
        self,
        text: str,
        where: QUERY = None,
        collection: str = None,
        limit: int = 10,
        relevance_factor: float = None,
        include=None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        """
        Search for objects in the collection that match the given text
        :param text:
        :param where:
        :param collection:
        :param limit:
        :param relevance_factor:
        :param include:
        :param kwargs:
        :return:
        """
        yield from self._search(
            text=text,
            where=where,
            collection=collection,
            limit=limit,
            relevance_factor=relevance_factor,
            include=include,
            **kwargs,
        )

    def _search(
        self,
        text: str,
        where: QUERY = None,
        collection: str = None,
        limit: int = 10,
        relevance_factor: float = None,
        model: str = None,
        include=None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        if relevance_factor is not None and relevance_factor < 1.0:
            yield from self._diversified_search(
                text=text,
                where=where,
                collection=collection,
                limit=limit,
                include=include,
                relevance_factor=relevance_factor,
                **kwargs,
            )
            return
        if include is None:
            include = {METADATAS, DOCUMENTS, DISTANCES}
        else:
            include = set(include)
        collection = self._get_collection(collection)
        cm = self.collection_metadata(collection)
        logger.info(f"Collection metadata={cm}")
        if model is None:
            if cm:
                model = cm.model
            if model is None:
                model = self.default_model
        logger.info(f"Model={model}")
        where_conditions = []
        if where:
            where_conditions.append(where)
        where_clause = " AND ".join(where_conditions)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        if relevance_factor is not None and relevance_factor < 1.0:
            yield from self._diversified_search(
                text, where, collection, limit, relevance_factor, include, **kwargs
            )
            return
        query_embedding = self._embedding_function(text, model)
        sql_safe_collection_name = f'"{collection}"'

        vec_dimension = self._get_embedding_dimension(model)

        # TODO: !VERY IMPORTANT! distance metrics between Chroma and DuckDB have very different, unclear implementations
        # https://duckdb.org/docs/sql/functions/array.html#array_distancearray1-array2
        # https://docs.trychroma.com/guides
        # chromaDB: by default l2, other options are ip and cosine
        # duckDB: by default none, array_distance() or 1-array_cosine_similarity(), both bring different distances
        # than chromaDBs distance metric
        results = self.conn.execute(
            f"""
            SELECT *, array_distance(embeddings::FLOAT[{vec_dimension}],
            {query_embedding}::FLOAT[{vec_dimension}]) as distance
            FROM {sql_safe_collection_name}
            {where_clause}
            ORDER BY distance
            LIMIT ?
        """,
            [limit],
        ).fetchall()
        yield from self.parse_duckdb_result(results, include)

    def _diversified_search(
        self,
        text: str,
        where: QUERY = None,
        collection: str = None,
        limit: int = 10,
        relevance_factor: float = 0.5,
        include=None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        if limit is None:
            limit = 10
        # we need to set this as we need EMBEDDINGS
        include = {METADATAS, DOCUMENTS, EMBEDDINGS, DISTANCES}
        collection = self._get_collection(collection)
        cm = self.collection_metadata(collection)
        where_conditions = []
        if where:
            for key, value in where.items():
                where_conditions.append(f"json_extract(metadata, '$.{key}') = '{value}'")
        where_clause = " AND ".join(where_conditions)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        query_embedding = self._embedding_function(text, model=cm.model)
        sql_safe_collection_name = f'"{collection}"'
        vec_dimension = self._get_embedding_dimension(cm.model)
        results = self.conn.execute(
            f"""
                    SELECT *, array_distance(embeddings::FLOAT[{vec_dimension}],
                    {query_embedding}::FLOAT[{vec_dimension}]) as distance
                    FROM {sql_safe_collection_name}
                    {where_clause}
                    ORDER BY distance
                    LIMIT ?
                """,
            [limit * 10],
        ).fetchall()
        results = list(self.parse_duckdb_result(results, include))
        if not results:
            return
        rows = [np.array(r[2]["_embeddings"]) for r in results]
        query = np.array(query_embedding)
        reranked_indices = mmr_diversified_search(
            query, rows, relevance_factor=relevance_factor, top_n=limit
        )
        for i in reranked_indices:
            yield results[i]

    def list_collection_names(self):
        """
        List the names of all collections in the database
        :return:
        """
        result = self.conn.execute("PRAGMA show_tables;").fetchall()
        return [row[0] for row in result]

    def collection_metadata(
        self, collection_name: Optional[str] = None, include_derived=False, **kwargs
    ) -> Optional[CollectionMetadata]:
        """
        Get the metadata for the collection
        :param collection_name:
        :param include_derived:
        :param kwargs:
        :return:
        """
        collection_name = self._get_collection(collection_name)
        sql_safe_collection_name = f'"{collection_name}"'
        try:
            result = self.conn.execute(
                f"SELECT metadata FROM {sql_safe_collection_name} WHERE id = '__metadata__'"
            ).fetchone()
            if result:
                metadata = json.loads(result[0])
                metadata_instance = CollectionMetadata(**metadata)
                if include_derived:
                    # not implemented yet
                    # metadata_instance.object_count = compute_object_count(collection_name
                    pass
                return metadata_instance
        except Exception as e:
            logger.error(f"Failed to retrieve metadata for collection {collection_name}: {str(e)}")
            return None

    def update_collection_metadata(self, collection: str, **kwargs):
        """
        Update the metadata for a collection. This function will merge new metadata provided
        via kwargs with existing metadata, if any, ensuring that only the specified fields
        are updated.
        :param collection:
        :param kwargs:
        :return:
        """
        if not collection:
            raise ValueError("Collection name must be provided.")
        current_metadata = self.collection_metadata(collection)
        if current_metadata is None:
            current_metadata = CollectionMetadata(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(current_metadata, key):
                    setattr(current_metadata, key, value)
        metadata_dict = current_metadata.dict(exclude_none=True)
        metadata_json = json.dumps(metadata_dict)
        sql_safe_collection_name = f'"{collection}"'
        self.conn.execute(
            f"""
                UPDATE {sql_safe_collection_name} SET metadata = ?
                WHERE id = '__metadata__'
                """,
            [metadata_json],
        )
        return current_metadata

    def set_collection_metadata(
        self, collection_name: Optional[str], metadata: CollectionMetadata, **kwargs
    ):
        """
        Set the metadata for the collection
        :param collection_name:
        :param metadata:
        :param kwargs:
        :return:
        """
        if collection_name is None:
            raise ValueError("Collection name must be provided.")

        metadata_json = json.dumps(metadata.dict(exclude_none=True))
        sql_safe_collection_name = f'"{collection_name}"'
        self.conn.execute(
            f"""
            UPDATE {sql_safe_collection_name}
            SET metadata = ?
            WHERE id = '__metadata__'
            """,
            [metadata_json],
        )

    def find(
        self,
        where: QUERY = None,
        projection: PROJECTION = None,
        collection: str = None,
        include=None,
        limit: int = 10,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        """
        Find objects in the collection that match the given query and projection

        :param where: the query to filter the results
        :param projection:
        :param collection: name of the collection to search
        :param include: fields to be included in output
        :param limit: maximum number of results to return
        :param kwargs:
        :return:

        Parameters
        ----------
        """
        collection = self._get_collection(collection)
        where_clause = self._parse_where_clause(where) if where else ""
        where_clause = f"WHERE {where_clause}" if where_clause else ""
        if include is None:
            include = [IDS, METADATAS, DOCUMENTS]
        sql_safe_collection_name = f'"{collection}"'
        query = f"""
                    SELECT id, metadata, embeddings, documents, NULL as distance
                    FROM {sql_safe_collection_name}
                    {where_clause}
                    LIMIT {limit}
                """
        results = self.conn.execute(query).fetchall()
        yield from self.parse_duckdb_result(results, include)

    def matches(self, obj: OBJECT, include=None, **kwargs) -> Iterator[SEARCH_RESULT]:
        """
        Find objects in the collection that match the given object
        :param obj:
        :param include:
        :param kwargs:
        :return:
        """
        if include is None:
            include = {IDS, METADATAS, DOCUMENTS, DISTANCES}
        else:
            include = set(include)
        text_field = self.text_lookup
        logger.info(f"## TEXT FIELD:{text_field}")
        text = self._text(obj, text_field)
        logger.info(f"{text}")
        logger.info(f"Query term: {text}")
        yield from self.search(text=text, include=include, **kwargs)

    def lookup(self, id: str, collection: str = None, include=None, **kwargs) -> OBJECT:
        """
        Lookup an object by its id
        :param id: ID of the object to lookup
        :param collection: Name of the collection to search
        :param include: List of fields to include in the output ['metadata', 'embeddings', 'documents']
        :param kwargs:
        :return:
        """
        if include is None:
            include = {METADATAS}
        else:
            include = set(include)
        sql_safe_collection_name = f'"{collection}"'
        result = self.conn.execute(
            f"""
                SELECT *
                FROM {sql_safe_collection_name}
                WHERE id = ?
            """,
            [id],
        ).fetchone()
        if isinstance(result, tuple) and len(result) > 1:
            search_result = DuckDBSearchResult(
                ids=result[0],
                metadatas=json.loads(result[1]),
                embeddings=result[2],
                documents=result[3],
                include=include,
            )
            return search_result.to_dict().get(METADATAS)

    def peek(
        self, collection: str = None, limit=5, include=None, offset: int = 0, **kwargs
    ) -> Iterator[SEARCH_RESULT]:
        """
        Peek at the first N objects in the collection
        :param collection:
        :param limit:
        :param include:
        :param offset:
        :param kwargs:
        :return:
        """
        if include is None:
            include = {IDS, METADATAS, DOCUMENTS}
        else:
            include = set(include)
        sql_safe_collection_name = f'"{collection}"'
        results = self.conn.execute(
            f"""
                SELECT id, metadata, embeddings, documents, NULL as distance
                FROM {sql_safe_collection_name}
                LIMIT ?
            """,
            [limit],
        ).fetchall()

        yield from self.parse_duckdb_result(results, include)

    def fetch_all_objects_memory_safe(self, collection: str = None, batch_size: int = 100, include=None, **kwargs) -> Iterator[
        OBJECT]:
        """
        Fetch all objects from a collection, in batches to avoid memory overload.
        """
        collection = self._get_collection(collection)
        offset = 0
        while True:
            if include is None:
                include = {IDS, METADATAS, DOCUMENTS, EMBEDDINGS}
            else:
                include = set(include)
            sql_safe_collection_name = f'"{collection}"'
            query = f"""
                                SELECT *
                                FROM {sql_safe_collection_name}
                                LIMIT ? OFFSET ?
                            """
            results = self.conn.execute(query, [batch_size, offset]).fetchall()
            if results:
                yield from self.parse_duckdb_result(results, include)
                offset += batch_size
            else:
                break


    def get_raw_objects(self, collection) -> Iterator[Dict]:
        """
        Get all raw objects (metadata) in the collection as they were inserted into the database
        :param collection:
        :return:
        """
        sql_safe_collection_name = f'"{collection}"'
        results = self.conn.execute(
            f"""
                SELECT metadata
                FROM {sql_safe_collection_name}
            """
        ).fetchall()
        for result in results:
            yield json.loads(result[0])

    def dump_then_load(
        self,
        collection: str = None,
        target: DBAdapter = None,
    ):
        """
        Dump the collection to a file and then load it into the target adapter
        :param collection:
        :param target:
        :param temp_file:
        :param format:
        :return:
        """
        if collection is None:
            raise ValueError("Collection name must be provided.")
        if not isinstance(target, DuckDBAdapter):
            raise ValueError("Target must be a DuckDBAdapter instance")

        result = self.get_raw_objects(collection)

        metadata = self.collection_metadata(collection)
        model = metadata["model"]
        vec_dimension = self._get_embedding_dimension(model)
        distance = metadata["hnsw_space"]
        # in case it exists already, remove
        target.remove_collection(collection, exists_ok=True)
        # using same collection name in target database
        target._create_table_if_not_exists(collection, vec_dimension, distance, model)
        target.set_collection_metadata(collection, metadata)
        batch_size = 5000
        for i in range(0, len(list(result)), batch_size):
            batch = result[i : i + batch_size]
            target.insert(batch, collection=collection)

    @staticmethod
    def kill_process(pid):
        """
        Kill the process with the given PID
        Returns
        -------

        """
        process = None
        try:
            process = psutil.Process(pid)
            process.terminate()  # Sends SIGTERM
            process.wait(timeout=5)
        except psutil.NoSuchProcess:
            logger.info("Process already terminated.")
        except psutil.TimeoutExpired:
            if process is not None:
                logger.warning("Process did not terminate in time, forcing kill.")
                process.kill()  # Sends SIGKILL as a last resort
        except Exception as e:
            logger.error(f"Failed to terminate process: {e}")

    @staticmethod
    def _generate_sql_command(collection: str, method: str) -> str:
        sql_safe_collection_name = f'"{collection}"'
        if method == "insert":
            return f"""
                INSERT INTO {sql_safe_collection_name} (id,metadata, embeddings, documents) VALUES (?, ?, ?, ?)
                """
        else:
            raise ValueError(f"Unknown method: {method}")

    def _is_openai(self, collection: str) -> bool:
        """
        Check if the collection uses a OpenAI Embedding model
        :param collection:
        :return:
        """
        collection = self._get_collection(collection)
        sql_safe_collection_name = f'"{collection}"'
        query = f"SELECT metadata FROM {sql_safe_collection_name} WHERE id = '__metadata__'"
        result = self.conn.execute(query).fetchone()
        if result:
            metadata = json.loads(result[0])
            if "model" in metadata and metadata["model"].startswith("openai:"):
                return True
        return False


    @staticmethod
    def parse_duckdb_result(results, include) -> Iterator[SEARCH_RESULT]:
        """
        Parse the results from the SQL
        :return: DuckDBSearchResultIterator
        ----------
        """
        for res in results:
            if res[0] != "__metadata__":
                D = DuckDBSearchResult(
                    ids=res[0],
                    metadatas=json.loads(res[1]),
                    embeddings=res[2],
                    documents=res[3],
                    distances=res[4],
                    include=include,
                )
                yield from D.__iter__()

    @staticmethod
    def _parse_where_clause(where: Dict[str, Any]) -> str:
        """
        Parse the where clause from the query
        Parameters
        ----------
        where

        Returns
        -------

        """
        conditions = []
        for key, condition in where.items():
            if isinstance(condition, dict):
                for op, value in condition.items():
                    if op == "$eq":
                        conditions.append(f"json_extract_string(metadata, '$.{key}') = '{value}'")
                    elif op == "$ne":
                        conditions.append(f"json_extract_string(metadata, '$.{key}') != '{value}'")
                    elif op == "$gt":
                        conditions.append(
                            f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) > '{value}'"
                        )
                    elif op == "$gte":
                        conditions.append(
                            f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) >= '{value}'"
                        )
                    elif op == "$lt":
                        conditions.append(
                            f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) < '{value}'"
                        )
                    elif op == "$lte":
                        conditions.append(
                            f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) <= '{value}'"
                        )
                    elif op == "$in":
                        conditions.append(
                            f"json_extract_string(metadata, '$.{key}') IN ({', '.join([f'{v}' for v in value])})"
                        )
                    elif op == "$nin":
                        conditions.append(
                            f"json_extract_string(metadata, '$.{key}') NOT IN ({', '.join([f'{v}' for v in value])})"
                        )
                    elif op == "$exists":
                        if value:
                            conditions.append(f"json_extract(metadata, '$.{key}') IS NOT NULL")
                        else:
                            conditions.append(f"json_extract(metadata, '$.{key}') IS NULL")
                    elif op == "$regex":
                        conditions.append(f"json_extract_string(metadata, '$.{key}') ~ '{value}'")
            else:
                conditions.append(f"json_extract_string(metadata, '$.{key}') = '{condition}'")
        return " AND ".join(conditions)

    def _get_embedding_dimension(self, model_name: str) -> int:
        if model_name is None or model_name.startswith(self.default_model):
            return DEFAULT_MODEL[self.default_model]
        if isinstance(model_name, str):
            if model_name.startswith("openai:"):
                model_key = model_name.split("openai:", 1)[1]
                if model_key == "" or model_key not in MODEL_MAP.keys():
                    model_key = DEFAULT_OPENAI_MODEL
                model_info = MODEL_MAP.get(model_key, DEFAULT_OPENAI_MODEL)
                return model_info[1]
            else:
                return MODEL_MAP[DEFAULT_OPENAI_MODEL][1]
