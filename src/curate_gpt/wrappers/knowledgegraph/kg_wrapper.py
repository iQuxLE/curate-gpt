import csv
import json
import pickle

import numpy as np
from typing import Dict, Set, Optional
from chromadb import Collection

from dataclasses import dataclass, field
from typing import Tuple, List

from curate_gpt.agents.kg_agent import KGAgent
from curate_gpt.store import ChromaDBAdapter


# pickle for instances
@dataclass
class KGWrapper:
    url: Optional[str]

    agent: KGAgent = field(init=False)
    database = ChromaDBAdapter
    gene_list: Optional[List[str]] = None
    gene_prefix: Optional[str] = None
    phenotype_prefix: Optional[str] = None
    association_prefix: Optional[str] = None

    def __post_init__(self):
        if self.url is not None:
            self.agent = KGAgent(
                url=self.url,
            )
        self.data = self.agent.download_and_create_df()

    @property
    def get_df(self):
        return self.agent.edges_df

    @property
    def human_gene_to_phenotype(self) -> dict:
        return self.agent.all_human_genes_to_pheno

    @property
    def mouse_gene_to_phenotype(self) -> dict:
        return self.agent.all_mouse_genes_to_pheno

    @property
    def get_all_genes_with_phenotypes(self) -> dict:
        return self.agent.all_genes_and_phenotypes(
            # gene_ids=self.gene_list,
            gene_prefix=self.gene_prefix,
            phenotype_prefix=self.phenotype_prefix)

    @property
    def phenotypes_for_gene_set(self) -> dict:
        return self.agent.given_set_genes_and_phenotypes(
            gene_ids=self.gene_list,
            gene_prefix=self.gene_prefix,
            phenotype_prefix=self.phenotype_prefix)

    # @property
    # def orthologues_genes_human_to_mouse_dict(self) -> dict:
    #     return self.data["orthologues_genes_human_to_mouse_dict"]
    #
    # @property
    # def get_orthologous(self) -> list[Tuple[str, str]]:
    #     return self.agent.get_orthologoues(gene_list)
    #
    # @property
    # def orthologues_genes_human_to_mouse_list(self) -> list[Tuple[str, str]]:
    #     return self.data["orthologues_genes_human_to_mouse"]
    #
    # @property
    # def random_1000_ortholgous_pairs(self) -> list[Tuple[str, str]]:
    #     return self.data["1000_orthologous_pairs"]
    #
    # @property
    # def random_1000_non_ortholgous_pairs(self) -> list[Tuple[str, str]]:
    #     return self.data["1000_non_orthologous_pairs"]


class GeneEmbeddingsHandler:
    _gene_embeddings: dict = None
    db: ChromaDBAdapter

    #
    # def __int__(self, db):
    #     self.db = db

    @property
    def gene_embeddings(self):
        return self._gene_embeddings

    # put the gene_embeddings in a file if for example human_gene
    def save_to_file(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    def load_from_file(self, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    # @staticmethod
    # def get_gene_embeddings(gene_to_hp: dict, collection: Collection) -> dict:
    #     """
    #     averages HP terms and inputs averaged embeddings for all of those per Gene
    #     """
    #     keys = [k for c, (k, v) in enumerate(gene_to_hp.items()) if c < 2]
    #     vs = [v for c, (k, v) in enumerate(gene_to_hp.items()) if c < 10]
    #
    #     print(f"gene_to_hps keys: {keys} + {vs}")
    #     gene_embeddings = {}
    #     phenotype_embeddings = {}
    #     col = collection.get(include=['metadatas', 'embeddings'])
    #     for embedding, metadata in zip(col.get("embeddings", {}), col.get("metadatas", {})):
    #         metadata_json = json.loads(metadata["_json"])
    #         phenotype_id = metadata_json.get("original_id")
    #         if phenotype_id:
    #             phenotype_embeddings[phenotype_id] = np.array(embedding)
    #     # {Gene1: [hp1, hp2, hp3]
    #     # Gene2: [hp3,hp4, hp5]
    #     # }
    #     keys = [k for c, (k, v) in enumerate(phenotype_embeddings.items()) if c < 2]
    #     vs = [len(v) for c, (k, v) in enumerate(phenotype_embeddings.items()) if c < 2]
    #     print(f"phenotype embeddings keys: {keys} + {vs}")
    #     for gene, phenotypes in gene_to_hp.items():
    #         # does gene by gene so phenotypes a list for this genes phenotypes
    #         embeddings = np.array(
    #             [phenotype_embeddings[phenotype] for phenotype in phenotypes if phenotype in phenotype_embeddings])
    #         # average whole list of embeddings from phenotypes per gene
    #         if embeddings.size > 0:
    #             average_embedding = np.mean(embeddings, axis=0)
    #             gene_embeddings[gene] = average_embedding
    #     for key, value in gene_embeddings.items():
    #         if value is None:
    #             print(key)
    #     keys = [k for c, (k, v) in enumerate(gene_embeddings.items()) if c < 2]
    #     print(f"gene_emebddings keys: {keys}")
    #     return gene_embeddings
    @staticmethod
    def get_gene_embeddings(gene_to_hp: dict, collection: Collection) -> dict:
        gene_embeddings = {}
        phenotype_embeddings = {}

        # Load all relevant data from the collection
        col = collection.get(include=['metadatas', 'embeddings'])

        # Process and store phenotype embeddings with proper error handling
        for embedding, metadata in zip(col.get("embeddings", []), col.get("metadatas", [])):
            try:
                metadata_json = json.loads(metadata["_json"])
                phenotype_id = metadata_json.get("original_id")
                if phenotype_id:
                    phenotype_embeddings[phenotype_id] = np.array(embedding)
            except json.JSONDecodeError:
                continue

        # Debugging: check if IDs match expectations
        print(f"Sample Phenotype IDs from Collection: {list(phenotype_embeddings.keys())[:5]}")
        phenotype_ids_from_genes = set([ph for phenotypes in gene_to_hp.values() for ph in phenotypes])
        print(f"Sample Phenotype IDs from Genes: {list(phenotype_ids_from_genes)[:5]}")
        print(f"Overlapping IDs: {phenotype_ids_from_genes.intersection(phenotype_embeddings.keys())}")

        # Calculate average embeddings for each gene
        for gene, phenotypes in gene_to_hp.items():
            embeddings = [phenotype_embeddings[ph] for ph in phenotypes if ph in phenotype_embeddings]
            if embeddings:
                gene_embeddings[gene] = np.mean(embeddings, axis=0)

        # Debugging: Final check on what's being returned
        print(f"Gene Embeddings Processed: {len(gene_embeddings)} items.")

        return gene_embeddings

    def upsert_gene_embeddings(self, gene_embeddings: Dict, collection_name: str):
        collection = self.db.client.get_or_create_collection(collection_name)
        print(collection)
        print(f"Gene Embeddings Dict Len: {len(gene_embeddings)}")
        gene_ids = list(gene_embeddings.keys())
        # print(type(gene_ids), gene_ids)
        # e = gene_embeddings.values()
        # k = gene_embeddings.keys()
        print(len(list(gene_embeddings.keys())))
        print(len(list(gene_embeddings.values())))
        embeddings = np.stack(list(gene_embeddings.values()))
        print(type(embeddings), embeddings)
        metadatas = [{"type": "Gene"}] * len(gene_ids)
        try:
            collection.upsert(ids=gene_ids, embeddings=embeddings, metadatas=metadatas)
        except Exception as e:
            print(f"Error in upserting gene embeddings: {e}")

    def upsert_gene_set_embeddings(self, gene_embeddings: Dict, collection_name: str):
        collection = self.db.client.get_or_create_collection(collection_name)

        embeddings = np.stack(list(gene_embeddings.values()))
        avg_embedding = np.mean(embeddings, axis=0)

        try:
            collection.upsert(ids=f"{collection_name}", embeddings=[avg_embedding], metadatas=[{"type": "Set"}])
        except Exception as e:
            print(f"Error in upserting gene embeddings: {e}")

    def create_gene_set_embeddings(
            self,
            gene_to_hps: dict,
            col: Collection,
            collection_name: str
    ) -> None:
        print(f"Gene To Hps Len: {len(gene_to_hps)}")
        self._gene_embeddings = self.get_gene_embeddings(gene_to_hp=gene_to_hps, collection=col)
        self.upsert_gene_set_embeddings(gene_embeddings=self._gene_embeddings, collection_name=collection_name)

    def create_gene_embeddings(
            self,
            gene_to_hps: Dict,
            col: Collection,
            collection_name: str
    ) -> None:
        """
        creates Gene Embeddings from gene_to_hps using either ont_hp or ont_mp
        upserts to collection
        Does it Gene by Gene

        """
        print(f"Handler: Gene To Hps Len: {len(gene_to_hps)}")
        print(f"Hanlder: Col: {col}")
        print(f"collection name: {collection_name}")
        self._gene_embeddings = self.get_gene_embeddings(gene_to_hp=gene_to_hps, collection=col)
        print(f"Handler: Gene Emebddings len: {len(self._gene_embeddings)}")
        self.upsert_gene_embeddings(gene_embeddings=self._gene_embeddings, collection_name=collection_name)


HGNC = str
MGI = str


class EmbeddingAnalyser:
    outfile: str
    # can also be written with one dict thats merged but lets do two
    collection_entity_one: Optional[Collection]
    collection_entity_two: Optional[Collection]

    collection_set_entity_one: Optional[Collection]
    collection_set_entity_two: Optional[Collection]

    # e.g. Human Genes Embeddings
    gene_embeddings_entity_one: Optional[dict]
    # e.g. Mouse Genes Embeddings
    gene_embeddings_entity_two: Optional[dict]

    # should not be optional, need for comparisons
    pairs = Optional[List[Tuple[HGNC, MGI]]]
    """
    [('HGNC:1723', 'MGI:1859866'), ('HGNC:19989', 'MGI:2139135'), ('HGNC:14452', 'MGI:1913406'), ('HGNC:7646', 'MGI:102537'), ('HGNC:7646', 'MGI:97279'), ('HGNC:7646', 'MGI:109201'), ('HGNC:20', 'MGI:2384560')]
    """

    gene_pairs: Optional[List[Tuple[str, str]]]

    @staticmethod
    # def cosine_similarity(vec1, vec2):
    #     dot_product = np.dot(vec1, vec2)
    #     norm_vec1 = np.linalg.norm(vec1)
    #     norm_vec2 = np.linalg.norm(vec2)
    #     similarity = dot_product / (norm_vec1 * norm_vec2)
    #     return similarity
    def cosine_similarity(vec1, vec2):
        # Flatten the arrays to ensure they are 1D (shape (1536,) instead of (1, 1536))
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()

        if vec1.size == 0 or vec2.size == 0:
            print(f"One of the vectors is empty. HGNC vec: {vec1.size} MGI vec: {vec2.size}")
            return 0  # Return a default or error value for empty vectors

        # Calculate the dot product
        dot_product = np.dot(vec1, vec2)

        # Calculate the norms of the vectors
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # Compute cosine similarity
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0  # Avoid division by zero if a vector is zero
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity

    def gene_pairwise_similarity_using_dictionaries(self) -> None:
        """
        Compares embeddings of e.g. orthologous-pairs or non-orthologoues-pairs of two entities
        """
        pairwise_similarities = []

        if self.gene_embeddings_entity_one or self.gene_embeddings_entity_two is None:
            raise RuntimeError(
                """
                First run the make gene_embeddings command twice to create gene_embeddings collection
                that you want to compare, QUOTE TO ME: Alternativly gene embeddings dictionary pickle,
                or a KGWrapper instance with that in a pickle, given that instance can have 2 dicts
                """)

        if self.gene_pairs is None:
            raise RuntimeError("You have to provide a list of ortholgous gene pairs or non-orthologous gene pairs")

        for gene1, gene2 in self.gene_pairs:
            if gene1 in self.gene_embeddings_entity_one and gene2 in self.gene_embeddings_entity_two:
                # typi = np.array(self.gene_embeddings_entity_one[gene1])
                # print(type(typi))
                similarity = self.cosine_similarity(np.array(self.gene_embeddings_entity_one[gene1]),
                                                    np.array(self.gene_embeddings_entity_two[gene2]))
                pairwise_similarities.append((gene1, gene2, similarity))

        with open(f"{self.outfile}.tsv", 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(['Entity_1', 'Entity_2', 'CosineSimilarity'])
            writer.writerows(pairwise_similarities)

    def gene_pairwise_similarity_using_collections(self) -> None:

        """ Gets pairwise similarity of gene pairs from gene embeddings from ChromaDB collections
        """
        results = []
        # for i, (hgnc, mgi) in enumerate(self.pairs):
        #     entity_one_embedding = self.collection_entity_one.get(ids=hgnc, include=["embeddings"])
        #     entity_two_embedding = self.collection_entity_two.get(ids=mgi, include=["embeddings"])
        #     print(entity_one_embedding.keys(), entity_two_embedding.keys())
        #     cosine = self.cosine_similarity(entity_one_embedding['embeddings'], entity_two_embedding['embeddings'])
        #     print(hgnc, mgi, cosine)
        #     results.append((i, hgnc, mgi, cosine))
        # for i, (hgnc, mgi) in enumerate(self.pairs):
        #     entity_one_embedding = self.collection_entity_one.get(ids=hgnc, include=["embeddings"])
        #     entity_two_embedding = self.collection_entity_two.get(ids=mgi, include=["embeddings"])
        #
        #     if 'embeddings' in entity_one_embedding and 'embeddings' in entity_two_embedding:
        #         cosine = self.cosine_similarity(entity_one_embedding['embeddings'], entity_two_embedding['embeddings'])
        #         print(f"Comparing: {hgnc} - {mgi}")
        #         print(f"Cosine similarity: {cosine}")
        #         results.append((i, hgnc, mgi, cosine))
        #     else:
        #         if 'embeddings' not in entity_one_embedding:
        #             print(f"No embedding found for: {hgnc}")
        #         if 'embeddings' not in entity_two_embedding:
        #             print(f"No embedding found for: {mgi}")
        #         results.append((i, hgnc, mgi, 0))
        for i, (hgnc, mgi) in enumerate(self.pairs):
            entity_one_embedding = self.collection_entity_one.get(ids=hgnc, include=["embeddings"])
            entity_two_embedding = self.collection_entity_two.get(ids=mgi, include=["embeddings"])

            # print(f"Comparing: {hgnc} - {mgi}")

            if 'embeddings' not in entity_one_embedding:
                print(f"No embedding found for: {hgnc}")
            if 'embeddings' not in entity_two_embedding:
                print(f"No embedding found for: {mgi}")

            if 'embeddings' in entity_one_embedding and 'embeddings' in entity_two_embedding:
                cosine = self.cosine_similarity(entity_one_embedding['embeddings'], entity_two_embedding['embeddings'])
                # print(f"Cosine similarity: {cosine}")
            else:
                cosine = 0

            results.append((i, hgnc, mgi, cosine))

        with open(f"{self.outfile}.tsv", "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['Index', 'Entity_1', 'Entity_2', 'CosineSimilarity'])
            writer.writerows(results)

    def gene_setwise_similarity_using_collections(self) -> None:
        results = []
        entity_one_embedding = self.collection_set_entity_one.get(include=["embeddings"])
        entity_two_embedding = self.collection_set_entity_two.get(include=["embeddings"])
        cosine = self.cosine_similarity(entity_one_embedding, entity_two_embedding)
        results.append((HGNC, MGI, cosine))

        with open(f"{self.outfile}.tsv", "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['Set_1', 'Set_2', 'CosineSimilarity'])
            writer.writerows(results)
