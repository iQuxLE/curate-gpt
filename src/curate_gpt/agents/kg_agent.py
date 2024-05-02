import json
import os
import random
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterator, List, Optional

import pandas as pd
import requests
import wget
from tqdm import tqdm

CACHE_FILE = "edges_df_cache.parquet"


@dataclass
class KGAgent:
    url: str
    _edges_df: pd.DataFrame = field(init=False, repr=False, default=None)
    _human_gene_to_pheno: dict = field(init=False, repr=False, default=None)
    _mouse_gene_to_pheno: dict = field(init=False, repr=False, default=None)

    # for orthology
    _human_genes_with_phenotypes: set = field(init=False, repr=False, default=None)
    _mouse_genes_with_phenotypes: set = field(init=False, repr=False, default=None)
    _orthologous_pairs_with_phenotypes: list = field(init=False, repr=False, default=None)
    _random_1000_orthologous_pairs: list = field(init=False, repr=False, default=None)
    _random_1000_non_orthologous_pairs: list = field(init=False, repr=False, default=None)

    @property
    def edges_df(self):
        if self._edges_df is None:
            self._edges_df = self.download_and_create_df()
        return self._edges_df

    @property
    def all_human_genes_to_pheno(self) -> dict:
        """ Loads all human genes and given phenotypes into a dict"""
        if self._human_gene_to_pheno is None:
            self._human_gene_to_pheno = self._process_human_gene_to_pheno()
        return self._human_gene_to_pheno

    @property
    def all_mouse_genes_to_pheno(self) -> dict:
        """ Loads all human genes and given phenotypes into a dict"""
        if self._mouse_gene_to_pheno is None:
            self._mouse_gene_to_pheno = self._process_mouse_gene_to_pheno()
        return self._mouse_gene_to_pheno
    @property
    def human_genes_with_phenotypes(self):
        if self._human_genes_with_phenotypes is None:
            self._human_genes_with_phenotypes = self.get_genes_with_phenotypes("HGNC:", "HP:")
        return self._human_genes_with_phenotypes

    @property
    def mouse_genes_with_phenotypes(self):
        if self._mouse_genes_with_phenotypes is None:
            self._mouse_genes_with_phenotypes = self.get_genes_with_phenotypes("MGI:", "MP:")
        return self._mouse_genes_with_phenotypes

    @property
    def orthologous_pairs_with_phenotypes(self):
        if self._orthologous_pairs_with_phenotypes is None:
            self._orthologous_pairs_with_phenotypes = self._get_orthologous_pairs_with_phenotypes()
        return self._orthologous_pairs_with_phenotypes
    @property
    def random_1000_orthologous_pairs(self):
        if self._random_1000_orthologous_pairs is None:
            self._random_1000_orthologous_pairs = self._process_random_1000_orthologous()
        return self._random_1000_orthologous_pairs
    @property
    def random_1000_non_orthologous_pairs(self):
        if self._random_1000_non_orthologous_pairs is None:
            self._random_1000_non_orthologous_pairs = self._process_random_1000_non_orthologues_pairs(target_count=1000)
        return self._random_1000_non_orthologous_pairs

    ## TODO THE LIST THING
    def all_genes_and_phenotypes(
            self,
            # gene_ids: list,
            gene_prefix: str,
            phenotype_prefix: str,
            predicate: str = 'biolink:has_phenotype'
    ):
        """
        processes the whole collection of genes with phenotypes
        """

        filtered_df = self._edges_df[self._edges_df['predicate'] == predicate]
        if gene_prefix:
            filtered_df = filtered_df[filtered_df['subject'].str.startswith(gene_prefix)]
            # filtered_df = filtered_df[filtered_df['subject'].isin(gene_ids)]
        if phenotype_prefix:
            filtered_df = filtered_df[filtered_df['object'].str.startswith(phenotype_prefix)]
        grouped = filtered_df.groupby('subject')['object'].agg(lambda x: list(set(x))).reset_index()
        return grouped.set_index('subject')['object'].to_dict()

    def given_set_genes_and_phenotypes(
            self,
            gene_ids: list,
            gene_prefix: str,
            phenotype_prefix: str,
            predicate: str = 'biolink:has_phenotype'
    ):
        """
        processes the input list only of genes with phenotypes
        """
        filtered_df = self._edges_df[self._edges_df['predicate'] == predicate]
        if gene_prefix:
            filtered_df = filtered_df[filtered_df['subject'].str.startswith(gene_prefix)]
            filtered_df = filtered_df[filtered_df['subject'].isin(gene_ids)]
        if phenotype_prefix:
            filtered_df = filtered_df[filtered_df['object'].str.startswith(phenotype_prefix)]
        grouped = filtered_df.groupby('subject')['object'].agg(lambda x: list(set(x))).reset_index()
        return grouped.set_index('subject')['object'].to_dict()

    def _process_human_gene_to_pheno(
            self
    ) -> dict:
        gene_to_phenotype = self._edges_df[(self._edges_df['subject'].str.startswith('HGNC')) & (
                    self._edges_df['predicate'] == 'biolink_has_phenotype')]
        gene_to_phenotype.groupby('subject')['object'].agg(lambda x: list(set(x))).reset_index()
        return gene_to_phenotype.set_index('subject')['object'].to_dict()

    def _process_mouse_gene_to_pheno(
            self
    ) -> dict:
        gene_to_phenotype_mouse = self._edges_df[(self._edges_df['subject'].str.startswith('MGI:')) & (
                    self._edges_df['predicate'] == 'biolink:has_phenotype')]
        gene_to_phenotype_mouse.groupby('subject')['object'].agg(lambda x: list(set(x))).reset_index()
        return gene_to_phenotype_mouse.set_index('subject')['object'].to_dict()

    # use for Mouse and Human once u run either mouse or human
    def get_genes_with_phenotypes(self, gene_prefix: str, phenotype_prefix: str) -> set:
        filtered_df = self._edges_df[
            (self._edges_df['predicate'] == 'biolink:has_phenotype') &
            (self._edges_df['subject'].str.startswith(gene_prefix)) &
            (self._edges_df['object'].str.startswith(phenotype_prefix))
            ]
        return set(filtered_df['subject'])

    def _get_orthologous_pairs_with_phenotypes(self, orthologous_to: str = "biolink:orthologous_to") -> List[
        Tuple[str, str]]:
        filtered_df = self._edges_df[
            (self._edges_df['predicate'] == orthologous_to) &
            (self._edges_df['subject'].isin(self._human_genes_with_phenotypes)) &
            (self._edges_df['object'].isin(self._mouse_genes_with_phenotypes))
            ]
        return list(zip(filtered_df['subject'], filtered_df['object']))

    def _process_random_1000_orthologous(
            self
    ) -> list[Tuple[str, str]]:
        if len(self._orthologous_pairs_with_phenotypes) < 1000:
            raise ValueError("Not enough pairs to select 1000 unique items")

        self._random_1000_orthologous_pairs = random.sample(self._orthologous_pairs_with_phenotypes, 1000)
        return self._random_1000_orthologous_pairs

    def _process_random_1000_non_orthologues_pairs(
            self,
            target_count: int,
            current_pairs: list[Tuple[str, str]] = None,
            attempt: int = 0,
            max_attempts: int = 100
    ) -> list[Tuple[str, str]]:
        """
        gets random NON ortholgous pairs from the subset of orthologous pairs with phenotypes
        """

        if current_pairs is None:
            current_pairs = []

        if len(current_pairs) >= target_count:
            return current_pairs

        human_genes, mouse_genes = zip(*self._orthologous_pairs_with_phenotypes)
        human = list(set(human_genes))
        mouse = list(set(mouse_genes))

        while len(current_pairs) < target_count and attempt < max_attempts:
            random.shuffle(mouse)
            potential_new_pairs = set(zip(human, mouse[:len(human)]))

            for pair in potential_new_pairs:
                if pair not in self._orthologous_pairs_with_phenotypes and tuple(reversed(pair)) not in self._orthologous_pairs_with_phenotypes:
                    current_pairs.append(pair)
                    if len(current_pairs) == target_count:
                        self._random_1000_non_orthologous_pairs = list(current_pairs)

                        return current_pairs

            attempt += 1

        current_pairs = self._process_random_1000_non_orthologues_pairs(
            target_count=target_count,
            current_pairs=current_pairs,
            attempt=attempt,
            max_attempts=max_attempts
        )
        self._process_random_1000_non_orthologous_pairs = list(current_pairs)
        return current_pairs


    def download_and_create_df(self):
        if os.path.exists(CACHE_FILE):
            self._edges_df = pd.read_parquet(CACHE_FILE)
            print("KG taken from Cache")
            return self._edges_df

        start = time.time()
        tmpdir = tempfile.TemporaryDirectory()
        tmpfile = tempfile.NamedTemporaryFile(suffix=".tar.gz").name

        # setup progress tracking
        response = requests.get(self.url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading KG file")

        # download with progress tracking
        with open(tmpfile, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        this_tar = tarfile.open(tmpfile, "r:gz")
        this_tar.extractall(path=tmpdir.name)
        edge_files = [f for f in os.listdir(tmpdir.name) if "edges" in f]
        if len(edge_files) != 1:
            raise RuntimeError(
                "Didn't find exactly one edge file in {}".format(tmpdir.name)
            )
        edge_file = edge_files[0]
        file_path = os.path.join(tmpdir.name, edge_file)
        self._edges_df = self.read_csv_with_progress(file_path, desc="Reading KG file")
        print("Download time:")
        print(time.time() - start)
        # self._edges_df.to_hdf(CACHE_FILE, key='edges_df', mode='w')
        self._edges_df.to_parquet(CACHE_FILE)
        return self._edges_df

    @staticmethod
    def count_rows(file_path, sep="\t"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def read_csv_with_progress(self, file_path, chunksize=10000, sep="\t", desc="Processing"):
        total_rows = self.count_rows(file_path, sep=sep)
        tqdm_iterator = tqdm(pd.read_csv(file_path, sep=sep, low_memory=False, chunksize=chunksize),
                             total=total_rows // chunksize, desc=desc)
        df = pd.concat(tqdm_iterator, ignore_index=True)
        return df

