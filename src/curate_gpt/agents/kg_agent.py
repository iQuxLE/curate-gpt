import os
import random
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import wget


@dataclass
class KGDownloadAgent:
    def download_and_process_data(
            self,
            url="https://data.monarchinitiative.org/monarch-kg/2024-02-13/monarch-kg.tar.gz"
    ) -> dict:
        # Download the knowledge graph data
        # Process the data and extract the required information
        # Return a dictionary containing the processed data
        tmpdir = tempfile.TemporaryDirectory()
        tmpfile = tempfile.NamedTemporaryFile().file.name
        wget.download(url, tmpfile)
        this_tar = tarfile.open(tmpfile, "r:gz")
        this_tar.extractall(path=tmpdir.name)
        edge_files = [f for f in os.listdir(tmpdir.name) if "edges" in f]
        if len(edge_files) != 1:
            raise RuntimeError(
                "Didn't find exactly one edge file in {}".format(tmpdir.name)
            )
        edge_file = edge_files[0]

        edges_df = pd.read_csv(os.path.join(tmpdir.name, edge_file), sep="\t")

        return {
            "human_gene_to_phenotype": self.process_human_gene_to_phenotype(df=edges_df),
            "mouse_gene_to_phenotype": self.process_mouse_gene_to_phenotype(df=edges_df),
            "orthologues_genes_human_to_mouse_dict": self.process_orthologues_genes_human_to_mouse(df=edges_df),
            "ortholgues_genes_human_to_mouse_set": self.process_orthologues_genes_human_to_mouse_set(df=edges_df)
        }

    @staticmethod
    def process_human_gene_to_phenotype(df: pd.DataFrame) -> dict:
        # Process and return the human gene to phenotype mapping
        gene_to_phenotype = df[(df['subject'].str.startswith('HGNC')) & (df['predicate'] == 'biolink:has_phenotype')]
        gene_to_phenotype.groupby('subject')['object'].agg(list).reset_index()
        return gene_to_phenotype.set_index('subject')['object'].to_dict()

    @staticmethod
    def process_mouse_gene_to_phenotype(df: pd.DataFrame) -> dict:
        # Process and return the mouse gene to phenotype mapping
        gene_to_phenotype_mouse = df[
            (df['subject'].str.startswith('MGI:')) & (df['predicate'] == 'biolink:has_phenotype')]
        gene_to_phenotype_mouse.groupby('subject')['object'].agg(list).reset_index()
        return gene_to_phenotype_mouse.set_index('subject')['object'].to_dict()

    @staticmethod
    def process_orthologues_genes_human_to_mouse(df: pd.DataFrame) -> dict:
        # Process and return the orthologous genes mapping between human and mouse
        hom_gene_to_gene = df[(df['predicate'].str.startswith("biolink:orthologoues_to")) & (
            df['subject'].str.startswith("HGNC:")) & (df['object'].str.startswith("MGI:"))]
        hom_gene_to_gene.groupby('subject')['object'].agg(list).reset_index()
        return hom_gene_to_gene.set_index('subject')['object'].to_dict

    @staticmethod
    def process_orthologues_genes_human_to_mouse_set(df: pd.DataFrame) -> set[Tuple]:
        # Process and return the orthologous genes mapping between human and mouse as a set of tuples
        hom_gene_to_gene = df[(df['predicate'].str.startswith("biolink:orthologoues_to")) &
                              (df['subject'].str.startswith("HGNC:")) & (df['object'].str.startswith("MGI:"))]

        homologues_pairs = set(zip(hom_gene_to_gene['subject'], hom_gene_to_gene['object']))
        return homologues_pairs

    @staticmethod
    def create_non_homologues_pairs(homologues_pairs: set[Tuple]) -> set[Tuple]:
        human_genes, mouse_genes = zip(*homologues_pairs)
        human, mouse = list(human_genes), list(mouse_genes)
        random.shuffle(mouse)
        new_pairs = set(zip(human, mouse))

        def is_valid_set(altered_pairs, original_pairs):
            for pair in altered_pairs:
                if pair in original_pairs:
                    return False
            return True

        while not is_valid_set(
                altered_pairs=new_pairs,
                original_pairs=homologues_pairs
                ):
            random.shuffle(mouse)
            new_pairs = set(zip(human, mouse))

        return new_pairs
