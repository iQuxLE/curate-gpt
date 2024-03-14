from dataclasses import dataclass
from typing import Dict, Tuple

from src.curate_gpt.agents.kg_agent import KGDownloadAgent
from src.curate_gpt.wrappers.base_wrapper import BaseWrapper


@dataclass
class KGWrapper(BaseWrapper):
    def __post_init__(self):
        self.download_agent = KGDownloadAgent()
        self.data = self.download_agent.download_and_process_data()

    @property
    def human_gene_to_phenotype(self) -> Dict:
        return self.data["human_gene_to_phenotype"]

    @property
    def mouse_gene_to_phenotype(self) -> Dict:
        return self.data["mouse_gene_to_phenotype"]

    @property
    def orthologues_genes_human_to_mouse_dict(self) -> Dict:
        return self.data["orthologues_genes_human_to_mouse_dict"]

    @property
    def orthologues_genes_human_to_mouse_set(self) -> set[Tuple]:
        return  self.data["orthologues_genes_human_to_mouse_set"]
