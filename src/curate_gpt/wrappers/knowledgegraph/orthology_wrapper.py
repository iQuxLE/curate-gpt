from dataclasses import field, dataclass
from typing import Optional

from curate_gpt.agents.kg_agent import KGAgent
from curate_gpt.wrappers.knowledgegraph.kg_wrapper import KGWrapper

@dataclass
class OrthologyHandler:
    """
    Wraps KG Agent for Orthology tasks
    """
    url: str
    agent: KGAgent = field(init=False)

    # gene_list: Optional[list] = field(init=False, repr=False, default=None) # if None all

    def __post_init__(self):
        if self.url is not None:
            self.agent = KGAgent(
                url=self.url,
            )
        self.data = self.agent.download_and_create_df()
        _ = self.agent.human_genes_with_phenotypes  # Explicitly initialize human genes with phenotypes
        _ = self.agent.mouse_genes_with_phenotypes  # Explicitly initialize mouse genes with phenotypes
        _ = self.agent.orthologous_pairs_with_phenotypes  # Explicitly initialize orthologous pairs with phenotypes
    def orthologous_pairs_with_phenotypes(self):
        return self.agent.orthologous_pairs_with_phenotypes

    def get_1000_random_orthologous_pairs(self):
        """use in gene orthology to compare"""
        return self.agent.random_1000_orthologous_pairs

    def get_1000_random_non_orthologous_pairs(self):
        """use in gene-orthology to compare"""
        return self.agent.random_1000_non_orthologous_pairs

    # def get_orthologous_pairs_for_gene_list(self):
    #     """Use if a gene list is provided via CLI"""
    #     return self.agent.get_orthologus_pairs_for_list

