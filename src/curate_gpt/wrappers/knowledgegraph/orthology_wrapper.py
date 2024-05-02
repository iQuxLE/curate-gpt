from dataclasses import field, dataclass
from curate_gpt.agents.kg_agent import KGAgent

@dataclass
class OrthologyHandler:
    """
    Wraps KG Agent for Orthology tasks
    """
    url: str
    agent: KGAgent = field(init=False)

    def __post_init__(self):
        if self.url is not None:
            self.agent = KGAgent(
                url=self.url,
            )
        self.data = self.agent.download_and_create_df()
        _ = self.agent.human_genes_with_phenotypes
        _ = self.agent.mouse_genes_with_phenotypes
        _ = self.agent.orthologous_pairs_with_phenotypes
    def orthologous_pairs_with_phenotypes(self):
        return self.agent.orthologous_pairs_with_phenotypes

    def get_1000_random_orthologous_pairs(self):
        """use in gene orthology to compare"""
        return self.agent.random_1000_orthologous_pairs

    def get_1000_random_non_orthologous_pairs(self):
        """use in gene-orthology to compare"""
        return self.agent.random_1000_non_orthologous_pairs


