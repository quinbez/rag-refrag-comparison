import torch
import torch.nn as nn

class RelevancePolicy(nn.Module):
    """
    Lightweight RL-trained relevance policy network.
    Both query and chunk use compressed embeddings (128-dim).
    """
    def __init__(self, chunk_dim=128, query_dim=128, hidden_dim=64):
        super().__init__()

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.chunk_proj = nn.Linear(chunk_dim, hidden_dim)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, query_emb: torch.Tensor, chunk_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: (128,) compressed query embedding
            chunk_emb: (128,) compressed chunk embedding
        Returns:
            score: scalar [0, 1]
        """
        q_proj = self.query_proj(query_emb)
        c_proj = self.chunk_proj(chunk_emb)
        combined = torch.cat([q_proj, c_proj], dim=-1)
        score = self.scorer(combined)
        return score.squeeze()