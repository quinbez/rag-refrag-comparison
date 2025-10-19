import torch
import torch.nn as nn
from retrievers.relevance_policy import RelevancePolicy

class RelevancePolicyTrainer:
    """
    Trainer for the lightweight relevance policy network.
    """
    def __init__(self, policy: RelevancePolicy, learning_rate: float = 1e-4):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss()

    def train_step(self, query_emb: torch.Tensor, chunk_emb: torch.Tensor, relevance_label: float) -> float:
        self.optimizer.zero_grad()
        pred_score = self.policy(query_emb, chunk_emb)
        target = torch.tensor([relevance_label], dtype=torch.float32)
        loss = self.loss_fn(pred_score.unsqueeze(0), target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self, path: str):
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        self.policy.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
