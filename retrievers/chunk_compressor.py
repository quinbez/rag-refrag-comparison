import torch
import torch.nn as nn

class ChunkCompressor(nn.Module):
    """
    Compresses token-level embeddings into fixed-size chunk representations.
    """
    def __init__(self, input_dim=384, output_dim=128):
        """
        Args:
            input_dim (int): Dimension of token embeddings (e.g., 384 for MiniLM).
            output_dim (int): Desired compressed chunk size.
        """
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compress token embeddings to a fixed-size vector.

        Args:
            token_embeddings: Tensor of shape (seq_len, hidden_dim)

        Returns:
            compressed: Tensor of shape (output_dim,)
        """
        attn_weights = torch.softmax(self.attention(token_embeddings), dim=0)
        weighted = (token_embeddings * attn_weights).sum(dim=0)
        compressed = self.layer_norm(self.projection(weighted))

        return compressed
