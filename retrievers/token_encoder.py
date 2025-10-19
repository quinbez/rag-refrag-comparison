import torch
from transformers import AutoTokenizer, AutoModel

class TokenLevelEncoder:
    """
    Encodes text at token level using a small, memory-efficient model (all-MiniLM-L6-v2).
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, text: str, max_length: int = 128) -> torch.Tensor:
        """
        Returns token-level embeddings for input text.

        Args:
            text (str): Input string.
            max_length (int): Maximum number of tokens to encode.

        Returns:
            torch.Tensor: Token-level embeddings of shape (seq_len, hidden_dim)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # last_hidden_state has shape (batch=1, seq_len, hidden_dim)
            token_embeddings = outputs.last_hidden_state.squeeze(0)

        return token_embeddings
