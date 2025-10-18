import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    """
    A simple wrapper for a Hugging Face sequence-to-sequence model
    that handles tokenization and text generation.
    """
    def __init__(self, model_name="google/flan-t5-small"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            sys.exit(1)

    def generate(self, prompt, max_tokens=150):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"ERROR in generation: {e}")
            return f"Generation failed: {str(e)}"