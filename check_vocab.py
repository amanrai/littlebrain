from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

# Load tokenizer
model_name = os.getenv('MODEL_NAME')
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer model max length: {tokenizer.model_max_length}")
except Exception as e:
    print(f"Error loading tokenizer from {model_name}: {e}")
    # Fallback to a working tokenizer
    print("Using GPT2 tokenizer as fallback...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"GPT2 tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"GPT2 tokenizer model max length: {tokenizer.model_max_length}")