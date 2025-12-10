from transformers import LlamaForCausalLM
from transformers import LlamaConfig
from dotenv import load_dotenv
import os

load_dotenv()

config = LlamaConfig(
    hidden_size=int(os.getenv('HIDDEN_SIZE')),
    num_attention_heads=int(os.getenv('NUM_ATTENTION_HEADS')),
    vocab_size=int(os.getenv('VOCAB_SIZE')),
    intermediate_size=int(os.getenv('INTERMEDIATE_SIZE')),
    num_hidden_layers=int(os.getenv('NUM_HIDDEN_LAYERS')),
    max_position_embeddings=int(os.getenv('MAX_POSITION_EMBEDDINGS')),
    rms_norm_eps=float(os.getenv('RMS_NORM_EPS'))
)

model = LlamaForCausalLM(config = config)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params / 1e6:.2f}M")