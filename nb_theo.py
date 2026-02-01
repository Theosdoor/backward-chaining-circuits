# %%
import math
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from transformer_lens import HookedTransformer, HookedTransformerConfig
from src import *

# Select best available device (CUDA, MPS on Apple Silicon, or CPU)
DEVICE = (
    "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
)

# %% 
def setup_model_and_dataset():
    file_name = "dataset.txt"
    n_examples = 300_000
    n_states = 16

    dataset = GraphDataset(n_states, file_name, n_examples)
    
    cfg = HookedTransformerConfig(
        n_layers=6,
        d_model=128,
        n_ctx=dataset.max_seq_length - 1,
        n_heads=1,
        d_mlp=512,
        d_head=128,
        d_vocab=len(dataset.idx2tokens),
        device=DEVICE,
        attention_dir= "causal",
        act_fn="gelu",
    )
    model = HookedTransformer(cfg)

    # Load in the model if weights are in the directory
    if os.path.exists("model.pt"):
        state_dict = torch.load("model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(DEVICE)
    else:
        print("Warning: model.pt not found. Using untrained model.")
        
    return model, dataset

model, dataset = setup_model_and_dataset()

# %%
