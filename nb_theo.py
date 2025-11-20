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
# --- Phase 1: Observational Analysis (Attention Weights) ---
def analyze_register_attention_magnitude(model, dataset):
    print("\n=== Phase 1: Observational Analysis (Attention Weights) ===")
    # 1. Generate a long path that requires registers (Path length > 6)
    # We force a split where the model must use registers
    pred = generate_example(16, 1234, path_length=10, order="backward")
    
    if not is_model_correct(model, dataset, pred):
        print("Model failed on this example, skipping.")
        return

    labels, cache = get_example_cache(pred, model, dataset)
    
    # Identify Register Tokens (Paper says usually indices 36, 38, 39...)
    # We look at the last token position (usually 47) and see what it attends to
    final_pos = 47
    register_indices = [36, 38, 39, 41, 42, 44, 45] # From figures.py
    
    print(f"Analyzing Attention at Final Position {final_pos}")
    
    # Check Layers 4 and 5 (where merging occurs according to the paper)
    for layer in [4, 5]:
        attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, 0, final_pos, :] # [Batch, Head, Q, K]
        
        print(f"\n--- Layer {layer} ---")
        for reg_idx in register_indices:
            weight = attn_pattern[reg_idx].item()
            # We also need to know WHAT is in that register.
            # This uses the probe logic from the paper to guess the node
            # For now, we just print the weight to see if there is variance.
            if weight > 0.01: # Filter out unused registers
                print(f"Register {reg_idx} ({labels[reg_idx]}): Attn Weight = {weight:.4f}")

# %%
# --- Phase 2: Geometric Analysis (Activation Norms) ---
def analyze_register_norms(model, dataset):
    print("\n=== Phase 2: Geometric Analysis (Activation Norms) ===")
    # Generate example
    pred = generate_example(16, seed=42, path_length=12, order="backward")
    labels, cache = get_example_cache(pred, model, dataset)
    
    # Extract residual stream at Layer 4 (before final merge)
    resid = cache["blocks.4.hook_resid_mid"][0] # [Seq_Len, D_Model]
    
    register_indices = [36, 38, 39, 41, 42, 44, 45]
    
    norms = []
    vectors = []
    
    print("\n--- Register Norm Analysis ---")
    for reg_idx in register_indices:
        vec = resid[reg_idx]
        norm = torch.norm(vec).item()
        
        # Simple heuristic: Check if this register is 'active' (variance from mean)
        # In a real experiment, you'd match this to the actual subpath content
        norms.append((reg_idx, norm))
        vectors.append(vec)
        print(f"Register {reg_idx}: Norm = {norm:.4f}")

    # Check Cosine Similarity between active registers
    # If similarity is High (>0.9) but Norms are different, 
    # it supports "Order by Scale".
    # If similarity is Low, they are using orthogonal subspaces (Matrix Binding).
    if len(vectors) >= 2:
        sim = torch.nn.functional.cosine_similarity(vectors[0].unsqueeze(0), vectors[1].unsqueeze(0))
        print(f"Cosine Sim betw. first two registers: {sim.item():.4f}")

# %%
# --- Phase 3: Causal Intervention (Scaling Ablation) ---
def scale_register_hook(resid, hook, register_idx, scale_factor):
    # resid shape: [Batch, Seq, Hidden]
    resid[:, register_idx, :] *= scale_factor
    return resid

def test_scaling_hypothesis(model, dataset):
    print("\n=== Phase 3: Causal Intervention (Scaling Ablation) ===")
    # 1. Get a clean example
    pred = generate_example(16, 55, path_length=10, order="backward")
    if not is_model_correct(model, dataset, pred): 
        print("Model failed on initial example.")
        return

    print(f"Original Path: {pred.split(':')[-1]}")
    
    # 2. Define intervention target (e.g., Register 39)
    target_reg = 39 
    scale_factors = [0.1, 0.5, 1.5, 2.0, 5.0, -1.0]
    
    for scale in scale_factors:
        model.reset_hooks()
        
        # Hook into the residual stream before the merging layers (e.g., Layer 4 output)
        hook_name = "blocks.4.hook_resid_post"
        
        hook_fn = partial(scale_register_hook, register_idx=target_reg, scale_factor=scale)
        model.add_hook(hook_name, hook_fn)
        
        # Run generation
        # Note: You need a generation function that creates the full path string
        # We use a simplified version here based on 'eval_model' logic
        with torch.no_grad():
            # Tokenize and run
            tokens = dataset.tokenize(pred)
            input_tokens = torch.tensor(tokens).unsqueeze(0).to(DEVICE)[:, :-1]
            # Just checking the logits at the end to see immediate next token prediction change
            logits = model(input_tokens) 
            next_token_id = logits[0, -1].argmax()
            next_token = dataset.idx2tokens[next_token_id]
            
        print(f"Scale {scale}: Next predicted token = {next_token}")
        
        model.reset_hooks()

# %%
def analyze_positional_norms(model):
    # Indices from your experiment
    reg_indices = [36, 38, 39, 41, 42, 44, 45]
    
    # Extract learned Positional Embeddings
    # Shape: [Context_Len, D_Model]
    W_pos = model.W_pos.detach()
    
    print("\n--- Positional Embedding Norms ---")
    for idx in reg_indices:
        pos_vec = W_pos[idx]
        norm = torch.norm(pos_vec).item()
        print(f"Pos {idx}: Norm = {norm:.4f}")
        
    # Check if they are collinear (cosine sim)
    vecs = W_pos[reg_indices]
    # Compute similarity matrix or just adjacent pairs
    sim = torch.nn.functional.cosine_similarity(vecs[0].unsqueeze(0), vecs[1].unsqueeze(0))
    print(f"Cosine Sim (Pos {reg_indices[0]} vs {reg_indices[1]}): {sim.item():.4f}")

analyze_positional_norms(model)

# %%


# # 1. Run the Causal Scaling Experiment (Phase 3) first.
# test_scaling_hypothesis(model, dataset)

# # 2. Run the Attention Magnitude analysis (Phase 1)
# analyze_register_attention_magnitude(model, dataset)

# # 3. Compare against "Subspace" hypothesis (Phase 2)
# analyze_register_norms(model, dataset)
