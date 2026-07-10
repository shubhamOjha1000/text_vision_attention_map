from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os
import torch

def load_hf_model(model_path: str, device: str) :

    # Import here to avoid circular dependency
    from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig

    # Load the tokenizer
    path = '/kaggle/input/hf-google-paligemma-3b-pt-224'
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="right")
    assert tokenizer.padding_side == "right"
    print("Loaded Tokenizer")

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    print("Loaded config.json")

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)
    print("Loaded state dict of the model")


    # Tie weights
    model.tie_weights()

    return (model, tokenizer)



def get_nth_smallest(r: torch.Tensor, N: int) -> torch.Tensor:
    """
    Returns the N-th smallest value along the last dimension for each batch.
    
    Args:
        r: Tensor of shape [B, L_v]
        N: Which smallest value to return (1-indexed for intuitive use)
           N=1 returns the smallest (minimum), N=2 returns second smallest, etc.
    
    Returns:
        Tensor of shape [B, 1] containing the N-th smallest value for each batch
    """
    # Validate input
   
    if N < 1:
        #raise ValueError(f"N must be >= 1, got {N}")
        return 0
    if N > r.size(-1):
        raise ValueError(f"N={N} cannot exceed sequence length L_v={r.size(-1)}")
    
    # Sort along the last dimension (L_v)
    sorted_r, _ = torch.sort(r, dim=-1)  # shape: [B, L_v]
     
    # Get the N-th smallest (0-indexed, so N-1)
    nth_smallest = sorted_r[:, N-1]  # shape: [B]
    
    # Add a dimension to make it [B, 1]
    return nth_smallest.unsqueeze(-1)

def safe_softmax_scaling(values: torch.Tensor, temperature: float = 1.0, dim: int = -1) -> torch.Tensor:
    """
    Safe softmax with standardization and temperature scaling for multi-dimensional tensors.
    
    Args:
        values: Input tensor of any shape
        temperature: Controls sharpness (lower = sharper, higher = smoother)
        dim: Dimension along which to compute softmax
    
    Returns:
        Stable softmax probabilities along specified dimension
    """
    # Store original shape for broadcasting
    orig_shape = values.shape
    
    # Standardize along the softmax dimension
    # Keep dimensions for proper broadcasting
    mean = values.mean(dim=dim, keepdim=True)
    std = values.std(dim=dim, keepdim=True)
    
    # Avoid division by zero (add small epsilon)
    std = std.clamp(min=1e-8)
    
    # Standardize
    standardized = (values - mean) / std
    
    # Apply temperature
    scaled = standardized / temperature
    return scaled












