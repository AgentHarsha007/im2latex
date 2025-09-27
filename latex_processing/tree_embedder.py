# -------------------------------
# Full LaTeX → Tree → Embeddings Pipeline
# -------------------------------
import sys
sys.path.append("./../")
import torch
import torch.nn as nn
import math
# -------------------------------
# Embedding Module
# -------------------------------
class TreeEmbedding(nn.Module):
    def __init__(self, vocab_list, embed_dim=64, max_depth=20, max_sibling=225):
        super().__init__()
        self.token_to_idx = {tok: i for i, tok in enumerate(vocab_list)}
        self.token_embed = nn.Embedding(len(vocab_list), embed_dim)
        self.depth_embed = nn.Embedding(max_depth, embed_dim)
        self.sibling_embed = nn.Embedding(max_sibling, embed_dim)
        self.embed_dim = embed_dim