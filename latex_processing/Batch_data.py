import sys
sys.path.append("./../")
# from im2latex.encoder_decoder import SinusoidalPositionalEncoding
from tree_embedder import TreeEmbedding
import torch
import torch.nn as nn
import math
def get_batch_data(Train_df, tree_embed_module, batch_size=32, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """
    Prepares a batch of data for training.
    """
    image_paths = [Train_df.iloc[idx]["image_path"] for idx in torch.randint(0, len(Train_df), (batch_size,))]
    batch_indexes = torch.randint(0, len(Train_df), (batch_size,))
    batch_tokens = [['[START]'] + Train_df.iloc[idx]["tokens"].split() for idx in batch_indexes]
    max_len = max(len(tokens) for tokens in batch_tokens)
    batch_depths = [list(map(int, Train_df.iloc[idx]["depths"].split())) for idx in batch_indexes]
    batch_siblings = [list(map(int, Train_df.iloc[idx]["siblings"].split())) for idx in batch_indexes]
    padded_tokens = []
    padded_depths = []
    padded_siblings = []
    pad_mask_lists = []
    for tokens, depths, siblings in zip(batch_tokens, batch_depths, batch_siblings):
        pad_length = max_len - len(tokens)
        padded_tokens.append(tokens + ["[PAD]"] * pad_length)
        padded_depths.append(depths + [0] * pad_length)
        padded_siblings.append(siblings + [0] * pad_length)
        pad_mask_lists.append([False] * len(tokens) + [True] * pad_length)
    pad_mask = torch.tensor(pad_mask_lists, device=device)
    token_to_idx = tree_embed_module.token_to_idx
    token_indices = torch.tensor([[token_to_idx.get(tok, token_to_idx["[UNK]"]) for tok in tokens] for tokens in padded_tokens], device=device)
    depth_indices = torch.tensor(padded_depths, device=device)
    sibling_indices = torch.tensor(padded_siblings, device=device)
    token_embeds = tree_embed_module.token_embed(token_indices)
    depth_embeds = tree_embed_module.depth_embed(depth_indices)
    sibling_embeds = tree_embed_module.sibling_embed(sibling_indices)
    combined_embeds = token_embeds + depth_embeds + sibling_embeds
    return token_indices, combined_embeds, pad_mask,image_paths
def get_vocabulary(vocab_filepath):
    vocab=[]
    with open(vocab_filepath, "r", encoding="utf-8") as f:
        vocab.extend([line.strip() for line in f.readlines()])
    return vocab
vocab_list = get_vocabulary("./vocab.txt")
print(f"Vocab size: {len(vocab_list)}")
print(vocab_list[:10])
# -------------------------------