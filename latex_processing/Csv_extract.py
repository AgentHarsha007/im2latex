import math
from collections import Counter, defaultdict
import pandas as pd
import torch
import torch.nn as nn
from pylatexenc.latexwalker import LatexWalker, LatexMacroNode, LatexGroupNode, LatexCharsNode
import os
import pandas as pd
def parse_latex_tree(latex_str: str):
    walker = LatexWalker(latex_str)
    nodes, pos, len_ = walker.get_latex_nodes()
    return nodes  # list of nodes

def tree_to_tokens(nodes, depth=0, sibling_idx=0):
    tokens, depths, siblings = [], [], []

    for idx, node in enumerate(nodes):
        if isinstance(node, LatexMacroNode):
            tokens.append(f"[{node.macroname.upper()}]")
            depths.append(depth)
            siblings.append(sibling_idx + idx)

            # Iterate macro arguments if available
            if getattr(node, "nodeargd", None) is not None and getattr(node.nodeargd, "argnlist", None) is not None:
                for arg_idx, arg in enumerate(node.nodeargd.argnlist):
                    if arg is not None and hasattr(arg, "nodelist"):
                        tokens.append("[ARG]")
                        depths.append(depth + 1)
                        siblings.append(arg_idx)

                        # Recurse into the argument's nodelist
                        child_tokens, child_depths, child_siblings = tree_to_tokens(
                            arg.nodelist, depth + 2, arg_idx
                        )
                        tokens.extend(child_tokens)
                        depths.extend(child_depths)
                        siblings.extend(child_siblings)

                        tokens.append("[/ARG]")
                        depths.append(depth + 1)
                        siblings.append(arg_idx)

            tokens.append(f"[/{node.macroname.upper()}]")
            depths.append(depth)
            siblings.append(sibling_idx + idx)

        elif isinstance(node, LatexGroupNode):
            tokens.append("[GROUP]")
            depths.append(depth)
            siblings.append(sibling_idx + idx)

            child_tokens, child_depths, child_siblings = tree_to_tokens(node.nodelist, depth + 1, 0)
            tokens.extend(child_tokens)
            depths.extend(child_depths)
            siblings.extend(child_siblings)

            tokens.append("[/GROUP]")
            depths.append(depth)
            siblings.append(sibling_idx + idx)

        elif isinstance(node, LatexCharsNode):
            # Emit each character literally
            for ch in node.chars:
                tokens.append(ch)
                depths.append(depth)
                siblings.append(sibling_idx + idx)

        else:
            # For other node types (environments, specials), you can extend handling here if needed.
            # If ignored, they won't contribute to the vocab.
            pass

    return tokens, depths, siblings
def build_vocab_from_formulas(train_df: pd.DataFrame, formula_col: str = "formula", image_col: str = "image_path",
                              add_specials=True,
                              min_freq: int = 1,
                              limit: int | None = None):
    """
    Parse and linearize all formulas from train_df[formula_col],
    collect tokens, and return a vocab list and frequency table.

    - add_specials: if True, prepend [PAD], [UNK]
    - min_freq: keep only tokens appearing at least min_freq times
    - limit: process only the first 'limit' rows for a quick dry run
    """
    token_counter = Counter()
    max_depth_seen = 0
    max_sibling_seen = 0
    rows=[]
    iterable = train_df
    if limit is not None:
        iterable = iterable.iloc[:limit]

    for _, row in iterable.iterrows():
        s = row[formula_col]
        img_path = row[image_col]
        if not isinstance(s, str):
            continue
        try:
            nodes = parse_latex_tree(s)
            tokens, depths, siblings = tree_to_tokens(nodes)
            token_counter.update(tokens)
            if depths:
                max_depth_seen = max(max_depth_seen, max(depths))
            if siblings:
                max_sibling_seen = max(max_sibling_seen, max(siblings))
            rows.append({
                "image":img_path,
                "formula": s,
                "tokens": tokens,
                "depths": depths,
                "siblings": siblings,
            })
        except Exception as e:
            # Optionally log or count parse failures
            # print(f"Parse error for formula: {s[:80]}... -> {e}")
            continue

    # Filter by frequency
    items = [(tok, freq) for tok, freq in token_counter.items() if freq >= min_freq]

    # Sort: by frequency (desc), then lexicographically
    items.sort(key=lambda x: (-x[1], x[0]))

    vocab = []
    if add_specials:
        vocab.extend(["[START]", "[PAD]", "[UNK]"])

    vocab.extend([tok for tok, _ in items])

    stats = {
        "num_tokens": len(vocab),
        "num_unique_observed": len(token_counter),
        "unknown_rate_expected": 0.0,  # for training set after construction it's near zero
        "max_depth_seen": max_depth_seen,
        "max_sibling_seen": max_sibling_seen,
        "most_common": token_counter.most_common(30),
    }
    df = pd.DataFrame(rows)
    return vocab, token_counter, stats ,df

#-extract and save vocab and node data
train_df = pd.read_csv("/content/im2latex100k/im2latex_train.csv")
# Add full image path
train_df['image_path'] = train_df['image'].apply(
    lambda x: os.path.join("/content/im2latex100k/formula_images_processed/formula_images_processed", x)
)
# Extract features for all images in batches
train_df['image_path']=train_df['image_path'].tolist()
vocab_list, token_freqs, stats ,df= build_vocab_from_formulas(train_df, "formula", add_specials=True, min_freq=1)
with open("vocab.txt", "w", encoding="utf-8") as f:
    for token in vocab_list:
        f.write(token + "\n")
df.to_csv("Latex_node_data.csv")
print(f"Vocab size: {len(vocab_list)}")
print("First vocab entries:", vocab_list)
print("Max depth seen:", stats["max_depth_seen"], "Max sibling seen:", stats["max_sibling_seen"])
print("Top tokens:", stats["most_common"][:10])