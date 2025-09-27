# pip install transformers torchvision pillow accelerate huggingface_hub
# Ensure 'accelerate' and 'huggingface_hub' are installed for this solution.

import warnings
warnings.filterwarnings("ignore") # Suppress warnings for a clean output

import torch
import torch.nn as nn
from typing import List, Tuple, Union
from PIL import Image
from huggingface_hub import hf_hub_download

from transformers import (
    AutoImageProcessor, SwinModel, SwinConfig,
    DetrImageProcessor, DetrForObjectDetection, DetrConfig,
    CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPVisionConfig
)

# -----------------------------
# Device Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Model IDs
# -----------------------------
SWIN_ID = "microsoft/swin-base-patch4-window7-224"
DETR_ID = "facebook/detr-resnet-50"
CLIP_ID = "openai/clip-vit-base-patch32"

# -----------------------------
# Fast Image Processors
# -----------------------------
swin_proc = AutoImageProcessor.from_pretrained(SWIN_ID, use_fast=True)
detr_proc = DetrImageProcessor.from_pretrained(DETR_ID, use_fast=True)
clip_proc = CLIPImageProcessor.from_pretrained(CLIP_ID, use_fast=True)

# ----------------------------------------------------------------------
# THE DEFINITIVE FIX: Manually build and load models to bypass 'meta' issues
# This is the most robust method and guarantees correct weight loading.
# ----------------------------------------------------------------------
print("\nLoading pretrained backbones with manual, direct method...")

# --- Load Swin ---
swin_config = SwinConfig.from_pretrained(SWIN_ID)
swin = SwinModel(swin_config)  # 1. Create empty model structure on CPU
swin_weights_path = hf_hub_download(repo_id=SWIN_ID, filename="pytorch_model.bin")
swin.load_state_dict(torch.load(swin_weights_path, map_location="cpu")) # 2. Load weights directly
swin = swin.to(device).eval() # 3. Move fully loaded model to target device
print("Swin model loaded successfully.")

# --- Load DETR ---
detr_config = DetrConfig.from_pretrained(DETR_ID)
detr = DetrForObjectDetection(detr_config)
detr_weights_path = hf_hub_download(repo_id=DETR_ID, filename="pytorch_model.bin")
detr.load_state_dict(torch.load(detr_weights_path, map_location="cpu"))
detr = detr.to(device).eval()
print("DETR model loaded successfully.")

# --- Load CLIP Vision ---
clip_config = CLIPVisionConfig.from_pretrained(CLIP_ID)
clip_vision = CLIPVisionModelWithProjection(clip_config)
clip_weights_path = hf_hub_download(repo_id=CLIP_ID, filename="pytorch_model.bin")
clip_vision.load_state_dict(torch.load(clip_weights_path, map_location="cpu"))
clip_vision = clip_vision.to(device).eval()
print("CLIP Vision model loaded successfully.")
print("-" * 30)

# ----------------------------------------------------------------------

CLIP_DIM = clip_vision.config.projection_dim

def load_images(imgs: List[Union[str, Image.Image]]) -> List[Image.Image]:
    out = []
    for im in imgs:
        if isinstance(im, Image.Image):
            out.append(im.convert("RGB"))
        else:
            try:
                out.append(Image.open(im).convert("RGB"))
            except FileNotFoundError:
                print(f"Warning: Image not found at {im}. Skipping.")
    return out

class Box2DPositionalEncoding(nn.Module):
    def __init__(self, dim: int = 128, base: float = 10000.0):
        super().__init__()
        assert dim % 8 == 0, "dim must be divisible by 8"
        self.dim = dim
        self.base = base

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        B, Q, C4 = boxes.shape
        assert C4 == 4
        d_per_coord = self.dim // 4
        freqs = d_per_coord // 2
        idx = torch.arange(freqs, device=boxes.device, dtype=boxes.dtype)
        div = torch.pow(self.base, idx / freqs).view(1, 1, 1, freqs)
        ang = boxes.unsqueeze(-1) / div
        pe_coord = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(B, Q, 4, -1)
        return pe_coord.reshape(B, Q, self.dim)

class SwinDetrClipFeatureEncoder(nn.Module):
    def __init__(self,
                 proj_dim: int = 512,
                 box_pe_dim: int = 128,
                 det_conf_thresh: float = 0.30,
                 det_max_objs: int = 100,
                 include_clip_token: bool = True):
        super().__init__()
        self.include_clip = include_clip_token
        self.det_conf = det_conf_thresh
        self.det_max_objs = det_max_objs

        self.swin_proj = nn.Linear(swin.config.hidden_size, proj_dim, bias=False)
        self.detr_enc_proj = nn.Linear(detr.config.d_model, proj_dim, bias=False)
        self.detr_obj_proj = nn.Linear(detr.config.d_model, proj_dim, bias=False)
        if self.include_clip:
            self.clip_proj = nn.Linear(CLIP_DIM, proj_dim, bias=False)
        self.box_pe = Box2DPositionalEncoding(box_pe_dim)
        self.box_proj = nn.Linear(box_pe_dim, proj_dim, bias=False)
        self.type_embed = nn.Embedding(4, proj_dim)

    def forward(self, images: List[Union[str, Image.Image]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs = load_images(images)
        if not imgs: return torch.empty(0), torch.empty(0), torch.empty(0)
        B = len(imgs)

        # Use no_grad for backbone inference to save memory and compute,
        # as we only need gradients for the projection layers.
        with torch.no_grad():
            sv = swin_proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            swin_features = swin(sv, return_dict=True).last_hidden_state

            detr_inputs = detr_proc(images=imgs, return_tensors="pt")
            dv = detr_inputs["pixel_values"].to(device)
            dm = detr_inputs.get("pixel_mask", torch.ones(dv.shape[:-1], device=device)).to(device)
            d_out = detr(dv, pixel_mask=dm, output_hidden_states=True, return_dict=True)

        swin_tok = self.swin_proj(swin_features)
        pad_s = torch.zeros((B, swin_tok.size(1)), dtype=torch.bool, device=device)
        detr_enc = self.detr_enc_proj(d_out.encoder_last_hidden_state)
        pad_e = torch.zeros((B, detr_enc.size(1)), dtype=torch.bool, device=device)
        detr_obj = self.detr_obj_proj(d_out.last_hidden_state)
        boxes = d_out.pred_boxes
        scores = d_out.logits.softmax(-1)[..., :-1].max(-1).values
        keep = scores >= self.det_conf
        if self.det_max_objs and self.det_max_objs < keep.size(1):
            topk = torch.topk(scores, k=self.det_max_objs, dim=1)
            cap = torch.zeros_like(keep).scatter_(1, topk.indices, True)
            keep = keep & cap
        box_pe = self.box_proj(self.box_pe(boxes))
        detr_obj = (detr_obj + box_pe).masked_fill(~keep.unsqueeze(-1), 0.0)
        pad_o = ~keep[:, :detr_obj.size(1)]
        parts = [swin_tok, detr_enc, detr_obj]
        type_cols = [torch.full((t.size(1),), i, dtype=torch.long, device=device) for i, t in enumerate(parts)]
        pad_parts = [pad_s, pad_e, pad_o]
        if self.include_clip:
            with torch.no_grad():
                cpv = clip_proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
                clip_embeds = clip_vision(pixel_values=cpv, return_dict=True).image_embeds
            clip_tok = self.clip_proj(clip_embeds).unsqueeze(1)
            parts.append(clip_tok)
            type_cols.append(torch.full((1,), 3, dtype=torch.long, device=device))
            pad_parts.append(torch.zeros((B, 1), dtype=torch.bool, device=device))
        tokens = torch.cat(parts, dim=1)
        type_ids = torch.cat(type_cols, dim=0).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.type_embed(type_ids)
        pad_mask = torch.cat(pad_parts, dim=1)
        return tokens, pad_mask, type_ids

@torch.no_grad()
def extract_image_tokens_infer(images: List[Union[str, Image.Image]], encoder: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder.eval()
    return encoder(images)

if __name__ == "__main__":
    try:
        # Create dummy images for a self-contained, runnable example
        Image.new('RGB', (800, 600), color='green').save("example1.png")
        Image.new('RGB', (600, 800), color='red').save("example2.jpg")
        print("\nCreated dummy images 'example1.png' and 'example2.jpg'.")
        batch_paths = ["example1.png", "example2.jpg"]
        
        # Instantiate the main encoder and move it to the correct device
        encoder_instance = SwinDetrClipFeatureEncoder().to(device)
        
        # --- Run in Training Mode ---
        print("\nRunning in training mode...")
        encoder_instance.train()
        tokens, pad_mask, type_ids = encoder_instance(batch_paths)
        print(f"Train mode successful. Output shapes:")
        print(f"  Tokens:   {tokens.shape}")
        print(f"  Pad Mask: {pad_mask.shape}")
        print(f"  Type IDs: {type_ids.shape}")
        
        # --- Run in Inference Mode ---
        print("\nRunning in inference mode...")
        tokens_infer, _, _ = extract_image_tokens_infer(batch_paths, encoder=encoder_instance)
        print(f"Infer mode successful. Output shape:")
        print(f"  Tokens:   {tokens_infer.shape}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have run 'pip install transformers torchvision pillow accelerate huggingface_hub'.")
