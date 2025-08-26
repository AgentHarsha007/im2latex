import numpy as np
import random
import cv2
import torch
import torch.functional as F
import torch.nn as nn
import pytesseract
from pytesseract import Output
from char_cnn import CNNModel
import torchvision.transforms as T
import math
from image_processing.char_cnn import CNNModel
from image_processing.char_pix import extract_char_boxes
class Image_data_Extractor:
    def __init__(self, model:CNNModel, device):
        self.model = model
        self.device = model.device
        self.transform=T.Compose([
            T.ToPILImage(),
            T.Grayscale(num_output_channels=1),  # force grayscale
            T.Resize((24, 24)),                  # resize to match training
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))          # if you normalized during training
            ])
    def get_batch_indices(self,Data_size,BatchSize):
        return random.sample(range(Data_size),BatchSize)
    def extract_batch_features(self, image_paths, device):
        batch_centers, batch_features = [], []
        seq_lens = []
        # Extract features and centers per image
        for path in image_paths:
            centers, char_crops = extract_char_boxes(path, preprocess=True)
            crops_tensor = torch.stack([self.transform(crop) for crop in char_crops]).to(device)
            with torch.no_grad():
                _, features = self.model(crops_tensor, return_logits=True)
            batch_centers.append(torch.tensor(centers, dtype=torch.float32, device=device))
            batch_features.append(features)
            seq_lens.append(len(char_crops))
        # Determine max sequence length
        max_len = max(seq_lens)
        feature_dim = batch_features[0].size(1)
        # Pad centers and features
        for i in range(len(batch_centers)):
            pad_len = max_len - seq_lens[i]
            if pad_len > 0:
                batch_centers[i] = F.pad(batch_centers[i], (0, 0, 0, pad_len))
                batch_features[i] = F.pad(batch_features[i], (0, 0, 0, pad_len))
        # Stack into batched tensors
        batch_centers_tensor = torch.stack(batch_centers)       # [B, T_max, 2]
        batch_features_tensor = torch.stack(batch_features)     # [B, T_max, feature_dim]
        # Create mask: 1 = valid token, 0 = padding
        batch_masks = torch.zeros(len(batch_centers), max_len, dtype=torch.bool, device=device)
        for i, l in enumerate(seq_lens):
            batch_masks[i, :l] = 1
        return (batch_centers_tensor, batch_features_tensor), batch_masks
    def get_batch_data(self,train_df, padded, batch_size, device):
        indices = self.get_batch_indices(len(train_df), batch_size)
        # Get image paths and token sequences
        img_paths = train_df.iloc[indices]['image_path'].tolist()
        input_seq = [padded[i][:-1] for i in indices]   # [B, T]
        output_seq = [padded[i][1:] for i in indices]   # [B, T]
        # Extract CNN features for the images
        centers,cnn_features,batched_masks = self.extract_image_features(img_paths, device) # [B,T,128]
        # Convert sequences to torch tensors
        input_seq = np.array(input_seq, dtype=np.long) # Convert to numpy array first
        output_seq = np.array(output_seq, dtype=np.long) # Convert to numpy array first

        input_seq = torch.tensor(input_seq, dtype=torch.long, device=device)
        output_seq = torch.tensor(output_seq, dtype=torch.long, device=device)

        return (((centers, cnn_features), batched_masks), input_seq), output_seq