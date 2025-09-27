import torch
import torch.nn as nn
from encoder_decoder import encoderblock,decoderblock,SinusoidalPositionalEncoding,RandomFourier2D
from latex_processing.tree_embedder import TreeEmbedder
from image_processing.image_feature_extracter import SwinDetrClipFeatureEncoder
import math 
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_list = ["<s>", "</s>", "<unk>", "[PAD]"]
#just create a vocab list for all the values
class ImageToLatexTransformer(nn.Module):
    def __init__(self,num_blocks=6,num_heads=8,vocabulary=None, vocab_size=None, embed_size=256,cnn_input_size=256, dropout=0.1,forward_expansion=4):
        super(ImageToLatexTransformer, self).__init__()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pad_token_id=vocab_list.index("[PAD]")
        self.project_cnn=nn.Linear(cnn_input_size,embed_size)
        # self.positional_encoding_fourier=RandomFourier2D(embed_size)
        # self.positional_encoding_encoder=SinusoidalPositionalEncoding(embed_size)
        self.positional_encoding_decoder=SinusoidalPositionalEncoding(embed_size)
        self.image_encoder=SwinDetrClipFeatureEncoder().to(device)
        self.image_encoder.train()
        self.TreeEmbedder = TreeEmbedder(vocabulary,embed_size, max_depth=20, max_sibling=225)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=self.pad_token_id)
        self.encoder=nn.ModuleList([encoderblock(num_heads,embed_size,dropout,forward_expansion) for _ in range(num_blocks)])
        self.decoder=nn.ModuleList([decoderblock(num_heads,embed_size,dropout,forward_expansion) for _ in range(num_blocks)])
        self.final_layer=nn.Linear(embed_size,vocab_size)
        self.value_head = nn.Linear(embed_size, 1)
    def forward(self,input,return_hidden=False):
        ((cnn_features,cnn_masks),(input_seq,input_masks))=input
        cnn_features=self.project_cnn(cnn_features)
        # cnn_features=self.positional_encoding_encoder(cnn_features)
        for layer in self.encoder:
            cnn_features=layer(cnn_features,cnn_masks)
        #process text_features
        B,T=input_seq.shape
        attention_mask=torch.triu(torch.ones(T,T, device=self.device) * float('-inf'), diagonal=1)
        # input_seq=self.embedding(input_seq)
        input_seq=self.positional_encoding_decoder(input_seq)
        for layer in self.decoder:
            input_seq=layer(cnn_features,input_seq,input_masks,attention_mask)
        logits=self.final_layer(input_seq)
        values = self.value_head(input_seq).squeeze(-1)
        if return_hidden:
          return logits, values
        return logits