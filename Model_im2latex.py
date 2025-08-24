import torch
import torch.nn as nn
from encoder_decoder import encoderblock,decoderblock,SinusoidalPositionalEncoding
class ImageToLatexTransformer(nn.Module):
    def __init__(self,num_blocks,num_heads, vocab_size, pad_token_id, embed_size, dropout, forward_expansion):
        super(ImageToLatexTransformer, self).__init__()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pad_token_id=pad_token_id
        self.project_cnn=nn.Linear(768,embed_size)
        self.positional_encoding_encoder=SinusoidalPositionalEncoding(embed_size)
        self.positional_encoding_decoder=SinusoidalPositionalEncoding(embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_id)
        self.encoder=nn.ModuleList([encoderblock(num_heads,embed_size,dropout,forward_expansion) for _ in range(num_blocks)])
        self.decoder=nn.ModuleList([decoderblock(num_heads,embed_size,dropout,forward_expansion) for _ in range(num_blocks)])
        self.final_layer=nn.Linear(embed_size,vocab_size)
        self.value_head = nn.Linear(embed_size, 1)
    def forward(self,input,return_hidden=False):
        cnn_features,input_seq=input
        cnn_features=self.project_cnn(cnn_features)
        cnn_features=self.positional_encoding_encoder(cnn_features)
        for layer in self.encoder:
            cnn_features=layer(cnn_features)
        #process text_features
        B,T=input_seq.shape
        padding_mask= (input_seq == self.pad_token_id)
        attention_mask=torch.triu(torch.ones(T,T, device=self.device) * float('-inf'), diagonal=1)
        input_seq=self.embedding(input_seq)
        input_seq=self.positional_encoding_decoder(input_seq)
        for layer in self.decoder:
            input_seq=layer(cnn_features,input_seq,padding_mask,attention_mask)
        logits=self.final_layer(input_seq)
        values = self.value_head(input_seq).squeeze(-1)
        if return_hidden:
          return logits, values
        return logits