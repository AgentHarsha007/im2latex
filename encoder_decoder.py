import torch
import torch.nn as nn
import math
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class RandomFourier2D(torch.nn.Module):
    def __init__(self, out_dim: int, sigma: float = 10.0, seed: int | None = None):
        super().__init__()
        assert out_dim % 2 == 0
        k = out_dim // 2
        g = torch.Generator()
        if seed is not None: g.manual_seed(seed)
        B = torch.randn(2, k, generator=g) * sigma
        self.register_buffer("B", B)
    def forward(self, pos):  # pos: (N,2) in [0,1] or pixel coords normalized
        proj = pos @ self.B                   # [N, k]
        return torch.cat([torch.sin(2*math.pi*proj),
                          torch.cos(2*math.pi*proj)], dim=-1)  # [N, out_dim]
class encoderblock(torch.nn.Module):
    def __init__(self,num_heads, embed_size, dropout,forward_expansion):
      super().__init__()
      self.attention=nn.MultiheadAttention(embed_size,num_heads=num_heads)
      self.feedforward=nn.Sequential(
          nn.Linear(embed_size,forward_expansion*embed_size),
          nn.ReLU(),
          nn.Linear(forward_expansion*embed_size,embed_size)
      )
      self.norm1=nn.LayerNorm(embed_size)
      self.norm2=nn.LayerNorm(embed_size)
      self.dropout1=nn.Dropout(dropout)
      self.dropout2=nn.Dropout(dropout)
    def forward(self,input,padding_mask=None):
      x=input.transpose(0,1)
      if padding_mask:
          attention_out,_=self.attention(x,x,x,key_padding_mask=padding_mask)
      else:
          attention_out,_=self.attention(x,x,x)
      add_norm=self.norm1(self.dropout1(attention_out)+x)
      x=self.feedforward(add_norm)
      x=self.norm2(add_norm+self.dropout2(x))
      x=x.transpose(0,1)
      return x
class decoderblock(torch.nn.Module):
  def __init__(self,num_heads,embed_size,dropout,forward_expasion):
    super().__init__()
    self.cross_attention=nn.MultiheadAttention(embed_size,num_heads=num_heads)
    self.attention=nn.MultiheadAttention(embed_size,num_heads=num_heads)
    self.dropout1=nn.Dropout(dropout)
    self.dropout2=nn.Dropout(dropout)
    self.dropout3=nn.Dropout(dropout)
    self.feed_forward=nn.Sequential(
        nn.Linear(embed_size,forward_expasion*embed_size),
        nn.ReLU(),
        nn.Linear(forward_expasion*embed_size,embed_size)
    )
    self.norm1=nn.LayerNorm(embed_size)
    self.norm2=nn.LayerNorm(embed_size)
    self.norm3=nn.LayerNorm(embed_size)
  def forward(self,encoder_input,decoder_input,padding_mask=None,attn_mask=None):
    x=decoder_input.transpose(0,1)
    attention_out,_=self.attention(x,x,x,attn_mask=attn_mask,key_padding_mask=padding_mask)
    add_norm=self.norm1(self.dropout1(attention_out)+x)
    cross_attention_out,_=self.cross_attention(add_norm,encoder_input.transpose(0,1),encoder_input.transpose(0,1))
    add_norm=self.norm2(self.dropout2(cross_attention_out)+add_norm)
    feed_out=self.feed_forward(add_norm)
    x=self.norm3(self.dropout3(feed_out)+add_norm)
    x=x.transpose(0,1)
    return x