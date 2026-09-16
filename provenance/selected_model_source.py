"""Candidate architecture: masked multiscale motifs and optional strand symmetry.

The symmetric variant averages representations of the two orientations before
classification. Supply the mean of the 30 descriptors computed for each strand.
It is exactly reverse-complement invariant in evaluation mode. This property
reflects dsRNA strand ambiguity; it is not a claim about biological potency.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model.seq_transformer import SinusoidalPositionalEncoding

def reverse_complement(ids,mask):
 length=mask.sum(1);position=torch.arange(ids.shape[1],device=ids.device)[None,:]
 index=(length[:,None]-1-position).clamp(min=0)
 complement=torch.tensor([0,4,3,2,1,5],device=ids.device)
 return complement[ids.gather(1,index)]*mask.to(ids.dtype)

class MaskedMultiScaleTransformer(nn.Module):
 def __init__(self,symmetric=False,dim=96,depth=3,heads=4,dropout=.15):
  super().__init__();self.symmetric=symmetric
  self.embedding=nn.Embedding(6,dim,padding_idx=0)
  self.local5=nn.Conv1d(dim,dim//2,5,padding=2);self.local9=nn.Conv1d(dim,dim//2,9,padding=4)
  self.mix=nn.Conv1d(dim,dim,1);self.dropout=nn.Dropout(dropout)
  self.position=SinusoidalPositionalEncoding(dim,641)
  layer=nn.TransformerEncoderLayer(dim,heads,dim_feedforward=3*dim,dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
  self.encoder=nn.TransformerEncoder(layer,depth);self.norm=nn.LayerNorm(dim)
  self.attention=nn.Linear(dim,1)
  self.head=nn.Sequential(nn.Linear(2*dim+30,dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim,1))
 def encode(self,ids,mask):
  m=mask[:,None,:];x=self.embedding(ids).transpose(1,2)*m
  x=torch.cat([F.gelu(self.local5(x)),F.gelu(self.local9(x))],dim=1)*m
  x=self.dropout(F.gelu(self.mix(x)))*m
  x=x.masked_fill(~m,float('-inf'));x=F.max_pool1d(x,4,ceil_mode=True)
  pm=F.max_pool1d(mask[:,None,:].float(),4,ceil_mode=True).squeeze(1)>0
  x=x.masked_fill(~pm[:,None,:],0).transpose(1,2)
  x=self.norm(self.encoder(self.position(x),src_key_padding_mask=~pm))
  mean=(x*pm[:,:,None]).sum(1)/pm.sum(1,keepdim=True)
  w=self.attention(x).squeeze(-1).masked_fill(~pm,float('-inf')).softmax(1)
  weighted=(x*w[:,:,None]).sum(1)
  return torch.cat([mean,weighted],dim=1)
 def forward(self,ids,mask,feat):
  if self.symmetric:
   n=len(ids);rc=reverse_complement(ids,mask);z=self.encode(torch.cat([ids,rc]),torch.cat([mask,mask]));z=(z[:n]+z[n:])/2
  else:z=self.encode(ids,mask)
  return {'lethality_logit':self.head(torch.cat([z,feat],dim=1)),'embedding':z}
