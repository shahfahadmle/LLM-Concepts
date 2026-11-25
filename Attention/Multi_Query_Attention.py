import torch
import torch.nn as nn
from torch.nn import attention
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.0):
        super().__init__()
        assert d_model%num_heads==0
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model // num_heads
        self.q_proj=nn.Linear(d_model,d_model)
        self.k_proj=nn.Linear(d_model,self.head_dim)
        self.v_proj=nn.Linear(d_model,self.head_dim)
        self.dropout=dropout
    def forward(self,x:torch.Tensor):
        B,T,_=x.size()
        q=self.q_proj(x)
        q=q.view(B,T,self.num_heads,self.head_dim)
        q=q.transpose(1,2)

        k=self.k_proj(x)
        k=k.view(B,T,1,self.head_dim)
        k=k.transpose(1,2)

        v=self.v_proj(x)
        v=v.view(B,T,1,self.head_dim)
        v=v.transpose(1,2)

        attention=F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.dropout else 0.0,is_causal=True,enable_gqa=True)
        output=attention.transpose(1,2).contiguous()
        output=output.view(B,T,self.d_model)
        return output

    
model = MultiQueryAttention(
    d_model=512,
    num_heads=8,
    dropout=0.1
)

x = torch.randn(2, 10, 512)

output = model(x)

print("Input:", x.shape)
print("Output:", output.shape)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)