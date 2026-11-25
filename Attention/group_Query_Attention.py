import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if d_model <= 0 or num_query_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("d_model and head counts must be positive.")
        if d_model % num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads.")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1.")

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.bias = bias
        self.head_dim = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads

        self.query_proj = nn.Linear(
            d_model, num_query_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.output_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.query_proj(x)
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        q = q.transpose(1, 2)

        k = self.k_proj(x)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)

        v = self.v_proj(x)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(1, 2)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, seq_len, self.d_model)
        return self.output_proj(attention)


if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    d_model = 512
    model = GroupQueryAttention(
        d_model=d_model,
        num_query_heads=8,
        num_kv_heads=2,
        dropout=0.1,
    )
    x = torch.randn(batch_size, seq_len, d_model)
    output = model(x)
    print(output.shape)
    print(f'Total Trainable Parameters is : {sum(p.numel() for p in model.parameters())}')
