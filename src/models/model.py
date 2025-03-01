from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from src.ops import rms_norm

@dataclass
class ModelArgs:
    dim_embed: int = 2048
    n_layers: int = 16
    n_heads: int = 24
    n_kv_heads: Optional[int] = None
    vocab_size: int = 55200
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 4096
    device: str = None
    
    use_triton: bool = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_triton: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor):
        if self.use_triton:
            return rms_norm.FastRMSNorm.apply(x, self.weight, self.eps)
        else:
            x_float = x.float()

            norm = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)

            return self.weight * norm.type_as(x)

def relu_activation(x:torch.Tensor)->torch.Tensor:
    return F.relu(x) ** 2

class Expert(nn.Module):
    def __init__(self,dim:int,hidden_dim:int):
        super().__init__()
        
        self.w1 = nn.Linear(dim,hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim,dim,bias=False)
        self.w3 = nn.Linear(dim,hidden_dim,bias=False)
        
    def forward(self,x:torch.Tensor)->torch.Tensor :
        return self.w2(relu_activation(self.w1(x)) * self.w3(x))
    
    
class FFN(nn.Module):
    def __init__(self,num_experts,dim,hidden_dim):
        super().__init__()
        
        self.experts = nn.ModuleList([Expert(dim,hidden_dim) for _ in range(num_experts)])
        
        self.gate = nn.Linear(dim,num_experts)
        
        self.num_experts = num_experts
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        
        batch_size, seq_len, dim = x.size()
        x_flat = x.view(-1,dim)
        
        # Get router probabilities
        gate_logits = self.gate(x_flat)
        router_probs = F.softmax(gate_logits, dim=-1)
        
        top2_probs, top2_indices = torch.topk(router_probs, k=2, dim=-1)
        
        top2_probs = top2_probs / top2_probs.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            mask = (top2_indices == i).any(dim=-1)
            if not mask.any():
                continue
                
            expert_inputs = x_flat[mask]
            
            expert_outputs = expert(expert_inputs)
            
            position_in_top2 = (top2_indices[mask] == i).nonzero(as_tuple=True)[1]
            expert_probs = top2_probs[mask, position_in_top2].unsqueeze(-1)
            
            final_output[mask] += expert_outputs * expert_probs
            
        return final_output.view(batch_size, seq_len, dim)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        # Projection matrices
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        
        # RoPE parameters - register as buffer so it moves to correct device
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cache = None
        
    def apply_rope(self, q, k):
        seq_len = q.size(2)
        t = torch.arange(seq_len, device=q.device, dtype=q.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb.unsqueeze(0).unsqueeze(0)
        
        q = (q * torch.cos(emb)) + (self.rotate_half(q) * torch.sin(emb))
        k = (k * torch.cos(emb)) + (self.rotate_half(k) * torch.sin(emb))
        return q, k  # Fix the missing 'k' here
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
                         
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        
        # Project to queries, keys, and values
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to q and k
        q, k = self.apply_rope(q, k)
        
        # Cache KV
        if self.cache is not None:
            k = torch.cat([self.cache['k'], k], dim=2)
            v = torch.cat([self.cache['v'], v], dim=2)
        self.cache = {'k': k, 'v': v}
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax and compute weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project to output dimension
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.wo(context)
        
        return output

# MoE Transformer Block with FlexAttention
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int,seq_len:int, head_dim: int, num_experts: int = 4, ffn_dim_mult: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        
        self.attn = MultiHeadAttention(dim, n_heads, head_dim, max_seq_len=seq_len)
        
        self.norm2 = RMSNorm(dim)
        
        hidden_dim = int(dim * ffn_dim_mult)
        
        self.ffn = FFN(num_experts=num_experts, dim=dim, hidden_dim=hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), mask)
        
        x = x + self.ffn(self.norm2(x))
        
        return x

class SimpleModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        self.token_embed = nn.Embedding(args.vocab_size, args.dim_embed)
        self.pos_embed = nn.Embedding(args.max_seq_len, args.dim_embed)
        
        # Calculate head dimensions
        head_dim = args.dim_embed // args.n_heads
        
        # Create layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=args.dim_embed, 
                seq_len=args.max_seq_len,
                n_heads=args.n_heads,
                head_dim=head_dim,
                num_experts=4,  # 4 experts in MoE
                ffn_dim_mult=4
            ) for i in range(args.n_layers)
        ])
        
        self.norm = RMSNorm(args.dim_embed, use_triton=args.use_triton)
        self.output = nn.Linear(args.dim_embed, args.vocab_size, bias=False)
        
    def forward(self, tokens: torch.Tensor):
        bsz, seq_len = tokens.shape
        
        # Token embedding
        h = self.token_embed(tokens)
        
        # Add positional embeddings
        positions = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0)
        h = h + self.pos_embed(positions)
        
        # Create attention mask (causal)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h.device) * float('-inf'), diagonal=1)
        
        # Process through layers
        for layer in self.layers:
            h = layer(h, mask)
        
        # Final normalization and output projection
        h = self.norm(h)
        logits = self.output(h)
        
        return logits
# Function to test if the model works
def test_model_functionality(use_triton=True):
    print("\n==== Testing Model Functionality ====")
    tiny_args = ModelArgs

    model = SimpleModel(tiny_args)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # tokens = torch.randint(0, tiny_args.vocab_size, (1, tiny_args.max_seq_len), device=device)
    # with torch.no_grad():
    #     logits = model(tokens)
    
    # expected_shape = (1, tiny_args.max_seq_len, tiny_args.vocab_size)
    # assert logits.shape == expected_shape, f"Output shape mismatch: {logits.shape} vs {expected_shape}"
    
    # print(f"Model test passed! Output shape: {logits.shape}")
    # print(f"Device: {device}")
    return model


if __name__ == "__main__":
    test_model = test_model_functionality(use_triton=True)
