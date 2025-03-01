import torch
import triton
import triton.language as tl

@triton.jit
def _rope_embeddings_kernel(
    Q, Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seq_len,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BLOCKSIZE: tl.constexpr
):
    ROPE_GROUP_SIZE = 4
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    
    offset = tl.arange(0, BLOCKSIZE)
    half_head_dim = head_dim // 2
    mask = offset < half_head_dim

    sin1 = tl.load(sin + (row_position % seq_len) * sin_row_stride + offset, mask=mask)
    cos1 = tl.load(cos + (row_position % seq_len) * cos_row_stride + offset, mask=mask)

    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = tl.min(head_start + ROPE_GROUP_SIZE, n_heads)
    
    # Loop over the selected heads.
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + offset
        offs_q2 = row_position * Q_row_stride + k * head_dim + offset + half_head_dim
        
        Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)

def rope_embeddings_triton(Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, BLOCKSIZE: int = 128):
    """
    Applies RoPE (Rotary Positional Embeddings) to the query tensor Q using a Triton kernel.
    
    This function assumes that:
      - Q is a CUDA tensor of shape (seq_len, n_heads, head_dim).
      - cos and sin are CUDA tensors of shape (seq_len, head_dim//2).
    The kernel performs an in-place update of Q.
    
    Args:
        Q (torch.Tensor): Query tensor on CUDA with shape (seq_len, n_heads, head_dim)
        cos (torch.Tensor): Cosine embeddings with shape (seq_len, head_dim//2)
        sin (torch.Tensor): Sine embeddings with shape (seq_len, head_dim//2)
        BLOCKSIZE (int): The number of elements processed per block (default: 128)
    
    Returns:
        torch.Tensor: The modified Q tensor with RoPE applied.
    """
    if not (Q.is_cuda and cos.is_cuda and sin.is_cuda):
        raise ValueError("All input tensors must be on CUDA.")

    seq_len, n_heads, head_dim = Q.shape

    Q_row_stride = Q.stride(0)
    cos_row_stride = cos.stride(0)
    sin_row_stride = sin.stride(0)
    
    ROPE_GROUP_SIZE = 4
    grid = (seq_len, (n_heads + ROPE_GROUP_SIZE - 1) // ROPE_GROUP_SIZE)
    
    _rope_embeddings_kernel[grid](
        Q, Q_row_stride,
        cos, cos_row_stride,
        sin, sin_row_stride,
        seq_len,
        head_dim,
        n_heads,
        BLOCKSIZE,
    )
    return Q
