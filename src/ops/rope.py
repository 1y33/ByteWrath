import triton
import triton.language as tl

def rope_embeddings(
    Q, Q_row_stride,
    cos,cos_row_stride,
    sin,sin_row_stride,
    seq_len,
    head_dim  : tl.constexpr,
    n_heads   : tl.constexpr,
    BLOCKSIZE : tl.constexpr
):
    ROPE_GROUP_SIZE = 4
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    
    offset = tl.arange(0,BLOCKSIZE)
    half_head_dim = head_dim//2
    mask = offset< half_head_dim
    
    sin1 = tl.load(sin +(row_position % seq_len)*sin_row_stride + half_head_dim*0 + offset,mask=mask)
    cos1 = tl.load(cos +(row_position % seq_len)*cos_row_stride + half_head_dim*0 + offset,mask=mask)
    
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)
    
    for k in range(head_start,head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + offset
        offs_q2 = row_position * Q_row_stride + k * head_dim + offset + half_head_dim
        
        Q1 = tl.load(Q + offs_q1, mask = mask, other = 0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask = mask, other = 0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1*cos1 - Q2*sin1, mask = mask)
        tl.store(Q + offs_q2, Q2*cos1 + Q1*sin1, mask = mask)