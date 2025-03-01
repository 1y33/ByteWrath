import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_forward(
    Y, Y_stride,
    X, X_stride,
    W, W_stride,
    r, r_stride,
    n_col, eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    # Create an index array with BLOCK_SIZE elements.
    offset = tl.arange(0, BLOCK_SIZE)
    
    # Compute pointers for the current row.
    X_ptr = X + row * X_stride
    Y_ptr = Y + row * Y_stride
    r_ptr = r + row * r_stride

    # Only process valid indices (< n_col)
    mask = tl.arange(0, BLOCK_SIZE) < n_col

    # Load the row data.
    X_row = tl.load(X_ptr + offset, mask=mask).to(tl.float32)
    W_row = tl.load(W + offset, mask=mask).to(tl.float32)

    # Compute mean square for the row.
    mean_square = tl.sum(X_row * X_row, axis=0) / n_col
    # Compute inverse RMS (i.e. normalization factor).
    inv_rms = 1.0 / tl.sqrt(mean_square + eps)
    tl.store(r_ptr, inv_rms)

    # Normalize and scale by weight.
    normed = X_row * inv_rms
    normed = normed.to(W_row.dtype)
    out = normed * W_row
    tl.store(Y_ptr + offset, out, mask=mask)

class FastRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, eps: float):
        shape = X.shape
        dim = shape[-1]
        X_flat = X.contiguous().view(-1, dim)
        n_rows, n_cols = X_flat.shape

        # Ensure BLOCK_SIZE is a power-of-2:
        # If n_cols is at least 1024, use 1024; otherwise, choose the next power of 2.
        if n_cols >= 1024:
            BLOCK_SIZE = 1024
        else:
            BLOCK_SIZE = 1 << ((n_cols - 1).bit_length())

        Y_flat = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
        r = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        rms_norm_forward[(n_rows,)](
            Y_flat, Y_flat.stride(0),
            X_flat, X_flat.stride(0),
            W,      W.stride(0),
            r,      r.stride(0),
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.eps = eps
        ctx.save_for_backward(X_flat, W, r)
        return Y_flat.view(*shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("Backward pass is not implemented for FastRMSNorm.")
