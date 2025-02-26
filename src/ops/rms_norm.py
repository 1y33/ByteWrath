import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_forwad(
    Y, Y_stride,
    X, X_stride,
    W, W_stride,
    r, r_stride,
    n_col, eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    X += row * X_stride
    Y += row * Y_stride
    r += row * r_stride
    mask = tl.arange(0, BLOCK_SIZE, 1) < n_col

    X_row = tl.load(X + offset, mask=mask).to(tl.float32)
    W_row = tl.load(W + offset, mask=mask).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis=0) / n_col
    inv_var = tl.sqrt(row_var + eps)
    tl.store(r, inv_var)

    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)
    out = normed * W_row
    tl.store(Y + offset, out, mask=mask)

class FastRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, eps: float):
        shape = X.shape
        dim = shape[-1]
        X_flat = X.contiguous().view(-1, dim)
        n_rows, n_cols = X_flat.shape

        BLOCK_SIZE = 1024 if n_cols >= 1024 else n_cols

        Y_flat = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
        r = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        rms_norm_forwad[(n_rows,)](
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

