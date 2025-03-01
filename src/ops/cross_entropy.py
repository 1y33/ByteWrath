import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def simple_cross_entropy_forward_kernel(
    logits_ptr,      # pointer to logits tensor (flattened 2D: [batch, vocab])
    labels_ptr,      # pointer to labels (one int per row)
    losses_ptr,      # pointer to output losses (one float per row)
    logits_row_stride: tl.constexpr,  # row stride for logits tensor
    VOCAB_SIZE: tl.constexpr,         # vocabulary size (constexpr)
):
    row = tl.program_id(0)
    
    # Pointer to the start of this row of logits.
    row_logits_ptr = logits_ptr + row * logits_row_stride
    
    # Create an index vector for all positions within the vocabulary dimension.
    offsets = tl.arange(0, VOCAB_SIZE)
    
    # Load the logits for this row.
    logits = tl.load(row_logits_ptr + offsets)
    
    # Numerically stable log-sum-exp: subtract the maximum logit.
    max_logit = tl.max(logits, axis=0)
    exp_sum = tl.sum(tl.exp(logits - max_logit), axis=0)
    lse = max_logit + tl.log(exp_sum)
    
    # Load the correct label for this row.
    label = tl.load(labels_ptr + row)
    # Load the logit corresponding to the correct label.
    correct_logit = tl.load(row_logits_ptr + label)
    
    # Compute the cross entropy loss: logsumexp - correct_logit.
    loss = lse - correct_logit
    tl.store(losses_ptr + row, loss)

def simple_cross_entropy_forward(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Forward pass using the Triton kernel.
    
    Args:
        logits: Tensor of shape (batch, vocab) with unnormalized log-probabilities.
        labels: Tensor of shape (batch,) with the correct class indices.
    
    Returns:
        losses: Tensor of shape (batch,) containing the per-sample loss.
    """
    assert logits.ndim == 2, "Logits must be a 2D tensor."
    batch, vocab = logits.shape
    
    losses = torch.empty((batch,), dtype=torch.float32, device=logits.device)
    
    grid = (batch,)
    simple_cross_entropy_forward_kernel[grid](
        logits,               # pointer to logits
        labels,               # pointer to labels
        losses,               # pointer to losses
        logits.stride(0),     # row stride of logits
        vocab,                # VOCAB_SIZE (constexpr)
    )
    
    return losses

class TritonCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        """
        Forward pass using the Triton kernel.
        
        Args:
            logits: Tensor of shape (batch, vocab).
            labels: Tensor of shape (batch,).
        
        Returns:
            loss: Tensor of shape (batch,) with the per-sample loss.
        """
        # Compute loss using the Triton kernel.
        loss = simple_cross_entropy_forward(logits, labels)
        # Save tensors for backward.
        ctx.save_for_backward(logits, labels)
        # For backward, compute the log-sum-exp in Python.
        ctx.lse = torch.logsumexp(logits, dim=1)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients with respect to logits.
        
        Args:
            grad_output: Gradient of the loss output, shape (batch,).
        
        Returns:
            grad_logits: Gradient w.r.t. logits, shape (batch, vocab).
            None: No gradient for labels.
        """
        logits, labels = ctx.saved_tensors
        lse = ctx.lse  # shape: (batch,)
        
        # Compute softmax probabilities: exp(x - lse)
        softmax = torch.exp(logits - lse.unsqueeze(1))
        
        # For each sample subtract 1 at the correct label index.
        grad_logits = softmax.clone()
        grad_logits[torch.arange(logits.size(0)), labels] -= 1.0
        
        # Multiply by grad_output (broadcasting from (batch,) to (batch, vocab))
        grad_logits = grad_logits * grad_output.unsqueeze(1)
        return grad_logits, None

class TritonCrossEntropyLoss(nn.Module):
    def __init__(self):
        """
        A simple nn.Module wrapper for the Triton-based cross entropy loss.
        """
        super().__init__()

    def forward(self, logits, labels):
        """
        Computes per-sample cross entropy loss.
        
        Args:
            logits: Tensor of shape (batch, vocab) with unnormalized scores.
            labels: Tensor of shape (batch,) with correct class indices.
        
        Returns:
            loss: Tensor of shape (batch,) containing the per-sample loss.
        """
        return TritonCrossEntropyFunction.apply(logits, labels)

# Example usage:
if __name__ == "__main__":
    batch_size = 4
    vocab_size = 8
    torch.manual_seed(0)
    
    # Create random logits and sample labels.
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, vocab_size, (batch_size,), device="cuda", dtype=torch.int64)
    
    loss_module = TritonCrossEntropyLoss()
    loss = loss_module(logits, labels)
    
    print("Logits:\n", logits)
    print("Labels:\n", labels)
    print("Per-sample loss:\n", loss)
