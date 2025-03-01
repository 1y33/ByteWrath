import torch
import time
import json
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from models import model


class LLaMA:
    def __init__(self, model, tokenizer, model_args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir, tokenizer_path, load_model, max_seq_len, max_batch_size, device):
        # Load checkpoint if required
        checkpoint = None
        if load_model:
            ckpt_files = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert ckpt_files, f"No checkpoint files found in {checkpoints_dir}"
            ckpt_path = ckpt_files[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        # Load model parameters
        params = json.loads(Path(checkpoints_dir, "params.json").read_text())
        model_args = model.ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, device=device, **params)

        # Setup tokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # Set default tensor type based on device
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        # Build the model and load weights if needed
        model = model.Transformer(model_args).to(device)
        if load_model:
            checkpoint.pop('rope.freqs', None)
            model.load_state_dict(checkpoint, strict=True)
            print("Model state loaded.")
        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts, temperature=0.6, top_p=0.9, max_gen_len=None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        # Tokenize prompts and prepare padded tensor
        prompt_tokens = [self.tokenizer.encode(p, out_type=int, add_bos=True) for p in prompts]
        batch_size = len(prompt_tokens)
        max_prompt_len = max(len(pt) for pt in prompt_tokens)
        assert batch_size <= self.args.max_batch_size, "Batch size exceeds maximum"
        assert max_prompt_len <= self.args.max_seq_len, "Prompt length exceeds maximum"

        total_len = min(self.args.max_seq_len, max_prompt_len + max_gen_len)
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device)
        for i, pt in enumerate(prompt_tokens):
            tokens[i, :len(pt)] = torch.tensor(pt, device=self.args.device)

        # Generate tokens one position at a time
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=self.args.device)
        prompt_mask = tokens != pad_id
        for cur_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.flatten()
            # If token already set (prompt token), keep it; else, update with generated token
            tokens[:, cur_pos] = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            eos_reached |= (~prompt_mask[:, cur_pos]) & (tokens[:, cur_pos] == self.tokenizer.eos_id)
            if eos_reached.all():
                break

        # Decode tokens and trim at EOS if present
        out_tokens, out_text = [], []
        for tok in tokens.tolist():
            if self.tokenizer.eos_id in tok:
                tok = tok[:tok.index(self.tokenizer.eos_id)]
            out_tokens.append(tok)
            out_text.append(self.tokenizer.decode(tok))
        return out_tokens, out_text

    def _sample_top_p(self, probs, p):
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs - sorted_probs > p
        sorted_probs[mask] = 0.0
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(sorted_probs, 1)
        return torch.gather(sorted_indices, -1, next_token)

if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
    ]
    
    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )
    
    out_tokens, out_texts = model.text_completion(prompts, max_gen_len=64)
    for text in out_texts:
        print(text)
