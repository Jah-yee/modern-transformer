import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import os
import time
import sys

import tiktoken
from torch.utils.data import Dataset, DataLoader

import torchinfo

import wandb
from dotenv import load_dotenv

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = "PROJECT_NAME"
WANDB_ENTITY = "TEAM_NAME"

# ----------------------------
# RoPE implementation from GPT OSS release, see: https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
# ----------------------------
def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )

            d_half = self.head_dim / 2
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key

class Config():

    def __init__(self):
        # ----------------------------
        # architecture configs
        # ----------------------------
        TOKENIZER_CONFIGS = {
            "gpt2": {"vocab_size": 50257},
            "o200k_base": {"vocab_size": 200019}
        }

        self.tokenizer_name = "gpt2"
        self.tokenizer = tiktoken.get_encoding(self.tokenizer_name)
        self.vocab_size = TOKENIZER_CONFIGS[self.tokenizer_name]["vocab_size"]
        self.embed_dim = 768

        self.num_heads = 16
        self.attention_dim = self.embed_dim // self.num_heads
        self.context_len = 512
        self.ffn_dim = int(2 * self.embed_dim / 3)

        self.num_blocks = 2

        assert self.embed_dim % self.num_heads == 0, f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"

        # ----------------------------
        # RoPE configs
        # ----------------------------
        self.rope_theta = 10000.0
        self.rope_scaling_factor = 1.0
        self.rope_ntk_alpha = 1.0
        self.rope_ntk_beta = 32.0

        # ----------------------------
        # regularization configs
        # ----------------------------
        self.grad_norm = False
        self.grad_norm_value = 1.0
        self.qk_norm = False

        # ----------------------------
        # data configs
        # ----------------------------

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        if torch.cuda.is_available():
            self.device = f"cuda:{self.local_rank}"
        elif torch.backends.mps.is_available() and self.world_size == 1:
            self.device = "mps"
        else:
            self.device = "cpu"

        self.dtype = torch.bfloat16
        self.batch_size = 4 # batch size is limited by memory
        self.target_batch_size = 512 # target batch size is the target for training stability
        assert self.target_batch_size % self.batch_size == 0, f"target_batch_size ({self.target_batch_size}) must be divisible by batch_size ({self.batch_size})"
        self.accum_steps = self.target_batch_size / (self.batch_size * self.world_size) # accum steps simulates larger batches
        self.max_batches = None

        print("# ----------------------------")
        print("using device: ", self.device)
        print("using dtype: ", self.dtype)
        print("using batch size of ", self.batch_size) 
        print("max batches of ", self.max_batches)     
        print("# ----------------------------")  

    def print_config(self):
        pass

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.attention_dim = self.embed_dim // self.num_heads

        self.W_qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False, dtype=config.dtype)
        self.W_out = nn.Linear(self.num_heads * self.attention_dim, config.embed_dim, bias=False, dtype=config.dtype)

        nn.init.normal_(self.W_qkv.weight, std=0.02)
        nn.init.normal_(self.W_out.weight, std=0.02)
  
        self.log_tau = nn.Parameter(torch.zeros(self.num_heads, dtype=config.dtype))

        self.rope = RotaryEmbedding(
            head_dim=self.attention_dim,
            base=config.rope_theta,
            dtype=config.dtype,
            initial_context_length=config.context_len,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=config.device,
        )

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]

        qkv = self.W_qkv(x)
        queries, keys, values = torch.chunk(qkv, 3, dim=-1)

        # reshaping for RoPE
        queries = einops.rearrange(queries, "batch seq_len (num_heads head_dim) -> seq_len (batch num_heads) head_dim", num_heads=self.num_heads, head_dim=self.attention_dim)
        keys = einops.rearrange(keys, "batch seq_len (num_heads head_dim) -> seq_len (batch num_heads) head_dim", num_heads=self.num_heads, head_dim=self.attention_dim)
        values = einops.rearrange(values, "batch seq_len (num_heads head_dim) -> batch seq_len num_heads head_dim", num_heads=self.num_heads)

        queries,keys = self.rope(queries, keys)

        # reshape back to (batch, num_heads, seq_len, head_dim) for attention
        queries = einops.rearrange(queries, "seq_len (batch num_heads) head_dim -> batch num_heads seq_len head_dim", batch=batch_size)
        keys = einops.rearrange(keys, "seq_len (batch num_heads) head_dim -> batch num_heads seq_len head_dim", batch=batch_size)
        values = einops.rearrange(values, "batch seq_len num_heads head_dim -> batch num_heads seq_len head_dim")

        attention = F.scaled_dot_product_attention(
            queries, keys, values,
            is_causal=True,
        )
        concatenated = einops.rearrange(attention, "batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)")
        final_out = self.W_out(concatenated)

        return final_out

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_layer = Attention(config)
        self.ffn_layer = FFN(config)
        self.rms_norm_1 = nn.RMSNorm(config.embed_dim, eps=1e-6)
        self.rms_norm_2 = nn.RMSNorm(config.embed_dim, eps=1e-6)

    def forward(self, x):

        x = x + self.attn_layer(self.rms_norm_1(x))
        x = x + self.ffn_layer(self.rms_norm_2(x))

        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_blocks = config.num_blocks
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_blocks)
        ])

        self.final_rms_norm = nn.RMSNorm(self.embed_dim, eps=1e-6)
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embed_dim, dtype=config.dtype)
        nn.init.normal_(self.embedding_layer.weight, std=0.02)

        self.final_linear = nn.Linear(self.embed_dim, self.vocab_size, bias=False, dtype=config.dtype)
        self.final_linear.weight = self.embedding_layer.weight

    def forward(self, x):
        x = self.embedding_layer(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_rms_norm(x)
        logits = self.final_linear(x)
        return logits

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.embed_dim
        self.ffn_dim = config.ffn_dim

        self.gate = nn.Linear(self.embed_dim, self.ffn_dim, bias=False, dtype=config.dtype)
        self.data = nn.Linear(self.embed_dim, self.ffn_dim, bias=False, dtype=config.dtype)
        self.out = nn.Linear(self.ffn_dim, self.embed_dim, bias=False, dtype=config.dtype)

        nn.init.xavier_uniform_(self.data.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.0, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        gate_output = F.silu(self.gate(x))
        data_output = self.data(x)
        return self.out(gate_output * data_output)

class TinyShakespeare(Dataset):
    def __init__(self, file, tokenizer, context_len: int):
        with open(file, 'r') as f:
            text = f.read()

        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.context_len]
        y = self.tokens[idx + 1:idx + self.context_len + 1]
        return x, y

def training(config, run_name):
    if config.world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config=vars(config)
        )

    dataset = TinyShakespeare("data/tiny_shakespeare.txt", config.tokenizer, config.context_len)
    if config.world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            sampler = sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    total_batches = len(dataloader)
    print("#### total batches: ", total_batches, " ####")

    model = Transformer(config).to(config.device)
    model = torch.compile(model, mode="default") # modes are `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`
    if config.world_size > 1:
        model = DDP(model, device_ids=[config.local_rank] if torch.cuda_is_available() else None)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches)

    total_loss = 0
    training_start_time = time.perf_counter()
    for epoch in range(1):
        for batch_idx, (x, y) in enumerate(dataloader):
            start_time = time.perf_counter()

            x, y = x.to(config.device, non_blocking=True), y.to(config.device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, config.vocab_size), y.view(-1))
            loss = loss / config.accum_steps  # Scale loss by accumulation steps
            total_loss += loss

            loss.backward()
            if (batch_idx+1) % config.accum_steps == 0: # if the batch_idx is a multiple of the accumulation steps do a backprop
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            elapsed = time.perf_counter() - start_time
            tokens_per_sec = (x.shape[0] * x.shape[1]) / elapsed # batch size * sequence length = total tokens in batch


            time_elapsed = time.perf_counter() - training_start_time
            
            if rank == 0:

                wandb.log({"loss": loss.item() * config.accum_steps, "epoch": epoch, "batch": batch_idx, "tokens_per_sec": tokens_per_sec, "perplexity": math.exp(loss.item() * config.accum_steps), "lr": scheduler.get_last_lr()[0], "time_elapsed": time_elapsed})

                if epoch == 0 and batch_idx == 0:
                    print("theoretical start loss: ", math.log(config.vocab_size))
                if batch_idx % 1 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{total_batches}, Loss: {loss.item() * config.accum_steps:.4f}, Tok/s: {tokens_per_sec:.0f}")
                
                if config.max_batches != None and batch_idx >= config.max_batches:
                    print(f"max batch of {config.max_batches} reached, exiting training process")

            if config.max_batches != None and batch_idx >= config.max_batches:
                break

    if rank == 0:
        wandb.finish()
    if config.world_size > 1:
        dist.destroy_process_group()

def count_params(config):
    model = Transformer(config).to(config.device)
    dummy_input = torch.randint(0, config.vocab_size, (4, config.context_len)).to(config.device)
    torchinfo.summary(model, input_data=dummy_input, verbose=1)

def inference(model, config):
    prefixes = [
        "What is",
        "How to",
        "Can you",
        "What can",
        "How does",
        "What happens",
        "What is the",
        "What are",
        "What is the purpose of",
        "What is the best way to"
    ]
    tokenized_prefixes = [config.tokenizer.encode(p) for p in prefixes]

    for i in tokenized_prefixes:
        print(i + " " + config.tokenizer.decode(model(i)))

if __name__ == "__main__":
    if "--run" in sys.argv:
        run_name = sys.argv[sys.argv.index("--run") + 1]
    else:
        run_name = None

    config = Config()
    training(config, run_name)