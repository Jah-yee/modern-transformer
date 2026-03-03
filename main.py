import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import os
import time
import sys

import tiktoken
from torch.utils.data import Dataset, DataLoader, random_split

import torchinfo

import wandb
from dotenv import load_dotenv

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "PROJECT_NAME")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "TEAM_NAME")


def parse_args():
    parser = argparse.ArgumentParser(description="Modern Transformer pretraining")
    parser.add_argument("--run", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--data", type=str, default="data/tiny_shakespeare.txt", help="Path to training data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--context_len", type=int, default=512, help="Context length")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches per run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project (overrides env)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (overrides env)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--checkpoint_every", type=int, default=None, help="Save checkpoint every N optimizer steps (default: disabled)")
    parser.add_argument("--grad_norm", type=float, default=None, help="Gradient norm clipping value (default: disabled)")
    parser.add_argument("--qk_norm", action="store_true", help="Apply RMSNorm to Q/K in attention (LLaMA-style)")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Fraction of data for validation (0 = disabled)")
    parser.add_argument("--val_every", type=int, default=None, help="Run validation every N optimizer steps (default: disabled)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "count_params", "inference"], help="Mode: train, count_params, or inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (required for inference)")
    parser.add_argument("--use_flash2", action="store_true", help="Use Flash Attention 2 backend for SDPA when available (CUDA)")
    parser.add_argument("--preset", type=str, default=None, choices=["tiny", "small", "base"], help="Architecture preset: tiny (2L), small (6L), base (12L); all 768d, 12 heads")
    parser.add_argument("--position_encoding", type=str, default="rope", choices=["rope", "alibi"], help="Position encoding: rope (default) or alibi")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear", "constant"], help="LR scheduler: cosine, linear, or constant")
    return parser.parse_args()


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
    """RoPE with optional NTK-aware scaling for longer context."""

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

class Config:
    """Training and model config; overrides from CLI args when provided."""

    def __init__(self, args=None):
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
        self.use_flash2 = False
        self.position_encoding = "rope"
        self.scheduler = "cosine"

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
        self.data_path = "data/tiny_shakespeare.txt"
        self.checkpoint_every = None
        self.resume = None
        self.val_ratio = 0.0
        self.val_every = None

        if args is not None:
            preset = getattr(args, "preset", None)
            if preset is not None:
                PRESETS = {"tiny": (2, 768, 12), "small": (6, 768, 12), "base": (12, 768, 12)}
                self.num_blocks, self.embed_dim, self.num_heads = PRESETS[preset]
                self.attention_dim = self.embed_dim // self.num_heads
                self.ffn_dim = int(2 * self.embed_dim / 3)
            self.data_path = getattr(args, "data", self.data_path)
            self.batch_size = getattr(args, "batch_size", self.batch_size)
            self.context_len = getattr(args, "context_len", self.context_len)
            self.max_batches = getattr(args, "max_batches", self.max_batches)
            self.checkpoint_every = getattr(args, "checkpoint_every", None)
            self.resume = getattr(args, "resume", None)
            if getattr(args, "grad_norm", None) is not None:
                self.grad_norm = True
                self.grad_norm_value = args.grad_norm
            self.qk_norm = getattr(args, "qk_norm", False)
            self.val_ratio = getattr(args, "val_ratio", 0.0) or 0.0
            self.val_every = getattr(args, "val_every", None)
            self.use_flash2 = getattr(args, "use_flash2", False)
            self.position_encoding = getattr(args, "position_encoding", "rope")
            self.scheduler = getattr(args, "scheduler", "cosine")
            assert self.target_batch_size % self.batch_size == 0, f"target_batch_size ({self.target_batch_size}) must be divisible by batch_size ({self.batch_size})"
            self.accum_steps = self.target_batch_size / (self.batch_size * self.world_size)

        print("# ----------------------------")
        print("using device: ", self.device)
        print("using dtype: ", self.dtype)
        print("using batch size of ", self.batch_size) 
        print("max batches of ", self.max_batches)     
        print("# ----------------------------")  

    def print_config(self):
        pass

class Attention(nn.Module):
    """Multi-head attention with RoPE, optional QK norm, and per-head temperature (log_tau)."""

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

        self.qk_norm = getattr(config, "qk_norm", False)
        self.qk_norm_layer = nn.RMSNorm(self.attention_dim, eps=1e-6) if self.qk_norm else None
        self.use_flash2 = getattr(config, "use_flash2", False)
        self.position_encoding = getattr(config, "position_encoding", "rope")
        self.rope = RotaryEmbedding(
            head_dim=self.attention_dim,
            base=config.rope_theta,
            dtype=config.dtype,
            initial_context_length=config.context_len,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=config.device,
        ) if self.position_encoding == "rope" else None
        n_h = self.num_heads
        if self.position_encoding == "alibi":
            self.register_buffer("alibi_slopes", torch.pow(2.0, -torch.arange(1, n_h + 1, dtype=torch.float32) * (8.0 / n_h)))

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]

        qkv = self.W_qkv(x)
        queries, keys, values = torch.chunk(qkv, 3, dim=-1)

        # reshaping for RoPE
        queries = einops.rearrange(queries, "batch seq_len (num_heads head_dim) -> seq_len (batch num_heads) head_dim", num_heads=self.num_heads, head_dim=self.attention_dim)
        keys = einops.rearrange(keys, "batch seq_len (num_heads head_dim) -> seq_len (batch num_heads) head_dim", num_heads=self.num_heads, head_dim=self.attention_dim)
        values = einops.rearrange(values, "batch seq_len (num_heads head_dim) -> batch seq_len num_heads head_dim", num_heads=self.num_heads)

        if self.rope is not None:
            queries, keys = self.rope(queries, keys)

        # reshape back to (batch, num_heads, seq_len, head_dim) for attention
        queries = einops.rearrange(queries, "seq_len (batch num_heads) head_dim -> batch num_heads seq_len head_dim", batch=batch_size)
        keys = einops.rearrange(keys, "seq_len (batch num_heads) head_dim -> batch num_heads seq_len head_dim", batch=batch_size)
        values = einops.rearrange(values, "batch seq_len num_heads head_dim -> batch num_heads seq_len head_dim")

        if self.qk_norm_layer is not None:
            queries = self.qk_norm_layer(queries)
            keys = self.qk_norm_layer(keys)
        scale = torch.exp(-self.log_tau / 2).view(1, -1, 1, 1)
        queries = queries * scale.to(queries.dtype)
        keys = keys * scale.to(keys.dtype)

        alibi_mask = None
        if self.position_encoding == "alibi":
            seq_len = queries.shape[2]
            positions = torch.arange(seq_len, device=queries.device, dtype=torch.float32)
            dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).clamp(min=0)
            alibi_mask = (-self.alibi_slopes.view(1, -1, 1, 1).to(queries.device) * dist.unsqueeze(0)).to(queries.dtype)

        if self.use_flash2 and queries.is_cuda and alibi_mask is None:
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attention = F.scaled_dot_product_attention(queries, keys, values, attn_mask=alibi_mask, is_causal=True)
            except Exception:
                attention = F.scaled_dot_product_attention(queries, keys, values, attn_mask=alibi_mask, is_causal=True)
        else:
            attention = F.scaled_dot_product_attention(queries, keys, values, attn_mask=alibi_mask, is_causal=True)
        concatenated = einops.rearrange(attention, "batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)")
        final_out = self.W_out(concatenated)

        return final_out

class TransformerBlock(nn.Module):
    """Pre-norm block: RMSNorm -> Attention -> residual, RMSNorm -> FFN -> residual."""

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
    """Decoder-only transformer with tied input/output embeddings."""

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
    """SwiGLU feed-forward: gate(x) * up(x) -> down."""

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
    """Sliding-window next-token dataset from a text file."""

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

def training(config, run_name=None, no_wandb=False, wandb_project=None, wandb_entity=None):
    project = wandb_project if wandb_project is not None else WANDB_PROJECT
    entity = wandb_entity if wandb_entity is not None else WANDB_ENTITY
    if config.world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
    else:
        rank = 0

    data_dir = os.path.dirname(config.data_path)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    if rank == 0 and not no_wandb:
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=vars(config)
        )

    dataset = TinyShakespeare(config.data_path, config.tokenizer, config.context_len)
    val_loader = None
    if getattr(config, "val_ratio", 0) and config.val_ratio > 0:
        n = len(dataset)
        n_val = max(1, int(n * config.val_ratio))
        n_train = n - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_dataset = train_ds
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    else:
        train_dataset = dataset

    if config.world_size > 1:
        sampler = DistributedSampler(train_dataset, shuffle=True)
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    else:
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
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
        model = DDP(model, device_ids=[config.local_rank] if torch.cuda.is_available() else None)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    sched_type = getattr(config, "scheduler", "cosine")
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches)
    elif sched_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_batches)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    opt_step = 0
    if getattr(config, "resume", None):
        ckpt = torch.load(config.resume, map_location=config.device, weights_only=True)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        opt_step = ckpt.get("opt_step", 0)
        if rank == 0:
            print(f"Resumed from {config.resume} at opt_step {opt_step}")

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
            if (batch_idx + 1) % config.accum_steps == 0:
                if getattr(config, "grad_norm", False):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_value)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                opt_step += 1
                if getattr(config, "checkpoint_every", None) and opt_step > 0 and opt_step % config.checkpoint_every == 0 and rank == 0:
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save({
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "opt_step": opt_step,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                    }, f"checkpoint_step_{opt_step}.pt")
                    print(f"Saved checkpoint_step_{opt_step}.pt")
                if val_loader and getattr(config, "val_every", None) and opt_step > 0 and opt_step % config.val_every == 0 and rank == 0:
                    raw_model = model.module if hasattr(model, "module") else model
                    raw_model.eval()
                    val_loss_sum, val_n = 0.0, 0
                    with torch.no_grad():
                        for vx, vy in val_loader:
                            vx, vy = vx.to(config.device), vy.to(config.device)
                            vlogits = model(vx)
                            val_loss_sum += F.cross_entropy(vlogits.float().view(-1, config.vocab_size), vy.view(-1), reduction="sum").item()
                            val_n += vx.numel()
                    raw_model.train()
                    val_loss = val_loss_sum / max(val_n, 1)
                    if not no_wandb:
                        wandb.log({"val_loss": val_loss, "val_perplexity": math.exp(val_loss), "val_at_step": opt_step})
                    print(f"Val step {opt_step} loss: {val_loss:.4f} perplexity: {math.exp(val_loss):.4f}")

            elapsed = time.perf_counter() - start_time
            tokens_per_sec = (x.shape[0] * x.shape[1]) / elapsed # batch size * sequence length = total tokens in batch


            time_elapsed = time.perf_counter() - training_start_time
            
            if rank == 0:
                if not no_wandb:
                    wandb.log({"loss": loss.item() * config.accum_steps, "epoch": epoch, "batch": batch_idx, "tokens_per_sec": tokens_per_sec, "perplexity": math.exp(loss.item() * config.accum_steps), "lr": scheduler.get_last_lr()[0], "time_elapsed": time_elapsed})
                if epoch == 0 and batch_idx == 0:
                    print("theoretical start loss: ", math.log(config.vocab_size))
                if batch_idx % 1 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{total_batches}, Loss: {loss.item() * config.accum_steps:.4f}, Tok/s: {tokens_per_sec:.0f}")
                
                if config.max_batches != None and batch_idx >= config.max_batches:
                    print(f"max batch of {config.max_batches} reached, exiting training process")

            if config.max_batches != None and batch_idx >= config.max_batches:
                break

    if rank == 0 and not no_wandb:
        wandb.finish()
    if config.world_size > 1:
        dist.destroy_process_group()

def count_params(config):
    model = Transformer(config).to(config.device)
    dummy_input = torch.randint(0, config.vocab_size, (4, config.context_len)).to(config.device)
    torchinfo.summary(model, input_data=dummy_input, verbose=1)

def inference(model, config, num_new_tokens=20):
    """Run greedy decoding from fixed prefixes."""
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
    model.eval()
    with torch.no_grad():
        for prefix in prefixes:
            ids = config.tokenizer.encode(prefix)
            x = torch.tensor([ids], dtype=torch.long, device=config.device)
            for _ in range(num_new_tokens):
                logits = model(x)
                next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
                x = torch.cat([x, next_id], dim=1)
            text = config.tokenizer.decode(x[0].tolist())
            print(prefix + " -> " + text)

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    config = Config(args)
    if args.mode == "train":
        training(
            config,
            run_name=args.run,
            no_wandb=args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )
    elif args.mode == "count_params":
        count_params(config)
    elif args.mode == "inference":
        if not args.checkpoint:
            sys.exit("--checkpoint is required for inference")
        model = Transformer(config).to(config.device)
        ckpt = torch.load(args.checkpoint, map_location=config.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        inference(model, config)