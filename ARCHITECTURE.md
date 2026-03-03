# Architecture

- **RoPE**: Rotary position embeddings on Q/K; supports NTK-aware scaling for longer context.
- **Pre-RMSNorm**: Each block applies RMSNorm before attention and before FFN, then residual add.
- **SwiGLU FFN**: Gate and up projections, SiLU(gate) * up, then down projection.
- **Attention**: Multi-head with optional QK RMSNorm and per-head temperature. SDPA with optional Flash Attention 2 on CUDA.
- **Training**: BF16, gradient accumulation, optional grad-norm clipping; DDP for multi-GPU; Metal on macOS.

Data flow: tokens → embedding → N × (PreNorm → Attention → residual, PreNorm → FFN → residual) → final RMSNorm → LM head (tied with embedding).
