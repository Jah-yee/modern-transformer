"""Smoke tests: forward/backward, count_params, config parsing, checkpoint save/load."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import Config, Transformer, parse_args, count_params


def test_config_default():
    old_argv = sys.argv
    sys.argv = ["main.py"]
    try:
        args = parse_args()
        config = Config(args)
    finally:
        sys.argv = old_argv
    assert config.embed_dim == 768
    assert config.num_heads in (12, 16)
    assert config.context_len == 512


def test_config_preset():
    old_argv = sys.argv
    sys.argv = ["main.py"]
    try:
        args = parse_args()
    finally:
        sys.argv = old_argv
    args.preset = "tiny"
    config = Config(args)
    assert config.num_blocks == 2
    assert config.embed_dim == 768
    assert config.num_heads == 12


def test_forward_backward():
    old_argv = sys.argv
    sys.argv = ["main.py"]
    try:
        args = parse_args()
    finally:
        sys.argv = old_argv
    args.preset = "tiny"
    config = Config(args)
    model = Transformer(config).to(config.device)
    x = torch.randint(0, config.vocab_size, (2, config.context_len), device=config.device)
    logits = model(x)
    assert logits.shape == (2, config.context_len, config.vocab_size)
    logits.sum().backward()


def test_count_params():
    old_argv = sys.argv
    sys.argv = ["main.py"]
    try:
        args = parse_args()
    finally:
        sys.argv = old_argv
    args.preset = "tiny"
    config = Config(args)
    count_params(config)


def test_checkpoint_save_load():
    old_argv = sys.argv
    sys.argv = ["main.py"]
    try:
        args = parse_args()
    finally:
        sys.argv = old_argv
    args.preset = "tiny"
    config = Config(args)
    model = Transformer(config).to(config.device)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save({"model_state_dict": model.state_dict()}, path)
        loaded = torch.load(path, map_location=config.device, weights_only=True)
        model2 = Transformer(config).to(config.device)
        model2.load_state_dict(loaded["model_state_dict"], strict=True)
        x = torch.randint(0, config.vocab_size, (1, config.context_len), device=config.device)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        torch.testing.assert_close(out1, out2)
    finally:
        os.unlink(path)
