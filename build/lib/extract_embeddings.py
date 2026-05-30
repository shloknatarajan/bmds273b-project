"""
Extract per-layer embeddings from frozen DNA language models (Task 1 — regulatory).

Loads each model in eval mode, forward passes with output_hidden_states=True,
mean-pools over the sequence axis per layer, writes to HDF5.

Models:
    evo2       arcinstitute/evo-2-7b                               
    hyenadna   LongSafari/hyenadna-large-1m-seqlen-hf            
    caduceus   kuleshov-group/caduceus-ps_seqlen-131k_...          
    nt         InstaDeepAI/nucleotide-transformer-v2-2500m-...    
    dnabert_s  zhihan1996/DNABERT-S                                

Output: <out_dir>/layer_{i:02d}.h5  shape (N, hidden_dim) float32
        <out_dir>/window_ids.txt


Usage:
    python src/extract_embeddings.py --model nt --out_dir data/embeddings/nt --device cuda
"""

import argparse
import gc
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


MODELS = {
    "evo2": {
        "hf_id": "arcinstitute/evo-2-7b",
        "loader": "causal", "fp16": True, "trc": True, "max_tok": 8_192,
    },
    "hyenadna": {
        "hf_id": "LongSafari/hyenadna-large-1m-seqlen-hf",
        "loader": "base", "fp16": False, "trc": True, "max_tok": 1_000_000,
    },
    "caduceus": {
        "hf_id": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
        "loader": "masked", "fp16": False, "trc": True, "max_tok": 131_072,
    },
    "nt": {
        "hf_id": "InstaDeepAI/nucleotide-transformer-v2-2500m-multi-species",
        "loader": "masked", "fp16": True, "trc": False, "max_tok": 2_048,
    },
    "dnabert_s": {
        "hf_id": "zhihan1996/DNABERT-S",
        "loader": "base", "fp16": False, "trc": True, "max_tok": 512,
    },
}

_LOADERS = {"causal": AutoModelForCausalLM, "masked": AutoModelForMaskedLM, "base": AutoModel}


def _cfg_int(config, *attrs):
    return next(getattr(config, a) for a in attrs if isinstance(getattr(config, a, None), int))


def load_model(name, device):
    cfg = MODELS[name]
    dtype = torch.float16 if cfg["fp16"] else torch.float32
    print(f"Loading {name} ({'fp16' if cfg['fp16'] else 'fp32'}) ...")
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"], trust_remote_code=cfg["trc"])
    model = _LOADERS[cfg["loader"]].from_pretrained(
        cfg["hf_id"], trust_remote_code=cfg["trc"],
        torch_dtype=dtype, device_map=device, output_hidden_states=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    n_layers   = _cfg_int(model.config, "num_hidden_layers", "n_layer", "num_layers", "n_layers")
    hidden_dim = _cfg_int(model.config, "hidden_size", "d_model", "hidden_dim", "model_dim", "embed_dim", "n_embd")
    print(f"  n_layers={n_layers}, hidden_dim={hidden_dim}")
    return tok, model, n_layers, hidden_dim


@torch.inference_mode()
def embed_batch(seqs, tok, model, device, max_tok):
    enc  = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_tok)
    ids  = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    if mask is not None:
        mask = mask.to(device)
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    m = (mask if mask is not None else torch.ones_like(ids)).unsqueeze(-1).float()
    return [
        ((hs.float() * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy().astype(np.float32)
        for hs in out.hidden_states
    ]


def open_h5s(out_dir, n_layers, n, hidden_dim):
    handles = {}
    for i in range(n_layers + 1):
        path = out_dir / f"layer_{i:02d}.h5"
        if path.exists():
            path.unlink()
        fh = h5py.File(path, "w")
        fh.create_dataset("embeddings", shape=(n, hidden_dim), dtype="float32",
                          chunks=(min(256, n), hidden_dim), compression="gzip", compression_opts=4)
        handles[i] = fh
    return handles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True, choices=list(MODELS))
    p.add_argument("--windows",    default="data/regulatory/windows.tsv")
    p.add_argument("--id_col",     default="window_id")
    p.add_argument("--out_dir",    default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--resume",     action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"data/embeddings/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df         = pd.read_csv(args.windows, sep="\t")
    seqs       = df["seq"].tolist()
    window_ids = df[args.id_col].astype(str).tolist()
    n_windows  = len(df)
    print(f"{n_windows} windows from {args.windows}")

    tok, model, n_layers, hidden_dim = load_model(args.model, args.device)
    max_tok = MODELS[args.model]["max_tok"]

    (out_dir / "window_ids.txt").write_text("\n".join(window_ids))
    cursor    = out_dir / ".cursor"
    start_row = int(cursor.read_text()) if args.resume and cursor.exists() else 0

    handles = open_h5s(out_dir, n_layers, n_windows, hidden_dim)
    bs = args.batch_size
    print(f"Extracting (batch_size={bs}, max_tok={max_tok}) ...")

    for row_start in range(start_row, n_windows, bs):
        row_end = min(row_start + bs, n_windows)
        try:
            layers = embed_batch(seqs[row_start:row_end], tok, model, args.device, max_tok)
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at row {row_start} — try --batch_size 1")
            for fh in handles.values(): fh.close()
            raise
        for layer_idx, embs in enumerate(layers):
            handles[layer_idx]["embeddings"][row_start:row_end] = embs
        cursor.write_text(str(row_end))
        print(f"  {row_end}/{n_windows} ({100*row_end/n_windows:.1f}%)", end="\r")
        torch.cuda.empty_cache()
        gc.collect()

    for fh in handles.values(): fh.close()
    print(f"\nDone → {out_dir}/")
    with h5py.File(out_dir / "layer_00.h5") as fh:
        print(f"  layer_00: {fh['embeddings'].shape}")
    with h5py.File(out_dir / f"layer_{n_layers:02d}.h5") as fh:
        print(f"  layer_{n_layers:02d}: {fh['embeddings'].shape}")


if __name__ == "__main__":
    main()
