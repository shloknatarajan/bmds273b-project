"""
Extract per-layer embeddings from frozen DNA language models (Task 1 — regulatory).

Loads each model in eval mode, forward-passes one sequence at a time with
output_hidden_states=True, mean-pools over the sequence axis per layer, writes
one HDF5 per layer.

Design notes (learned the hard way — see logs/ debugging):
  * One sequence per forward pass. The dataset is small (~4k windows), so we
    don't batch. No padding ⇒ no attention mask needed ⇒ uniform handling
    across heterogeneous models (HyenaDNA's forward rejects `attention_mask`,
    so we simply never pass it).
  * Load on CPU then `.to(device)`. `device_map=` puts weights on the `meta`
    device, which crashes models whose __init__ does real tensor work
    (DNABERT-S alibi tensor).
  * Robust pooling handles both (1, seq, hidden) and (seq, hidden) hidden
    states — DNABERT-S's custom BertLayer drops the batch dim.

Models:
    evo2       arcinstitute/evo-2-7b                       (deferred — needs A100)
    hyenadna   LongSafari/hyenadna-large-1m-seqlen-hf
    caduceus   kuleshov-group/caduceus-ps_seqlen-131k_...
    nt         InstaDeepAI/nucleotide-transformer-2.5b-multi-species
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
        # NOTE: the v2 series tops out at 500m; the 2.5B multi-species model is
        # the v1 id below. The old "...-v2-2500m-..." id 404s (HF 401/not-found).
        # max_tok=1000: NT 2.5B's position embeddings only span 1000 tokens;
        # feeding more triggers a CUDA device-side assert (index overflow).
        "hf_id": "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        "loader": "masked", "fp16": True, "trc": False, "max_tok": 1_000,
    },
    "dnabert_s": {
        "hf_id": "zhihan1996/DNABERT-S",
        "loader": "base", "fp16": False, "trc": True, "max_tok": 512,
        # DNABERT-S picks attention by dropout: p_dropout>0 forces the standard
        # PyTorch path; p_dropout==0 uses its bundled Triton-1.x flash kernel
        # (broken on modern Triton, asserts CUDA). We're in eval() so dropout is
        # inactive — this only routes around the broken kernel, embeddings unchanged.
        "extra": {"attention_probs_dropout_prob": 0.1},
        # DNABERT-S's custom BertModel doesn't populate hidden_states, so capture
        # per-layer outputs with forward hooks instead.
        "use_hooks": True,
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
    # low_cpu_mem_usage=False forces real-tensor __init__ instead of meta-device
    # init (transformers 5.x default), which crashes models that do tensor work
    # in __init__ (DNABERT-S alibi). No device_map for the same reason.
    # The dtype kwarg renamed `torch_dtype` -> `dtype` across transformers 4.x/5.x;
    # try the new name, fall back to the old so old-stack images (DNABERT-S) work.
    common = dict(trust_remote_code=cfg["trc"], output_hidden_states=True,
                  low_cpu_mem_usage=False, **cfg.get("extra", {}))
    try:
        model = _LOADERS[cfg["loader"]].from_pretrained(cfg["hf_id"], dtype=dtype, **common)
    except TypeError:
        model = _LOADERS[cfg["loader"]].from_pretrained(cfg["hf_id"], torch_dtype=dtype, **common)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    n_layers   = _cfg_int(model.config, "num_hidden_layers", "n_layer", "num_layers", "n_layers")
    hidden_dim = _cfg_int(model.config, "hidden_size", "d_model", "hidden_dim", "model_dim", "embed_dim", "n_embd")
    print(f"  n_layers={n_layers}, hidden_dim={hidden_dim}")
    return tok, model, n_layers, hidden_dim


def pool_hidden(hs: torch.Tensor) -> np.ndarray:
    """Mean-pool one hidden-state tensor over its sequence axis → (hidden_dim,).

    Handles (1, seq, hidden) [standard] and (seq, hidden) [DNABERT-S drops the
    batch dim].
    """
    t = hs.float()
    if t.dim() == 3:
        v = t.mean(dim=1).squeeze(0)
    elif t.dim() == 2:
        v = t.mean(dim=0)
    else:
        raise RuntimeError(f"unexpected hidden-state shape {tuple(t.shape)}")
    return v.cpu().numpy().astype(np.float32)


@torch.inference_mode()
def embed_one(seq, tok, model, device, max_tok):
    """Return a list of (hidden_dim,) vectors, one per hidden state, for one seq.

    Single sequence, no padding ⇒ no attention mask needed (full attention by
    default), which keeps every model on the same code path.
    """
    enc = tok(seq, return_tensors="pt", truncation=True, max_length=max_tok)
    ids = enc["input_ids"].to(device)
    out = model(input_ids=ids, output_hidden_states=True)
    return [pool_hidden(hs) for hs in _hidden_states(out)]


class LayerHooks:
    """Capture per-layer outputs via forward hooks — for models (DNABERT-S) whose
    forward doesn't return hidden_states. Hooks the embedding module + each
    encoder layer, in order, giving one hidden state per layer."""

    def __init__(self, model):
        self.store = {}
        # locate the BertModel core (AutoModel may wrap it)
        core = getattr(model, "bert", model)
        modules = [core.embeddings] + list(core.encoder.layer)
        self.n = len(modules)
        self._handles = [m.register_forward_hook(self._mk(i)) for i, m in enumerate(modules)]

    def _mk(self, i):
        def fn(_module, _inp, out):
            self.store[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        return fn

    def collect(self):
        return [self.store[i] for i in range(self.n)]

    def remove(self):
        for h in self._handles:
            h.remove()


@torch.inference_mode()
def embed_one_hooked(seq, tok, model, device, max_tok, hooks):
    enc = tok(seq, return_tensors="pt", truncation=True, max_length=max_tok)
    ids = enc["input_ids"].to(device)
    model(input_ids=ids)
    return [pool_hidden(hs) for hs in hooks.collect()]


def _hidden_states(out):
    """Return the per-layer hidden-state sequence from a model output that may be
    a ModelOutput (`.hidden_states`) or a plain tuple (DNABERT-S, return_dict=False)."""
    hs = getattr(out, "hidden_states", None)
    if hs is not None:
        return hs
    if isinstance(out, (tuple, list)):
        # find the element that is itself a sequence of 3D/2D hidden-state tensors
        for item in out:
            if (isinstance(item, (tuple, list)) and len(item) > 1
                    and hasattr(item[0], "dim")):
                return item
    raise RuntimeError("could not locate hidden_states in model output")


def open_h5s(out_dir, n_hidden, n, hidden_dim, resume=False):
    """Open one HDF5 per hidden state. When resume=True, reopen existing files in
    'r+' so prior rows are preserved; otherwise (re)create them fresh."""
    handles = {}
    for i in range(n_hidden):
        path = out_dir / f"layer_{i:02d}.h5"
        if resume and path.exists():
            fh = h5py.File(path, "r+")  # keep prior rows, append remaining
        else:
            if path.exists():
                path.unlink()
            fh = h5py.File(path, "w")
            # No gzip: per-row writes into gzip chunks forced a decompress/
            # recompress cycle every row (×N layers) — the extraction bottleneck.
            # Uncompressed + block-aligned writes are ~100x faster; embeddings are
            # only ~1-2 GB/model uncompressed.
            fh.create_dataset("embeddings", shape=(n, hidden_dim), dtype="float32",
                              chunks=(min(256, n), hidden_dim))
        handles[i] = fh
    return handles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True, choices=list(MODELS))
    p.add_argument("--windows",    default="data/regulatory/windows.tsv")
    p.add_argument("--id_col",     default="window_id")
    p.add_argument("--out_dir",    default=None)
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
    # Resume only if a prior cursor AND the layer files actually exist.
    resume_ok = bool(args.resume and cursor.exists() and (out_dir / "layer_00.h5").exists())
    # A run SIGKILLed mid-write leaves corrupt HDF5s — verify the first opens
    # cleanly, else fall back to a fresh restart instead of crashing.
    if resume_ok:
        try:
            with h5py.File(out_dir / "layer_00.h5", "r"):
                pass
        except Exception:
            print("  prior HDF5s are corrupt (interrupted mid-write); restarting fresh.")
            resume_ok = False
    start_row = int(cursor.read_text()) if resume_ok else 0
    if args.resume and not resume_ok:
        print("  --resume: no usable prior progress; starting fresh.")
    elif resume_ok:
        print(f"  resuming from row {start_row}")

    # DNABERT-S: capture layers via hooks (its forward returns no hidden_states).
    hooks = LayerHooks(model) if MODELS[args.model].get("use_hooks") else None
    embed = ((lambda s: embed_one_hooked(s, tok, model, args.device, max_tok, hooks))
             if hooks else
             (lambda s: embed_one(s, tok, model, args.device, max_tok)))

    # Probe the first sequence to size the HDF5s by the ACTUAL number of hidden
    # states and hidden_dim — the HF wrappers (e.g. HyenaDNA) can return a
    # different count than config.num_hidden_layers+1.
    probe = embed(seqs[0])
    n_hidden, probe_dim = len(probe), probe[0].shape[0]
    print(f"  actual hidden states={n_hidden}, hidden_dim={probe_dim}")

    handles = open_h5s(out_dir, n_hidden, n_windows, probe_dim, resume=resume_ok)
    print(f"Extracting (one seq/pass, max_tok={max_tok}) ...")

    # Buffer BLOCK rows in RAM, then write each layer contiguously (chunk-aligned)
    # — one write per block per layer instead of one per row.
    BLOCK = 256
    bufs = np.zeros((n_hidden, BLOCK, probe_dim), dtype=np.float32)

    def flush(buf_start, count):
        if count == 0:
            return
        for i in range(n_hidden):
            handles[i]["embeddings"][buf_start:buf_start + count] = bufs[i, :count]
        cursor.write_text(str(buf_start + count))

    buf_start, fill = start_row, 0
    for row in range(start_row, n_windows):
        try:
            layers = embed(seqs[row])
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at row {row}")
            flush(buf_start, fill)
            for fh in handles.values(): fh.close()
            raise
        for i, emb in enumerate(layers):
            bufs[i, fill] = emb
        fill += 1
        if fill == BLOCK:
            flush(buf_start, fill)
            buf_start += fill
            fill = 0
            torch.cuda.empty_cache()
            gc.collect()
        if row % 100 == 0 or row == n_windows - 1:
            print(f"  {row + 1}/{n_windows} ({100*(row+1)/n_windows:.1f}%)", end="\r")

    flush(buf_start, fill)  # final partial block
    for fh in handles.values(): fh.close()
    print(f"\nDone → {out_dir}/")
    with h5py.File(out_dir / "layer_00.h5") as fh:
        print(f"  layer_00: {fh['embeddings'].shape}")
    last = n_hidden - 1
    with h5py.File(out_dir / f"layer_{last:02d}.h5") as fh:
        print(f"  layer_{last:02d}: {fh['embeddings'].shape}")


if __name__ == "__main__":
    main()
