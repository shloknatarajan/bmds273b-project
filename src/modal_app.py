"""Modal app for the regulatory (Task 1) pipeline.

This box (EC2) is the control plane; all heavy RAM + GPU work runs in Modal
containers driven from here via the MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in
`.env`. A single persistent Volume holds the data and embedding outputs so they
survive across calls and the 4-model fan-out.

Stages (run from the repo root):

    # one-time: pull GENCODE / ENCODE / hg38 into the Volume (~3.5 GB)
    modal run src/modal_app.py::download

    # fix-then-build: produce windows.tsv on the Volume (32 GB CPU box)
    modal run src/modal_app.py::build

    # smoke test the extractor on a tiny slice before the full run
    modal run src/modal_app.py --stage smoke --models dnabert_s

    # the real thing: 4 non-Evo2 models concurrently, one GPU container each
    modal run src/modal_app.py --stage extract

Evo 2 is intentionally absent — it needs an A100 + the finicky `evo2`/Vortex
install and is handled separately (plan Phase 7).
"""

import subprocess
import sys

import modal

app = modal.App("bmds273b-regulatory")

# Persistent storage: data/regulatory + embeddings live here across all calls.
vol = modal.Volume.from_name("bmds273b-reg-data", create_if_missing=True)
VOL = "/vol"
DATA = f"{VOL}/regulatory"
EMB = f"{DATA}/embeddings"

# Repo source mounted into every image so we can invoke the existing CLIs.
SRC = "/root/src"


def _with_src(image: modal.Image) -> modal.Image:
    return image.add_local_dir("src", remote_path=SRC)


# --- CPU image: data download + window building (pyranges / pyfaidx) ---
cpu_image = _with_src(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc", "g++", "wget", "gzip")
    .pip_install("pandas", "numpy", "pyranges==0.1.4", "pyfaidx>=0.9", "biopython")
)

# --- GPU image for HyenaDNA + NT (verified on transformers 5.x) ---
gpu_image = _with_src(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1",
        "transformers>=4.44",
        "accelerate",
        "einops",
        "sentencepiece",
        "h5py",
        "numpy",
        "pandas",
    )
)

# --- DNABERT-S image: full OLD stack. DNABERT-S bundles a Triton flash-attention
# kernel calling tl.dot(..., trans_b=True), removed in Triton >=2.1. Triton 2.0
# (which has trans_b) ships with torch 2.0.x, which needs Python 3.10. transformers
# <5 also avoids the meta-device __init__ crash. So everything is pinned old. ---
dnabert_image = _with_src(
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.0.1",          # pulls triton 2.0.0 (has trans_b)
        "transformers==4.44.2",  # <5: no meta-device init crash
        "accelerate",
        "einops",
        "h5py",
        "numpy<2",               # numpy 2 is incompatible with the torch-2.0.1 era
        "pandas",
    )
)

# NOTE: Caduceus lives in a SEPARATE app file (src/modal_caduceus.py). Its
# mamba-ssm / causal-conv1d CUDA extensions need a torch-matched prebuilt wheel,
# and Modal eagerly builds every registered function's image at `modal run` — so
# keeping a fragile Caduceus image in THIS file would block the working models.
MODEL_CFG = {
    "hyenadna":  {"batch": 1},
    "nt":        {"batch": 4},
    "dnabert_s": {"batch": 8},
}
DEFAULT_MODELS = ["hyenadna", "nt", "dnabert_s"]


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root")


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=60 * 60)
def download():
    """Download GENCODE / ENCODE / hg38 into the Volume and pyfaidx-index hg38."""
    import os

    os.makedirs(DATA, exist_ok=True)
    env = {**os.environ, "DATA_DIR": DATA}
    print("+ bash src/regulatory/download_data.sh", flush=True)
    subprocess.run(
        ["bash", f"{SRC}/regulatory/download_data.sh"],
        check=True, cwd="/root", env=env,
    )
    vol.commit()
    print("download complete; contents:")
    _run(["ls", "-la", DATA])


@app.function(image=cpu_image, volumes={VOL: vol}, memory=32768, timeout=2 * 60 * 60)
def build(n_windows: int = 5000, pool_size: int = 40000):
    """Build GC-matched, center-on-site windows → windows.tsv on the Volume.

    Runs with 32 GB RAM so the pyranges interval ops never OOM (the blocker that
    killed this on the 7.6 GB EC2 box).
    """
    vol.reload()
    _run([
        sys.executable, f"{SRC}/regulatory/build_windows.py",
        "--gencode_gtf",  f"{DATA}/gencode.v47.annotation.gtf.gz",
        "--encode_bed",   f"{DATA}/GRCh38-cCREs.bed",
        "--genome_fasta", f"{DATA}/hg38.fa",
        "--out_dir",      DATA,
        "--n_windows",    str(n_windows),
        "--pool_size",    str(pool_size),
    ])
    vol.commit()
    _run(["cat", f"{DATA}/split_stats.txt"])


def _extract_impl(model: str, limit: int | None, resume: bool = False, device: str = "cuda"):
    """Shared body for the per-model extraction (full run or smoke slice)."""
    vol.reload()
    out_dir = f"{EMB}/{model}"
    windows = f"{DATA}/windows.tsv"
    if limit is not None:
        # Smoke test: head the windows into a tiny temp TSV so the path is real
        # but cheap, without touching the GPU code path.
        import pandas as pd

        slice_path = f"{DATA}/windows_smoke_{model}.tsv"
        pd.read_csv(windows, sep="\t").head(limit).to_csv(slice_path, sep="\t", index=False)
        windows = slice_path
        out_dir = f"{EMB}/_smoke_{model}"
    # extract_embeddings processes one sequence per pass (no batching), so no
    # --batch_size flag.
    cmd = [
        sys.executable, f"{SRC}/extract_embeddings.py",
        "--model", model,
        "--windows", windows,
        "--out_dir", out_dir,
        "--device", device,
    ]
    if resume:
        cmd.append("--resume")  # continue from .cursor instead of re-extracting
    _run(cmd)
    vol.commit()
    return out_dir


# Two extraction functions: the 3 standard HF models share gpu_image; Caduceus
# gets its isolated mamba-ssm image. All run on A10G, so the only thing that
# differs is the image — hence two functions rather than a per-call override
# (Modal 1.4 Functions have no per-call .options(image=...)).
@app.function(image=gpu_image, volumes={VOL: vol}, gpu="A10G", timeout=4 * 60 * 60)
def extract(model: str, limit: int | None = None, resume: bool = False):
    return _extract_impl(model, limit, resume)


# DNABERT-S on GPU. We force the standard (non-Triton) attention path via
# attention_probs_dropout_prob>0 (see MODELS["dnabert_s"]["extra"]), so the
# broken Triton-1.x flash kernel is never invoked. Old-stack image only because
# transformers<5 avoids the meta-device __init__ crash.
@app.function(image=dnabert_image, volumes={VOL: vol}, gpu="A10G", timeout=4 * 60 * 60)
def extract_dnabert(model: str, limit: int | None = None, resume: bool = False):
    return _extract_impl(model, limit, resume, device="cuda")


def _spawn_for(model: str, limit: int | None, resume: bool):
    fn = extract_dnabert if model == "dnabert_s" else extract
    return fn.spawn(model, limit, resume)


@app.local_entrypoint()
def main(stage: str = "extract", models: str = "", limit: int = 0, resume: bool = False):
    """Drive the pipeline from this box.

    stage:  extract (full) | smoke (tiny slice)
    models: comma list to override the defaults
    limit:  rows for smoke (default 50 when stage=smoke)
    resume: continue each model from its .cursor instead of re-extracting
    """
    selected = [m.strip() for m in models.split(",") if m.strip()] or DEFAULT_MODELS
    for m in selected:
        if m not in MODEL_CFG:
            raise SystemExit(f"unknown model {m!r}; choose from {list(MODEL_CFG)}")

    smoke_limit = (limit or 50) if stage == "smoke" else None
    print(f"stage={stage} models={selected} limit={smoke_limit} resume={resume}")

    # Fan out: spawn all models (each lands in its own A10G container), then
    # collect. Each spawn/get is guarded so one model's failure never discards
    # the rest.
    handles, failures = [], {}
    for m in selected:
        try:
            handles.append((m, _spawn_for(m, smoke_limit, resume)))
        except Exception as e:
            failures[m] = f"spawn/build: {e}"
            print(f"[{m}] SPAWN FAILED: {e}")

    for m, h in handles:
        try:
            out = h.get()
            print(f"[{m}] done → {out}")
        except Exception as e:
            failures[m] = f"run: {e}"
            print(f"[{m}] RUN FAILED: {e}")

    ok = [m for m, _ in handles if m not in failures]
    print(f"\nSUMMARY: {len(ok)}/{len(selected)} succeeded: {ok}")
    if failures:
        print(f"  failed: {failures}")
