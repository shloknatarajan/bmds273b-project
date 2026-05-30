"""Caduceus extraction — isolated Modal app.

Caduceus needs mamba-ssm + causal-conv1d, whose CUDA extensions must match the
runtime torch ABI exactly. Compiling on Modal's GPU-less builder fails, and the
PyPI wheels were built against a different torch (the `undefined symbol` crash).
Fix: install official GitHub-release wheels for **torch 2.6 + cu12 + cp312 +
cxx11abiFALSE**, matching PyPI torch's ABI (the torch-2.7 wheels only ship
cxx11abiTRUE, which mismatches PyPI torch 2.7's FALSE ABI → undefined symbol).
torch 2.6 + the FALSE wheel matches, so no compile is needed.

Kept in its own app file because Modal builds every registered function's image
at `modal run`, so a fragile Caduceus image must not live in modal_app.py.

    modal run src/modal_caduceus.py --limit 20   # smoke
    modal run src/modal_caduceus.py              # full
"""

import subprocess
import sys

import modal

app = modal.App("bmds273b-caduceus")
vol = modal.Volume.from_name("bmds273b-reg-data", create_if_missing=True)
VOL = "/vol"
DATA = f"{VOL}/regulatory"
EMB = f"{DATA}/embeddings"
SRC = "/root/src"

# Prebuilt mamba/causal wheels all mismatched PyPI torch's ABI (undefined symbol),
# even the abiFALSE/torch2.6 variants — that wheel is simply mis-built. So compile
# from source against the exact torch in a CUDA-devel image (has nvcc). FORCE_BUILD
# + --no-build-isolation make pip build the extensions, not fetch a wheel.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("setuptools", "wheel", "packaging", "ninja",
                 "torch==2.6.0", "transformers>=4.44", "accelerate", "einops",
                 "h5py", "numpy", "pandas")
    # TORCH_CUDA_ARCH_LIST=8.6 (A10G only) compiles ONE arch instead of 4 — cuts
    # builder RAM ~4x (the multi-arch compile was OOM-ing). MAX_JOBS=2 trims more.
    .env({"MAX_JOBS": "2", "TORCH_CUDA_ARCH_LIST": "8.6",
          "CAUSAL_CONV1D_FORCE_BUILD": "TRUE", "MAMBA_FORCE_BUILD": "TRUE"})
    .pip_install("causal-conv1d>=1.4.0", extra_options="--no-build-isolation")
    .pip_install("mamba-ssm>=2.2.0", extra_options="--no-build-isolation")
    .add_local_dir("src", remote_path=SRC)
)


@app.function(image=image, volumes={VOL: vol}, gpu="A10G", timeout=4 * 60 * 60)
def extract_caduceus(limit: int | None = None, resume: bool = False):
    vol.reload()
    out_dir = f"{EMB}/caduceus"
    windows = f"{DATA}/windows.tsv"
    if limit is not None:
        import pandas as pd
        sp = f"{DATA}/windows_smoke_caduceus.tsv"
        pd.read_csv(windows, sep="\t").head(limit).to_csv(sp, sep="\t", index=False)
        windows, out_dir = sp, f"{EMB}/_smoke_caduceus"
    cmd = [sys.executable, f"{SRC}/extract_embeddings.py", "--model", "caduceus",
           "--windows", windows, "--out_dir", out_dir, "--device", "cuda"]
    if resume:
        cmd.append("--resume")
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root")
    vol.commit()
    return out_dir


@app.local_entrypoint()
def main(limit: int = 0, resume: bool = False):
    out = extract_caduceus.remote(limit or None, resume)
    print(f"caduceus done → {out}")
