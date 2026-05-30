"""Evo 2 7B extraction — isolated Modal app (A100).

Evo 2 7B (StripedHyena 2) needs the native `evo2` package + Flash Attention; the
HF AutoModel path is broken. Flash-attn's CUDA extension must match torch's ABI,
so (as with Caduceus) we pin torch 2.6 + the cu12/torch2.6/cp312/cxx11abiFALSE
prebuilt flash-attn wheel, then `pip install evo2`. Runs on an A100-80GB.

    modal run src/modal_evo2.py --limit 10   # smoke
    modal run src/modal_evo2.py              # full
"""

import subprocess
import sys

import modal

app = modal.App("bmds273b-evo2")
vol = modal.Volume.from_name("bmds273b-reg-data", create_if_missing=True)
VOL = "/vol"
DATA = f"{VOL}/regulatory"
EMB = f"{DATA}/embeddings"
SRC = "/root/src"

_FLASH = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
          "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.6.0", "numpy", "pandas", "h5py", "ninja", "packaging")
    .pip_install(_FLASH)
    .pip_install("evo2")
    .add_local_dir("src", remote_path=SRC)
)


@app.function(image=image, volumes={VOL: vol}, gpu="A100-80GB", timeout=6 * 60 * 60)
def extract_evo2(limit: int | None = None):
    vol.reload()
    out_dir = f"{EMB}/evo2"
    windows = f"{DATA}/windows.tsv"
    if limit is not None:
        import pandas as pd
        sp = f"{DATA}/windows_smoke_evo2.tsv"
        pd.read_csv(windows, sep="\t").head(limit).to_csv(sp, sep="\t", index=False)
        windows, out_dir = sp, f"{EMB}/_smoke_evo2"
    cmd = [sys.executable, f"{SRC}/extract_embeddings_evo2.py",
           "--windows", windows, "--out_dir", out_dir, "--device", "cuda:0"]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root")
    vol.commit()
    return out_dir


@app.local_entrypoint()
def main(limit: int = 0):
    print(f"evo2 done → {extract_evo2.remote(limit or None)}")
