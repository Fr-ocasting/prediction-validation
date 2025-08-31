"""
quick_bench.py
--------------
Micro‑benchmark pour choisir rapidement :

* type de précision (FP32, TF32, AMP‑BF16)
* DataLoader : num_workers, pin_memory, persistent_workers, prefetch_factor
* accélération torch.compile / torch.jit ou aucune

Résultats enregistrés dans benchmark_results.csv
"""

import time, argparse, itertools, json, csv, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler


def make_dataset(n_samples: int, dim: int = 128):
    x = torch.randn(n_samples, dim)
    y = torch.randn(n_samples, dim)
    return TensorDataset(x, y)


class SimpleNet(torch.nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def prepare_model(precision: str, compile_mode: str) -> tuple[torch.nn.Module, bool]:
    """Return model ready for training and flag use_scaler."""
    device = "cuda"
    model = SimpleNet().to(device)

    # precision does not affect weights dtype (stay FP32) – autocast covers activations
    use_scaler = precision == "amp_fp16"

    # Optional compilation
    if compile_mode == "jit":
        model = torch.jit.script(model)
    elif compile_mode == "compile":
        if not hasattr(torch, "compile"):
            print("torch.compile non disponible sur cette version de PyTorch; ignoré.")
        else:
            model = torch.compile(model, mode="max-autotune")

    return model, use_scaler


def bench_one(precision: str,
              compile_mode: str,
              dataset_size: int,
              batch_size: int,
              num_workers: int,
              iters: int = 40):
    tf32 = precision in ("tf32", "amp_bf16", "amp_fp16")
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    amp = precision.startswith("amp")
    dtype = torch.bfloat16 if precision == "amp_bf16" else torch.float16

    ds = make_dataset(dataset_size)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )

    model, need_scaler = prepare_model(precision, compile_mode)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=need_scaler)

    device = "cuda"
    torch.cuda.synchronize()
    start = time.perf_counter()

    for i, (x, y) in zip(range(iters), dl):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        if amp:
            with autocast(dtype=dtype):
                out = model(x)
                loss = (out - y).pow(2).mean()
            if need_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        else:
            loss = (model(x) - y).pow(2).mean()
            loss.backward()
            opt.step()

    torch.cuda.synchronize()
    t = time.perf_counter() - start
    throughput = (iters * batch_size) / t
    return t, throughput


def main():
    results = []
    precisions = ["fp32", "tf32", "amp_bf16"]
    compile_modes = ["none", "jit", "compile"]
    configs = list(itertools.product(
        (2000, 128), (10000, 256), (0, 4), precisions, compile_modes
    ))

    # explicit loops for clarity
    for ds_size, bs in [(2000, 128), (10000, 256)]:
        for nw in (0, 4):
            for prec in precisions:
                for cmode in compile_modes:
                    t, th = bench_one(prec, cmode, ds_size, bs, nw)
                    results.append({
                        "dataset": ds_size,
                        "batch": bs,
                        "num_workers": nw,
                        "precision": prec,
                        "compile": cmode,
                        "elapsed_s": round(t, 3),
                        "throughput": round(th, 1),
                        "persistent": nw > 0,
                        "prefetch_factor": 4,
                    })
                    print(results[-1])

    out_path = Path("benchmark_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nRésultats écrits dans {out_path.resolve()}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("CUDA introuvable – exécute sur une machine avec GPU.")
    main()
