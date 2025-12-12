# train_framework/components/loader/load_npz_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Literal, Optional, Iterable, Set
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

Split = Literal["train", "validate"]
Task  = Literal["cls", "reg"]


def discover_split_parts(
    base_dir: str | Path,
    task: Task,          # "cls" or "reg"
    stem: str,           # exporter stem
    key: str,            # exporter key ("diff", etc.)
    split: Split,        # "train" or "validate"
) -> List[Path]:
    """
    Return all {base_dir}/{task}/{stem}/{key}/{split}.partNNNN.npz
    (or a single {split}.npz if parts are not used)
    """
    d = Path(base_dir) / task / stem / key
    if not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {d}")
    parts = sorted(d.glob(f"{split}.part*.npz"))
    if not parts:
        single = d / f"{split}.npz"
        if single.exists():
            parts = [single]
    if not parts:
        raise FileNotFoundError(f"No NPZ files for {task}/{stem}/{key}/{split}")
    return parts


class NPZPartsDataset(Dataset):
    """
    Lazy, memory-mapped dataset across many NPZ parts.
    For classification: expects keys {X, y_cls}
    For regression:     expects keys {X, y_reg}

    Optional symbol filtering: if select_symbols is provided, keeps only rows
    whose 'symbols' entry is in that set. Requires that NPZs contain 'symbols'.
    """

    def __init__(
        self,
        files: List[Path],
        task: Task,
        select_symbols: Optional[Iterable[str]] = None,
    ):
        self.task = task
        self.files = list(files)

        # self._z:  List[np.lib.npyio.NpzFile] = []
        self._X:  List[np.ndarray] = []
        self._y:  List[np.ndarray] = []
        self._symbols: List[np.ndarray] = []
        self._rows: List[np.ndarray] = []   # indices kept per file (after filtering)
        self._len: List[int] = []           # per-file kept length
        self._off: List[int] = []           # global offsets per file

        sym_filter: Optional[Set[str]] = set(select_symbols) if select_symbols else None

        total = 0
        for f in self.files:
            z = np.load(f, allow_pickle=False, mmap_mode="r")
            X = z["X"]
            # y_key = "y_cls" if task == "cls" else "y_reg"
            y_key = "y"
            if y_key not in z:
                raise KeyError(f"{f} missing '{y_key}'")
            y = z[y_key]
            n = int(X.shape[0])
            if int(y.shape[0]) != n:
                raise ValueError(f"Row mismatch in {f}: X {X.shape} vs y {y.shape}")

            # Resolve symbol filtering (if requested)
            if sym_filter is not None:
                if "symbols" not in z:
                    raise KeyError(f"{f} missing 'symbols' required for select_symbols filtering")
                syms = z["symbols"]
                # syms is typically 1D array of strings (np.str_ or object)
                # Use np.isin for vectorized filtering
                mask = np.isin(syms, list(sym_filter))
                rows = np.nonzero(mask)[0]
                kept = int(rows.shape[0])
                self._symbols.append(syms)   # keep ref (mmap)
            else:
                rows = np.arange(n, dtype=np.int64)
                kept = n
                self._symbols.append(None)   # type: ignore

            if kept == 0:
                # still keep bookkeeping so __len__ works; this file contributes 0 rows
                pass

            # self._z.append(z)
            self._X.append(X)
            self._y.append(y)
            self._rows.append(rows)
            self._len.append(kept)
            self._off.append(total)
            total += kept

        self._N = total

    def __len__(self) -> int:
        return self._N

    def _loc(self, idx: int) -> Tuple[int, int]:
        # binary search over prefix sums of kept lengths
        lo, hi = 0, len(self._off) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            off, ln = self._off[mid], self._len[mid]
            if off <= idx < off + ln:
                # map global idx -> per-file kept row index j
                return mid, idx - off
            if idx < off:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        fi, j = self._loc(idx)
        ridx = self._rows[fi][j]           # real row index inside file fi
        x = self._X[fi][ridx]
        y = self._y[fi][ridx]
        x_t = torch.from_numpy(np.asarray(x))
        if self.task == "cls":
            y_t = torch.as_tensor(y, dtype=torch.long)
        else:
            y_t = torch.as_tensor(y, dtype=torch.float32)
        return x_t, y_t


def build_loader_for_split(
    *,
    base_dir: str | Path,
    stem: str,
    key: str,
    task: Task,
    split: Split,                     # "train" | "validate"
    select_symbols: Optional[Iterable[str]] = None,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    files = discover_split_parts(base_dir, task, stem, key, split)
    ds = NPZPartsDataset(files, task=task, select_symbols=select_symbols)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
    )
    return loader


def build_loaders_from_export(
    *,
    base_dir: str | Path,
    stem: str,
    key: str,
    task: Task,                # "cls" or "reg"
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    select_symbols_train: Optional[Iterable[str]] = None,
    select_symbols_val:   Optional[Iterable[str]] = None,
):
    """
    Build (train_loader, val_loader) using ALL train/validate parts lazily.
    Optional symbol filtering per split.
    """
    train_files = discover_split_parts(base_dir, task, stem, key, "train")
    val_files   = discover_split_parts(base_dir, task, stem, key, "validate")

    train_ds = NPZPartsDataset(train_files, task=task, select_symbols=select_symbols_train)
    val_ds   = NPZPartsDataset(val_files,   task=task, select_symbols=select_symbols_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return train_loader, val_loader
