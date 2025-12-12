from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Sequence, Union, List
import numpy as np

try:
    from framework.component import Component, Context
except Exception:
    class Context(dict): ...
    class Component:
        def run(self, ctx: Context): return ctx


@dataclass
class NpzPredictLoader(Component):
    """
    Load features (X), labels (y), and ids from one or more NPZ files, then put
    them into ctx, and make sure the shapes and ctx['input_size'] match the
    training pipeline.

    Expects in ctx:
      - 'npz_paths': str or Sequence[str]
          Path or list of paths to .npz files.
        (predict_pipeline.py sets this for you.)

    Writes into ctx:
      - 'X':  np.ndarray, shape:
              * (N, F, 1) if original X was (N, F)
              * (N, L, F) if original X was (N, L, F)
      - 'y':  np.ndarray, shape (N,)
      - 'id': np.ndarray of ids, shape (N,)
      - 'seq_len': int
      - 'input_size': int  (must match training-time value)
    """

    npz_paths_key: str = "npz_paths"
    x_key: str = "X"
    y_key: str = "y"
    id_key: str = "id"

    def _normalize_paths(self, paths: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(paths, str):
            return [paths]
        return list(paths)

    def run(self, ctx: Context) -> Context:
        paths = ctx.get(self.npz_paths_key)
        if paths is None:
            raise KeyError(f"NpzPredictLoader requires ctx['{self.npz_paths_key}']")

        paths = self._normalize_paths(paths)

        all_X = []
        all_y = []
        all_ids = []

        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"NPZ file not found: {p}")

            z = np.load(p, allow_pickle=False)

            if "X" not in z:
                raise KeyError(f"File {p} has no 'X' array")

            # --- extract y (unified label field) ---
            if "y" in z.files:
                y = z["y"]
            else:
                # if you're in transition, you can optionally look for old names:
                if "y_cls" in z.files:
                    y = z["y_cls"]
                elif "y_reg" in z.files:
                    y = z["y_reg"]
                else:
                    raise KeyError(f"File {p} has no 'y' (or y_cls/y_reg) array")

            # Prefer 'ids', then 'id', then fall back to 'symbols' if needed
            if "ids" in z.files:
                ids = z["ids"]
            elif "id" in z.files:
                ids = z["id"]
            elif "symbols" in z.files:
                ids = z["symbols"]
            else:
                # Fallback: make numeric ids if nothing else is present
                n_rows = z["X"].shape[0]
                ids = np.arange(n_rows, dtype=np.int64)

            X = z["X"]

            # basic consistency checks
            X = np.asarray(X)
            y = np.asarray(y)
            ids = np.asarray(ids)

            if ids.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Length mismatch in file {p}: X has {X.shape[0]} rows, "
                    f"but ids has {ids.shape[0]}"
                )
            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Length mismatch in file {p}: X has {X.shape[0]} rows, "
                    f"but y has {y.shape[0]}"
                )

            all_X.append(X)
            all_y.append(y)
            all_ids.append(ids)

        if not all_X:
            raise ValueError("No data loaded from NPZ files")

        # Concatenate over the row dimension
        X_cat = np.concatenate(all_X, axis=0)
        y_cat = np.concatenate(all_y, axis=0)
        ids_cat = np.concatenate(all_ids, axis=0)

        # ---- Mirror training BuildLoadersFromExport logic ----
        # Training collate does:
        #   - if X is (B, F)      -> (B, F, 1), input_size=1
        #   - if X is (B, L, F)   -> (B, L, F), input_size=F
        X_cat = np.asarray(X_cat)

        if X_cat.ndim == 2:
            # Stored as (N, F), interpret as sequence length F, one feature
            N, F = X_cat.shape
            X_cat = X_cat.reshape(N, F, 1)
            ctx["seq_len"] = int(F)
            ctx["input_size"] = 1
            print(f"[NpzPredictLoader] reshaped X from (N, F) -> (N, F, 1); "
                  f"seq_len={ctx['seq_len']}, input_size=1")
        elif X_cat.ndim == 3:
            N, L, F = X_cat.shape
            if F == 1:
                # Sequence length L, single feature
                ctx["seq_len"] = int(L)
                ctx["input_size"] = 1
                print(f"[NpzPredictLoader] detected X shape (N, L, 1); "
                      f"seq_len={ctx['seq_len']}, input_size=1")
            else:
                # Sequence length L, F features
                ctx["seq_len"] = int(L)
                ctx["input_size"] = int(F)
                print(f"[NpzPredictLoader] detected X shape (N, L, F); "
                      f"seq_len={ctx['seq_len']}, input_size={ctx['input_size']}")
        else:
            raise ValueError(
                f"[NpzPredictLoader] Unexpected X_cat shape {X_cat.shape}; "
                f"expected 2D or 3D."
            )

        ctx[self.x_key] = X_cat
        ctx[self.y_key] = y_cat
        ctx[self.id_key] = ids_cat

        print(f"[NpzPredictLoader] Loaded X.shape={X_cat.shape}, y.shape={y_cat.shape}, "
              f"ids.shape={ids_cat.shape}")
        return ctx
