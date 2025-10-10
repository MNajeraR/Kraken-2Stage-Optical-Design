#!/usr/bin/env python3

# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Dict

import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches


""" Looking for if KrakenOS is installed, if not, it assumes that

an folder downloaded from github is run"""



required = {'KrakenOS'}

installed = {pkg.key for pkg in pkg_resources.working_set}

missing = required - installed



if missing:

    print("Not installed")

    import sys

    sys.path.append("../..")

import KrakenOS as Kos


@dataclass(frozen=True)

class _GlassVals:
    idx: int
    WT: np.ndarray
    PT: np.ndarray
    n: float
    vd: float

class Glass_Selector:
    """
    Selection and filtering of glasses in the (n, v_d) plane, with a window over
    transmission PT and WT.

    - Applies filters: WT range, minimum transmission threshold, and a window
      around a reference glass.
    - Keeps the last occurrence per name (previous duplicates are recorded).
    - Does not print in __init__; exposes information via attributes and methods.
    """

    def __init__(
        self,
        confi: Any,
        WR: Tuple[float, float],
        name_glass: Optional[str] = None,
        delta_n: float = 0.025,
        delta_vd: float = 5.0,
        pt_threshold: float = 0.7,
    ) -> None:
        """
        Parameters
        ----------
        confi.IT : iterable with tuples/lists, each item: (WT, PT, ...)
        confi.NAMES : names by index
        confi.NM : list; assume NM[i][2] = n, NM[i][3] = v_d
        WR : (low, high) accepted WT range
        name_glass : reference glass name (to center the (n, v_d) window)
        delta_n, delta_vd : half-widths of the window around the reference glass
        pt_threshold : minimum transmission threshold (fraction if <=1; % if >1)
        """
        self.low_r, self.high_r = WR
        self.pt_threshold = (pt_threshold / 100.0) if pt_threshold > 1 else float(pt_threshold)

        self.glass_dict: Dict[str, _GlassVals] = {}       # name -> _GlassVals
        self.duplicates_dict: Dict[str, List[dict]] = {}  # name -> history
        self.skipped_info: List[dict] = []                # [{idx, name, reason}, ...]
        self.ref_i: Optional[int] = None                  # reference glass index

        names_arr = np.asarray(getattr(confi, "NAMES", []), dtype=object)
        IT = getattr(confi, "IT", [])
        NM = getattr(confi, "NM", [])

        # --- locate reference glass (last occurrence) ---
        center_n = center_vd = None
        if name_glass is not None and names_arr.size:
            hits = np.where(names_arr == name_glass)[0]
            if hits.size == 0:
                self.skipped_info.append({"idx": -1, "name": str(name_glass),
                                          "reason": "reference glass not found"})
            else:
                self.ref_i = int(hits[-1])
                try:
                    rec = NM[self.ref_i]
                    center_n = float(rec[2])
                    center_vd = float(rec[3])
                except Exception:
                    self.skipped_info.append({"idx": int(self.ref_i), "name": str(name_glass),
                                              "reason": "invalid reference glass n/v_d"})
                    center_n = center_vd = None  # disable (n, v_d) filter

        # --- iterate configuration and build dictionaries ---
        for i, item in enumerate(IT):
            name = (names_arr[i] if i < names_arr.size else f"idx_{i}")

            if item is None or (hasattr(item, "__len__") and len(item) < 2):
                self.skipped_info.append({"idx": i, "name": name, "reason": "None or malformed"})
                continue

            WT = np.asarray(item[0], dtype=float).ravel()
            PT = np.asarray(item[1], dtype=float).ravel()
            WT_sel, PT_sel = self._filter_pair(WT, PT)

            # read n and v_d
            if i < len(NM) and len(NM[i]) > 3:
                try:
                    n_Glass = float(NM[i][2])
                    vd_Glass = float(NM[i][3])
                except Exception:
                    self.skipped_info.append({"idx": i, "name": name,
                                              "reason": "n/v_d not convertible to float"})
                    continue
            else:
                self.skipped_info.append({"idx": i, "name": name, "reason": "malformed n and v_d"})
                continue

            if WT_sel.size > 0:
                if name in self.glass_dict:
                    # save the previous one as duplicate
                    prev = self.glass_dict[name]
                    self.duplicates_dict.setdefault(name, []).append({
                        "idx": prev.idx, "WT": prev.WT, "PT": prev.PT, "n": prev.n, "v_d": prev.vd
                    })
                    self.skipped_info.append({"idx": prev.idx, "name": name, "reason": "duplicate (overwritten)"})
                self.glass_dict[name] = _GlassVals(i, WT_sel, PT_sel, n_Glass, vd_Glass)
            else:
                self.skipped_info.append({"idx": i, "name": name, "reason": "no values in range"})

        # --- convert dictionary to arrays/keys ---
        if self.glass_dict:
            vals = list(self.glass_dict.values())
            keys = list(self.glass_dict.keys())

            self.Glass_idx   = np.fromiter((v.idx for v in vals), dtype=int, count=len(vals))
            self.WT_all      = [v.WT for v in vals]  # variable lengths -> list
            self.PT_all      = np.array([v.PT for v in vals], dtype=object)
            self.n_all       = np.fromiter((v.n for v in vals), dtype=float, count=len(vals))
            self.vd_all      = np.fromiter((v.vd for v in vals), dtype=float, count=len(vals))
            self.Names_Glass = np.asarray(keys, dtype=object)
        else:
            # initialize consistent empties
            self.Glass_idx   = np.array([], dtype=int)
            self.WT_all      = []
            self.PT_all      = np.array([], dtype=object)
            self.n_all       = np.array([], dtype=float)
            self.vd_all      = np.array([], dtype=float)
            self.Names_Glass = np.array([], dtype=object)

        # --- sanity filter on n/vd ---
        if self.n_all.size:
            finite = np.isfinite(self.n_all) & np.isfinite(self.vd_all)
            final_mask = finite & (self.n_all >= 1.0) & (self.vd_all != 0)
            idx_keep = np.where(final_mask)[0]

            self.Glass_idx   = self.Glass_idx[final_mask]
            self.WT_all      = [self.WT_all[i] for i in idx_keep]
            self.PT_all      = self.PT_all[final_mask]
            self.n_all       = self.n_all[final_mask]
            self.vd_all      = self.vd_all[final_mask]
            self.Names_Glass = self.Names_Glass[final_mask]
        else:
            idx_keep = np.array([], dtype=int)

        # --- (n, v_d) window relative to reference glass ---
        if center_n is not None and center_vd is not None and self.n_all.size:
            mask_nv = (
                (self.n_all  >= center_n  - delta_n) & (self.n_all  <= center_n  + delta_n) &
                (self.vd_all >= center_vd - delta_vd) & (self.vd_all <= center_vd + delta_vd)
            )
        else:
            mask_nv = np.ones(self.n_all.shape, dtype=bool)

        self.Glass_idx_filtered   = self.Glass_idx[mask_nv]
        self.n_all_filtered       = self.n_all[mask_nv]
        self.vd_all_filtered      = self.vd_all[mask_nv]
        self.Names_Glass_filtered = self.Names_Glass[mask_nv]
        self.idx_filtered         = np.where(mask_nv)[0]

        # --- per-element min PT in that subset (NaN/empty-safe) ---
        if self.idx_filtered.size:
            PT_sub = self.PT_all[self.idx_filtered]
            self.mins_filtered = np.array([
                (np.nanmin(pt) if getattr(pt, "size", 0) > 0 else np.nan) for pt in PT_sub
            ], dtype=float)
        else:
            self.mins_filtered = np.array([], dtype=float)

        # --- transmission filter ---
        mask_PT_sub = (self.mins_filtered >= self.pt_threshold) & np.isfinite(self.mins_filtered)

        self.Glass_idx_possible   = self.Glass_idx_filtered[mask_PT_sub]
        self.n_all_possible       = self.n_all_filtered[mask_PT_sub]
        self.vd_all_possible      = self.vd_all_filtered[mask_PT_sub]
        self.Names_Glass_possible = self.Names_Glass_filtered[mask_PT_sub]

        # flag: did the reference glass survive into "possible"?
        self.reference_survives = (
            self.ref_i is not None and self.Glass_idx_possible.size and
            np.any(self.Glass_idx_possible == self.ref_i)
        )

    # -------- helper methods --------
    def _filter_pair(self, WT: np.ndarray, PT: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Trim WT/PT to the same length and filter by [self.low_r, self.high_r]."""
        m = min(WT.size, PT.size)
        if m == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        WT, PT = WT[:m], PT[:m]
        mask = (WT >= self.low_r) & (WT <= self.high_r)
        return WT[mask], PT[mask]

    def get_accepted(self, sort_by_idx: bool = True) -> List[Tuple[str, int, np.ndarray, np.ndarray, float, float]]:
        """Return [(name, idx, WT_sel, PT_sel, n, v_d), ...]."""
        items = [(name, v.idx, v.WT, v.PT, v.n, v.vd) for name, v in self.glass_dict.items()]
        if sort_by_idx:
            items.sort(key=lambda x: x[1])  # by idx
        return items

    def print_skipped(self) -> None:
        print("=== Skipped ===")
        for s in self.skipped_info:
            print(f"[{s['idx']}] {s['name']} → {s['reason']}")
        print("Total skipped:", len(self.skipped_info))

    def get_reference_info(self) -> Optional[dict]:
        """Return info of the reference glass if it remains in 'possible'."""
        if not self.reference_survives:
            return None
        m = (self.Glass_idx_possible == self.ref_i)
        return {
            "Index": int(self.Glass_idx_possible[m][0]),
            "Name": str(self.Names_Glass_possible[m][0]),
            "n": float(self.n_all_possible[m][0]),
            "v_d": float(self.vd_all_possible[m][0]),
        }

    # -------- plot --------
    def plot_nv_with_inset(
        self,
        mx: float = 1.0, my: float = 0.01,
        figsize: Tuple[float, float] = (9, 7),
        inset_size: str = "40%", inset_loc: str = "upper left", borderpad: float = 1.2,
        title: str = "Optical Glass Selection",
        legend_loc: str = "lower right",
        inset_legend_loc: str = "lower right",
        fmt_n: str = "{:.2f}", fmt_vd: str = "{:.2f}",
        s_ref: float = 30.0
    ):
        """
        Draw the n–v_d diagram with a zoomed inset.
        - Highlights the reference glass as a black dot (if applicable).
        - The inset legend shows name + n and v_d values.
        """

        def _limits_from(arr_x: np.ndarray, arr_y: np.ndarray, pad_x: float, pad_y: float):
            return (np.nanmin(arr_x) - pad_x, np.nanmax(arr_x) + pad_x,
                    np.nanmin(arr_y) - pad_y, np.nanmax(arr_y) + pad_y)

        vd_possible, n_possible = self.vd_all_possible, self.n_all_possible
        vd_all, n_all = self.vd_all, self.n_all

        if vd_possible.size:
            x_low, x_high, y_low, y_high = _limits_from(vd_possible, n_possible, mx, my)
        elif vd_all.size:
            x_low, x_high, y_low, y_high = _limits_from(vd_all, n_all, mx, my)
        else:
            raise ValueError("No data to plot (vd_all/vd_all_possible are empty).")

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.scatter(vd_all, n_all, marker='.', alpha=0.35, label='All')
        ax.scatter(vd_possible, n_possible, marker='*', label='Possible')

        ax.set_xlabel(r"$v_d$", fontsize=16)
        ax.set_ylabel(r"$n$", fontsize=16)
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)

        # invert X and put n ticks/label on the right
        ax.invert_xaxis()
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')

        # zoom rectangle (note: X axis is inverted)
        rect = patches.Rectangle((x_high, y_low), (x_low - x_high), (y_high - y_low),
                                 linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)

        # inset
        axins = inset_axes(ax, width=inset_size, height=inset_size, loc=inset_loc, borderpad=borderpad)
        axins.scatter(vd_all, n_all, marker='.', alpha=0.15)
        axins.scatter(vd_possible, n_possible, marker='*')

        axins.set_xlim(x_low, x_high)
        axins.set_ylim(y_low, y_high)
        axins.invert_xaxis()
        axins.grid(True, alpha=0.2)
        axins.tick_params(labelsize=8)
        axins.yaxis.set_ticks_position('right')
        axins.yaxis.set_label_position('right')

        # find reference (search first in 'possible')
        x_ref = y_ref = None
        label_ref = ""
        if self.ref_i is not None:
            for idx_arr, vx, vy, names in (
                (self.Glass_idx_possible, vd_possible, n_possible, self.Names_Glass_possible),
                (self.Glass_idx_filtered, self.vd_all_filtered, self.n_all_filtered, self.Names_Glass_filtered),
                (self.Glass_idx, vd_all, n_all, self.Names_Glass),
            ):
                if idx_arr.size:
                    m = (idx_arr == self.ref_i)
                    if np.any(m):
                        x_ref = float(vx[m][0])
                        y_ref = float(vy[m][0])
                        label_ref = str(names[m][0])
                        break

        if x_ref is not None and y_ref is not None:
            ax.scatter(x_ref, y_ref, s=s_ref, marker='o', zorder=6, label=label_ref)
            inset_label = rf"{label_ref}: $n={fmt_n.format(y_ref)}$, $v_d={fmt_vd.format(x_ref)}$"
            axins.scatter(x_ref, y_ref, s=s_ref, marker='o', zorder=7, label=inset_label)
            axins.legend(loc=inset_legend_loc, fontsize=8, frameon=True, borderpad=0.4,
                         handlelength=1.0, handletextpad=0.5)

        # de-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc=legend_loc, fontsize=12)

        return fig, ax, axins
    
def pick_name_universe(selector, prefer: str = "possible"):
    """
    Return (names_array, label) choosing where to search for the center:
    - 'possible' if it exists and is non-empty
    - otherwise, 'filtered'
    - otherwise, 'all'
    """
    if prefer == "possible" and getattr(selector, "Names_Glass_possible", None) is not None:
        if selector.Names_Glass_possible.size:
            return selector.Names_Glass_possible, "possible"
    if getattr(selector, "Names_Glass_filtered", None) is not None and selector.Names_Glass_filtered.size:
        return selector.Names_Glass_filtered, "filtered"
    return selector.Names_Glass, "all"
    
def _find_center_index(arr: np.ndarray, center: Any) -> int:
    """Robust: exact → case-insensitive exact → case-insensitive substring."""
    a = np.asarray(arr)
    if a.size == 0:
        raise ValueError("Empty array; cannot find center.")
    # direct equality
    idx = np.where(a == center)[0]
    if idx.size:
        return int(idx[0])
    # strings: case-insensitive and substring
    if a.dtype == object and isinstance(center, str):
        A = np.array([str(x) for x in a], dtype=object)
        c = center.casefold()
        idx = np.where(np.array([x.casefold() == c for x in A]))[0]
        if idx.size:
            return int(idx[0])
        idx = np.where(np.array([c in x.casefold() for x in A]))[0]
        if idx.size:
            return int(idx[0])
    raise ValueError(f"Center value {center!r} not found in provided array.")   
        
def centered_windows_safe(
    lists: Sequence[Sequence[Any]],
    centers: Sequence[Any],
    target_len: Optional[int] = None,
    return_center_indices: bool = False,
    strict: bool = False,
) -> Tuple[List[np.ndarray], Optional[List[int]]]:
    """
    Centered windows with a common length:
      - If target_len > feasible, it is reduced to the feasible length (unless strict=True).
      - Robust center lookup.
    """
    if len(lists) != len(centers):
        raise ValueError("`lists` and `centers` must have the same length.")
    arrays = [np.array(seq, dtype=object) for seq in lists]
    if any(len(a) == 0 for a in arrays):
        raise ValueError("All input sequences must be non-empty.")

    center_indices = [_find_center_index(a, c) for a, c in zip(arrays, centers)]

    # per-list maximum symmetric window length
    max_possible = []
    for a, c in zip(arrays, center_indices):
        left  = c + 1
        right = len(a) - c
        max_possible.append(2 * min(left, right) - 1)

    feasible_len = min(max_possible)
    if target_len is None:
        L = feasible_len
    else:
        if target_len > feasible_len:
            if strict:
                raise ValueError(f"target_len={target_len} is too large; maximum feasible: {feasible_len}")
            L = feasible_len
        elif target_len < 1:
            raise ValueError("target_len must be >= 1")
        else:
            L = target_len

    def crop(arr: np.ndarray, c: int, L: int) -> np.ndarray:
        left = (L - 1) // 2
        start = max(0, c - left)
        end = start + L
        if end > len(arr):
            end = len(arr)
            start = end - L
        return arr[start:end]

    cropped = [crop(a, c, L) for a, c in zip(arrays, center_indices)]
    return (cropped, center_indices) if return_center_indices else (cropped, None)    
        
# ---------- AUTOMATOR FOR n GLASSES ----------

def select_and_center_many(
     confi: Any,
     reference_names: Sequence[str],
     WR: Tuple[float, float] = (0.34, 1.10),
     delta_n: float = 0.025,
     delta_vd: float = 5.0,
     pt_threshold: float = 0.8,
     prefer_universe: str = "possible",
     target_len: Optional[int] = 15,
     plot_each: bool = False,
 ) -> Dict[str, Any]:
     """
     Build a Glass_Selector for each reference name in `reference_names`,
     choose the appropriate universe to center on (possible→filtered→all),
     and return centered windows with a common length.
 
     Returns a dict with:
       - 'selectors': {name: Glass_Selector}
       - 'universes': {name: ('possible'|'filtered'|'all')}
       - 'name_arrays': [np.ndarray, ...] in the order of reference_names
       - 'cropped_names': [np.ndarray, ...] centered windows
       - 'center_indices': [int, ...] center indices in each base array
       - 'feasible_len': int effective length used
     """
     selectors = {}
     universes = {}
     name_arrays: List[np.ndarray] = []
 
     # Build selectors
     for ref in reference_names:
         sel = Glass_Selector(
             confi, WR=WR, name_glass=ref,
             delta_n=delta_n, delta_vd=delta_vd, pt_threshold=pt_threshold
         )
         selectors[ref] = sel
         names_arr, where = pick_name_universe(sel, prefer=prefer_universe)
         universes[ref] = where
         name_arrays.append(names_arr)
 
         if plot_each:
             print(f"\n[{ref}] Universe: {where} | nv candidates: {sel.Glass_idx_filtered.size} | Possible: {sel.Glass_idx_possible.size}")
             fig, ax, axins = sel.plot_nv_with_inset(title=f"n-vd around '{ref}'")
             plt.show()
 
     # Center windows by reference name
     cropped, center_indices = centered_windows_safe(
         name_arrays,
         centers=list(reference_names),
         target_len=target_len,
         return_center_indices=True,
         strict=False
     )
 
     # Effective length used
     if len(cropped):
         feasible_len = len(cropped[0])
     else:
         feasible_len = 0
 
     return {
         "selectors": selectors,
         "universes": universes,
         "name_arrays": name_arrays,
         "cropped_names": cropped,
         "center_indices": center_indices,
         "feasible_len": feasible_len,
     }   
            
        
# _________________________________________#

P_Obj = Kos.surf()

P_Obj.Rc = 0.0

P_Obj.Thickness = 10

P_Obj.Glass = "AIR"

P_Obj.Diameter = 30.0


# _________________________________________#



L1a = Kos.surf()

L1a.Rc = 9.284706570002484E+001

L1a.Thickness = 6.0

L1a.Glass = "BK7"

L1a.Diameter = 30.0

L1a.Axicon = 0

L1a.Color = [.8, .7, .4]



# _________________________________________#



L1b = Kos.surf()

L1b.Rc = -3.071608670000159E+001

L1b.Thickness = 3.0

L1b.Glass = "F2"

L1b.Diameter = 30

L1b.Color = [.7, .4, .4]



# _________________________________________#



L1c = Kos.surf()

L1c.Rc = -7.819730726078505E+001

L1c.Thickness = 9.737604742910693E+001

L1c.Glass = "AIR"

L1c.Diameter = 30



# _________________________________________#



P_Ima = Kos.surf()

P_Ima.Rc = 0.0

P_Ima.Thickness = 0.0

P_Ima.Glass = "AIR"

P_Ima.Diameter = 100.0

P_Ima.Name = "Plano imagen"



# _________________________________________#



A = [P_Obj, L1a, L1b, L1c, P_Ima]

configuracion_1 = Kos.Setup()

a = len(configuracion_1.NAMES)

                
# Lista de vidrios de referencia 
refs = ["S-FPL51", "F8", "BK7", "S-FPL53"]

result = select_and_center_many(
    confi = configuracion_1,
    reference_names = refs,
    WR = (0.34, 1.10),
    delta_n = 0.025,
    delta_vd = 5.0,
    pt_threshold = 0.8,
    prefer_universe = "possible",   # si no está en possible, cae a filtered → all
    target_len = 15,
    plot_each = True                # True para una figura por vidrio
)

# Resumen
print("\nConjunto elegidos por vidrio:")
for ref, where in result["Conjunto"].items():
    print(f"  {ref}: {where}")

print("\nÍndices de centro (por arreglo base al conjunto elegido):")
for ref, idx in zip(refs, result["center_indices"]):
    print(f"  {ref}: {idx}")

print(f"\nLongitud efectiva de los arreglos: {result['feasible_len']}")

print("\nListas centradas (por vidrio):")
for ref, win in zip(refs, result["cropped_names"]):
    print(f"  {ref}: {list(win)}")

























