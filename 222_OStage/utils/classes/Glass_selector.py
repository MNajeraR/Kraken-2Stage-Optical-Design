# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 17:49:03 2025

@author: MORGANRHAINAJERAROA
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Any, List, Optional, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches



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
        mx=1.0,
        my=0.01,
        figsize=(9,7),
        inset_size="40%",
        inset_loc="upper left",
        borderpad=1.2,
        title="Optical Glass Selection",
        legend_loc="lower right",
        inset_legend_loc="lower right",
        fmt_n="{:.2f}",
        fmt_vd="{:.2f}",
        s_ref=55,
    ):
    
    
        # ==========================================================
        # Colors (matching presentation)
        # ==========================================================
    
        BG      = (242/255,243/255,245/255)
    
        TEXT    = (24/255,24/255,24/255)
    
        AXIS    = (80/255,80/255,80/255)
    
        #GRID    = (200/255,200/255,200/255)
    
        ALLPTS  = (110/255,110/255,110/255)
    
        ORANGE  = (220/255,125/255,30/255)
    
        GREEN   = (0/255,120/255,70/255)
    
        def _limits_from(arr_x, arr_y, pad_x, pad_y):
            return (
                np.nanmin(arr_x)-pad_x,
                np.nanmax(arr_x)+pad_x,
                np.nanmin(arr_y)-pad_y,
                np.nanmax(arr_y)+pad_y,
            )
    
        vd_possible = self.vd_all_possible
        n_possible  = self.n_all_possible
    
        vd_all = self.vd_all
        n_all  = self.n_all
    
        if vd_possible.size:
            x_low,x_high,y_low,y_high = _limits_from(
                vd_possible,
                n_possible,
                mx,
                my,
            )
    
        elif vd_all.size:
            x_low,x_high,y_low,y_high = _limits_from(
                vd_all,
                n_all,
                mx,
                my,
            )
    
        else:
            raise ValueError("No glasses to plot.")
    
        fig,ax = plt.subplots(
            figsize=figsize,
            constrained_layout=True,
        )
    
        # ----------------------------------------------------------
        # Background
        # ----------------------------------------------------------
    
        fig.patch.set_facecolor(BG)
    
        ax.set_facecolor(BG)
    
        # ----------------------------------------------------------
        # Scatter
        # ----------------------------------------------------------
    
        ax.scatter(
            vd_all,
            n_all,
            color=ALLPTS,
            alpha=0.45,
            s=18,
            label="Catalog"
        )
    
        ax.scatter(
            vd_possible,
            n_possible,
            marker="*",
            color=ORANGE,
            s=65,
            label="Candidates",
            zorder=3
        )
    
        # ----------------------------------------------------------
        # Axis
        # ----------------------------------------------------------
    
        ax.set_xlabel(r"$V_d$",fontsize=20,color=TEXT)
    
        ax.set_ylabel(r"$n$",fontsize=20,color=TEXT)
    
        #ax.set_title(title,fontsize=17,color=TEXT)
    
        ax.tick_params(
            axis="both",
            labelsize=20,
            colors=TEXT,
            width=1.5,
        )
    
        # ax.grid(
        #     True,
        #     color=GRID,
        #     linewidth=0.8,
        #     alpha=0.8,
        # )
    
        for s in ax.spines.values():
            s.set_linewidth(1.6)
            s.set_color(AXIS)
    
        ax.invert_xaxis()
    
        ax.yaxis.set_ticks_position('right')
    
        ax.yaxis.set_label_position('right')
    
        # ----------------------------------------------------------
        # Zoom rectangle
        # ----------------------------------------------------------
    
        rect = patches.Rectangle(
            (x_high,y_low),
            (x_low-x_high),
            (y_high-y_low),
            linewidth=2.0,
            edgecolor=ORANGE,
            linestyle="--",
            facecolor="none",
        )
    
        ax.add_patch(rect)
    
        # ----------------------------------------------------------
        # Inset
        # ----------------------------------------------------------
    
        axins = inset_axes(
            ax,
            width=inset_size,
            height=inset_size,
            loc=inset_loc,
            borderpad=borderpad,
        )
    
        axins.set_facecolor(BG)
    
        axins.scatter(
            vd_all,
            n_all,
            color=ALLPTS,
            alpha=0.20,
            s=12,
        )
    
        axins.scatter(
            vd_possible,
            n_possible,
            marker="*",
            color=ORANGE,
            s=55,
        )
    
        axins.set_xlim(x_low,x_high)
    
        axins.set_ylim(y_low,y_high)
    
        axins.invert_xaxis()
    
        # axins.grid(
        #     True,
        #     color=GRID,
        #     alpha=0.6,
        #     linewidth=0.7,
        # )
    
        axins.tick_params(
            labelsize=12,
            colors=TEXT,
            width=1.2,
        )
    
        for s in axins.spines.values():
            s.set_linewidth(1.2)
            s.set_color(AXIS)
    
        axins.yaxis.set_ticks_position('right')
    
        axins.yaxis.set_label_position('right')
    
        # ----------------------------------------------------------
        # Reference glass
        # ----------------------------------------------------------
    
        x_ref = None
        y_ref = None
        label_ref = ""
    
        if self.ref_i is not None:
    
            for idx_arr,vx,vy,names in (
    
                (self.Glass_idx_possible,
                 vd_possible,
                 n_possible,
                 self.Names_Glass_possible),
    
                (self.Glass_idx_filtered,
                 self.vd_all_filtered,
                 self.n_all_filtered,
                 self.Names_Glass_filtered),
    
                (self.Glass_idx,
                 vd_all,
                 n_all,
                 self.Names_Glass),
            ):
    
                if idx_arr.size:
    
                    m = idx_arr == self.ref_i
    
                    if np.any(m):
    
                        x_ref = float(vx[m][0])
    
                        y_ref = float(vy[m][0])
    
                        label_ref = str(names[m][0])
    
                        break
    
        if x_ref is not None:
    
            ax.scatter(
                x_ref,
                y_ref,
                s=s_ref*2,
                color=GREEN,
                edgecolor=TEXT,
                linewidth=0.8,
                zorder=6,
                label="Reference",
            )
    
            axins.scatter(
                x_ref,
                y_ref,
                s=s_ref*2,
                color=GREEN,
                edgecolor=TEXT,
                linewidth=0.8,
                zorder=7,
                label=rf"{label_ref}: $n={fmt_n.format(y_ref)}$, $V_d={fmt_vd.format(x_ref)}$",
            )
    
            leg=axins.legend(
                loc=inset_legend_loc,
                fontsize=12,
                frameon=True,
            )
    
            leg.get_frame().set_facecolor(BG)
    
            leg.get_frame().set_edgecolor(AXIS)
    
        # ----------------------------------------------------------
        # Legend
        # ----------------------------------------------------------
    
        handles,labels=ax.get_legend_handles_labels()
    
        by_label=dict(zip(labels,handles))
    
        leg=ax.legend(
            by_label.values(),
            by_label.keys(),
            loc=legend_loc,
            fontsize=11,
            frameon=True,
        )
    
        leg.get_frame().set_facecolor(BG)
    
        leg.get_frame().set_edgecolor(AXIS)
    
        return fig,ax,axins