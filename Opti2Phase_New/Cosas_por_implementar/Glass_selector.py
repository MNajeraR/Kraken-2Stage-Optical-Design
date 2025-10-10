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
    Selección y filtrado de vidrios en el plano (n, v_d), con ventana sobre transmisión PT y WT.
    - Aplica filtros: rango de WT, umbral de transmisión mínima, y ventana alrededor de un vidrio de referencia.
    - Conserva el último ejemplar por nombre (duplicados anteriores quedan registrados).
    - No imprime en __init__; expone información vía atributos y métodos.
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
        confi.IT : iterable con tuplas/listas, cada item: (WT, PT, ...)
        confi.NAMES : nombres por índice
        confi.NM : lista; se asume NM[i][2] = n, NM[i][3] = v_d
        WR : (bajo, alto) rango aceptado para WT
        name_glass : nombre del vidrio de referencia (para centrar ventana en n, v_d)
        delta_n, delta_vd : ventanas alrededor del vidrio de referencia
        pt_threshold : umbral mínimo de transmisión (fracción si <=1; % si >1)
        """
        self.low_r, self.high_r = WR
        self.pt_threshold = (pt_threshold / 100.0) if pt_threshold > 1 else float(pt_threshold)

        self.glass_dict: Dict[str, _GlassVals] = {}       # name -> _GlassVals
        self.duplicates_dict: Dict[str, List[dict]] = {}  # name -> historial
        self.skipped_info: List[dict] = []                # [{idx, name, reason}, ...]
        self.ref_i: Optional[int] = None                  # índice del vidrio de referencia

        names_arr = np.asarray(getattr(confi, "NAMES", []), dtype=object)
        IT = getattr(confi, "IT", [])
        NM = getattr(confi, "NM", [])

        # --- localizar vidrio de referencia (última ocurrencia) ---
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
                    center_n = center_vd = None  # deshabilita filtro (n, v_d)

        # --- recorrer configuración y construir diccionarios ---
        for i, item in enumerate(IT):
            name = (names_arr[i] if i < names_arr.size else f"idx_{i}")

            if item is None or (hasattr(item, "__len__") and len(item) < 2):
                self.skipped_info.append({"idx": i, "name": name, "reason": "None or malformed"})
                continue

            WT = np.asarray(item[0], dtype=float).ravel()
            PT = np.asarray(item[1], dtype=float).ravel()
            WT_sel, PT_sel = self._filter_pair(WT, PT)

            # leer n y v_d
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
                    # guardar el previo como duplicado
                    prev = self.glass_dict[name]
                    self.duplicates_dict.setdefault(name, []).append({
                        "idx": prev.idx, "WT": prev.WT, "PT": prev.PT, "n": prev.n, "v_d": prev.vd
                    })
                    self.skipped_info.append({"idx": prev.idx, "name": name, "reason": "duplicate (overwritten)"})
                self.glass_dict[name] = _GlassVals(i, WT_sel, PT_sel, n_Glass, vd_Glass)
            else:
                self.skipped_info.append({"idx": i, "name": name, "reason": "no values in range"})

        # --- convertir diccionario a arreglos/llaves ---
        if self.glass_dict:
            vals = list(self.glass_dict.values())
            keys = list(self.glass_dict.keys())

            self.Glass_idx   = np.fromiter((v.idx for v in vals), dtype=int, count=len(vals))
            self.WT_all      = [v.WT for v in vals]  # longitudes variables -> lista
            self.PT_all      = np.array([v.PT for v in vals], dtype=object)
            self.n_all       = np.fromiter((v.n for v in vals), dtype=float, count=len(vals))
            self.vd_all      = np.fromiter((v.vd for v in vals), dtype=float, count=len(vals))
            self.Names_Glass = np.asarray(keys, dtype=object)
        else:
            # inicializar vacíos coherentes
            self.Glass_idx   = np.array([], dtype=int)
            self.WT_all      = []
            self.PT_all      = np.array([], dtype=object)
            self.n_all       = np.array([], dtype=float)
            self.vd_all      = np.array([], dtype=float)
            self.Names_Glass = np.array([], dtype=object)

        # --- filtro de sanidad en n/vd ---
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

        # --- filtro por ventana (n, v_d) relativo al vidrio de referencia ---
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

        # --- PT mínimo por elemento del subconjunto (NaN/empty-safe) ---
        if self.idx_filtered.size:
            PT_sub = self.PT_all[self.idx_filtered]
            self.mins_filtered = np.array([
                (np.nanmin(pt) if getattr(pt, "size", 0) > 0 else np.nan) for pt in PT_sub
            ], dtype=float)
        else:
            self.mins_filtered = np.array([], dtype=float)

        # --- filtro de transmisión ---
        mask_PT_sub = (self.mins_filtered >= self.pt_threshold) & np.isfinite(self.mins_filtered)

        self.Glass_idx_possible   = self.Glass_idx_filtered[mask_PT_sub]
        self.n_all_possible       = self.n_all_filtered[mask_PT_sub]
        self.vd_all_possible      = self.vd_all_filtered[mask_PT_sub]
        self.Names_Glass_possible = self.Names_Glass_filtered[mask_PT_sub]

        # bandera: ¿sobrevivió el vidrio de referencia?
        self.reference_survives = (
            self.ref_i is not None and self.Glass_idx_possible.size and
            np.any(self.Glass_idx_possible == self.ref_i)
        )

    # -------- helper methods --------
    def _filter_pair(self, WT: np.ndarray, PT: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Recorta WT/PT al mismo tamaño y filtra por [self.low_r, self.high_r]."""
        m = min(WT.size, PT.size)
        if m == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        WT, PT = WT[:m], PT[:m]
        mask = (WT >= self.low_r) & (WT <= self.high_r)
        return WT[mask], PT[mask]

    def get_accepted(self, sort_by_idx: bool = True) -> List[Tuple[str, int, np.ndarray, np.ndarray, float, float]]:
        """Devuelve [(name, idx, WT_sel, PT_sel, n, v_d), ...]."""
        items = [(name, v.idx, v.WT, v.PT, v.n, v.vd) for name, v in self.glass_dict.items()]
        if sort_by_idx:
            items.sort(key=lambda x: x[1])  # por idx
        return items

    def print_skipped(self) -> None:
        print("=== Skipped ===")
        for s in self.skipped_info:
            print(f"[{s['idx']}] {s['name']} → {s['reason']}")
        print("Total skipped:", len(self.skipped_info))

    def get_reference_info(self) -> Optional[dict]:
        """Regresa info del vidrio de referencia si permanece en 'possible'."""
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
        Dibuja el diagrama n–v_d con zoom (inset).
        - Marca el vidrio de referencia como punto negro (si aplica).
        - La leyenda del inset muestra nombre + valores n y v_d.
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
            raise ValueError("No hay datos para graficar (vd_all/vd_all_possible vacíos).")

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.scatter(vd_all, n_all, marker='.', alpha=0.35, label='All')
        ax.scatter(vd_possible, n_possible, marker='*', label='Possible')

        ax.set_xlabel(r"$v_d$", fontsize=16)
        ax.set_ylabel(r"$n$", fontsize=16)
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)

        # invertir X y ticks de n a la derecha
        ax.invert_xaxis()
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')

        # rectángulo del zoom (ojo: eje X invertido)
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

        # localizar referencia (busca primero en 'possible')
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

        # leyenda sin duplicados
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc=legend_loc, fontsize=12)

        return fig, ax, axins
    
    
    
#     def __init__(self, confi, WR, name_glass=None, delta_n=0.025, delta_vd=5.0, pt_threshold=0.7):
#         """
#         confi.IT: iterable; each item must contain (WT, PT, ...)
#         confi.NAMES: list/array of names by index
#         confi.NM: list with data; assume NM[i][2] = n, NM[i][3] = v_d
#         WR: (low_r, high_r) accepted range for WT
#         name_glass: (optional) reference glass name to center (n, v_d) filter
#         delta_n, delta_vd: windows around the reference glass
#         pt_threshold: minimum transmission threshold (0.7 if fraction; 70 if %)
#         """
#         self.low_r, self.high_r = WR

#         # Normalize pt_threshold to fraction
#         self.pt_threshold = pt_threshold / 100.0 if pt_threshold > 1 else float(pt_threshold)

#         self.glass_dict = {}       # name -> (idx, WT_sel, PT_sel, n, v_d)
#         self.duplicates_dict = {}  # name -> [ {idx, WT, PT, n, v_d}, ... ]
#         self.skipped_info = []     # [{idx, name, reason}, ...]
#         self.ref_i = None          # reference index if name_glass found  # <-- FIX

#         # --- find last occurrence of reference glass (if provided) ---
#         center_n = center_vd = None
#         if name_glass is not None:
#             names_arr = np.array(confi.NAMES)
#             hits = np.where(names_arr == name_glass)[0]
            
#             if hits.size == 0:
#                 self.skipped_info.append({"idx": -1, "name": str(name_glass),
#                                           "reason": "reference glass not found"})
#             else:
#                 self.ref_i = int(hits[-1])
#                 try:
#                     center_n  = float(confi.NM[self.ref_i][2])
#                     center_vd = float(confi.NM[self.ref_i][3])
#                 except Exception:
#                     self.skipped_info.append({"idx": int(self.ref_i), "name": str(name_glass),
#                                               "reason": "invalid reference glass n/v_d"})
#                     center_n = center_vd = None  # disable (n, v_d) filter

#         # --- iterate config and build dicts ---
#         for i, item in enumerate(confi.IT):
#             name = confi.NAMES[i] if i < len(confi.NAMES) else f"idx_{i}"

#             if item is None or len(item) < 2:
#                 self.skipped_info.append({"idx": i, "name": name, "reason": "None or malformed"})
#                 continue

#             WT = np.asarray(item[0], dtype=float).ravel()
#             PT = np.asarray(item[1], dtype=float).ravel()

#             WT_sel, PT_sel = self._filter_pair(WT, PT)

#             # Read n and v_d
#             if i < len(confi.NM) and len(confi.NM[i]) > 3:
#                 try:
#                     n_Glass  = float(np.asarray(confi.NM[i][2], dtype=float))
#                     vd_Glass = float(np.asarray(confi.NM[i][3], dtype=float))
#                 except Exception:
#                     self.skipped_info.append({"idx": i, "name": name, "reason": "n/v_d not convertible to float"})
#                     continue
#             else:
#                 self.skipped_info.append({"idx": i, "name": name, "reason": "malformed n and v_d"})
#                 continue

#             if WT_sel.size > 0:
#                 if name in self.glass_dict:
#                     prev_idx, prev_WT, prev_PT, prev_n, prev_vd = self.glass_dict[name]
#                     self.duplicates_dict.setdefault(name, []).append({
#                         "idx": prev_idx, "WT": prev_WT, "PT": prev_PT,
#                         "n": prev_n, "v_d": prev_vd
#                     })
#                     self.skipped_info.append({"idx": prev_idx, "name": name, "reason": "duplicate (overwritten)"})
#                 self.glass_dict[name] = (i, WT_sel, PT_sel, n_Glass, vd_Glass)
#             else:
#                 self.skipped_info.append({"idx": i, "name": name, "reason": "no values in range"})
                
             

#         # --- lists (accepted without duplicates) ---
#         self.Glass_idx   = np.array([v[0] for v in self.glass_dict.values()])
#         self.WT_all      = [v[1] for v in self.glass_dict.values()]
#         self.PT_all      = np.array([v[2] for v in self.glass_dict.values()], dtype=object)
#         self.n_all       = np.array([v[3] for v in self.glass_dict.values()])
#         self.vd_all      = np.array([v[4] for v in self.glass_dict.values()])
#         self.Names_Glass = np.array(list(self.glass_dict.keys()))

#         # --- sanity filter for bogus n/vd entries (remove non-finite; keep n>=1.0 and vd!=0) ---
#         finite = np.isfinite(self.n_all) & np.isfinite(self.vd_all)
#         final_mask = finite & (self.n_all >= 1.0) & (self.vd_all != 0) 
        
#         # Apply mask consistently
#         idx_keep = np.where(final_mask)[0]
        
#         self.Glass_idx   = self.Glass_idx[final_mask]
#         # WT_all is a list -> use comprehension or convert to object array
#         self.WT_all      = [self.WT_all[i] for i in idx_keep]
#         # PT_all is np.array(dtype=object); boolean mask works
#         self.PT_all      = self.PT_all[final_mask]
#         self.n_all       = self.n_all[final_mask]
#         self.vd_all      = self.vd_all[final_mask]
#         self.Names_Glass = self.Names_Glass[final_mask]

#         # --- (n, v_d) filter if reference glass is valid ---
#         if (center_n is not None) and (center_vd is not None):
#             mask_nv = (
#                 (self.n_all  >= center_n  - delta_n) & (self.n_all  <= center_n  + delta_n) &
#                 (self.vd_all >= center_vd - delta_vd) & (self.vd_all <= center_vd + delta_vd)
#             )
#         else:
#             mask_nv = np.ones_like(self.n_all, dtype=bool)

#         self.Glass_idx_filtered   = self.Glass_idx[mask_nv]
#         self.n_all_filtered       = self.n_all[mask_nv]
#         self.vd_all_filtered      = self.vd_all[mask_nv]
#         self.Names_Glass_filtered = self.Names_Glass[mask_nv]

#         # indices of the (n, v_d) subset in original universe
#         self.idx_filtered = np.where(mask_nv)[0]

#         # min PT per element in that subset (NaN/empty-safe)
#         self.mins_filtered = np.array([
#             (np.nanmin(self.PT_all[i]) if getattr(self.PT_all[i], "size", 0) > 0 else np.nan)
#             for i in self.idx_filtered
#         ])

#         # PT filter (works with fraction)
#         mask_PT_sub = (self.mins_filtered >= self.pt_threshold) & ~np.isnan(self.mins_filtered)

#         # apply to the subset
#         self.Glass_idx_possible   = self.Glass_idx_filtered[mask_PT_sub]
#         self.n_all_possible       = self.n_all_filtered[mask_PT_sub]
#         self.vd_all_possible      = self.vd_all_filtered[mask_PT_sub]
#         self.Names_Glass_possible = self.Names_Glass_filtered[mask_PT_sub]

#         # --- print reference info if it survived into "possible" ---
#         if self.ref_i is not None and self.Glass_idx_possible.size:
#             mask_glass_sele = (self.Glass_idx_possible == self.ref_i)   # <-- FIX (self.)
#             if np.any(mask_glass_sele):
#                 print('')
#                 print("Index:", self.Glass_idx_possible[mask_glass_sele][0])
#                 print("Name:", self.Names_Glass_possible[mask_glass_sele][0])
#                 print("n:", self.n_all_possible[mask_glass_sele][0])
#                 print("v_d:", self.vd_all_possible[mask_glass_sele][0])
                
                
#             else:
#                 print("The reference glass did NOT remain in the 'possible' subset.")

#     # -------- helper methods -------- 
#     def _filter_pair(self, WT, PT):
#         """Trim WT/PT to the same size and filter by [self.low_r, self.high_r]."""
#         m = min(WT.size, PT.size)
#         if m == 0:
#             return np.array([], dtype=float), np.array([], dtype=float)
#         WT, PT = WT[:m], PT[:m]
#         mask = (WT >= self.low_r) & (WT <= self.high_r)
#         return WT[mask], PT[mask]

#     def get_accepted(self, sort_by_idx=True):
#         """Return a list of (name, idx, WT_sel, PT_sel, n, v_d)."""
#         items = [(name, *vals) for name, vals in self.glass_dict.items()]
#         if sort_by_idx:
#             items.sort(key=lambda x: x[1])  # by idx
#         return items

#     def print_skipped(self):
#         print("=== Skipped ===")
#         for s in self.skipped_info:
#             print(f"[{s['idx']}] {s['name']} → {s['reason']}")
#         print("Total skipped:", len(self.skipped_info))
        
        
#     def plot_nv_with_inset(self,
#                        mx=1.0, my=0.01,
#                        figsize=(9, 7),
#                        inset_size="40%", inset_loc="upper left", borderpad=1.2,
#                        title="Optical Glass Selection",
#                        legend_loc="lower right",
#                        inset_legend_loc="lower right",
#                        fmt_n="{:.2f}", fmt_vd="{:.2f}",
#                        s_ref=30):
#             """
#             Draw the n–v_d diagram with a zoom inset.
#             Highlights the reference glass (self.ref_i) as a black dot and:
#               - Adds it to the main plot legend.
#               - Shows in the inset a legend with the selected glass n and v_d.

#             Extra parameters:
#               inset_legend_loc : legend location within the inset.
#               fmt_n, fmt_vd     : numeric format for n and v_d.
#             """
        
#             # --- Zoom limits (with fallback if 'possible' is empty) ---
#             def _limits_from(arr_x, arr_y, pad_x, pad_y):
#                 return (np.nanmin(arr_x) - pad_x, np.nanmax(arr_x) + pad_x,
#                         np.nanmin(arr_y) - pad_y, np.nanmax(arr_y) + pad_y)
        
#             if getattr(self, "vd_all_possible", None) is not None and self.vd_all_possible.size:
#                 x_low, x_high, y_low, y_high = _limits_from(self.vd_all_possible, self.n_all_possible, mx, my)
#             elif getattr(self, "vd_all", None) is not None and self.vd_all.size:
#                 x_low, x_high, y_low, y_high = _limits_from(self.vd_all, self.n_all, mx, my)
#             else:
#                 raise ValueError("No data to plot (vd_all/vd_all_possible are empty).")
        
#             # --- Main figure ---
#             fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
#             ax.scatter(self.vd_all, self.n_all, marker='.', color='blue', alpha=0.35, label='All')
#             ax.scatter(self.vd_all_possible, self.n_all_possible, marker='*', color='red', label='Possible')
        
#             ax.set_xlabel(r"$v_d$", fontsize=16)
#             ax.set_ylabel(r"$n$", fontsize=16)
#             ax.set_title(title, fontsize=16)
#             ax.tick_params(axis='both', labelsize=12)
#             ax.grid(True, alpha=0.3)
        
#             # Invert X and move ticks/label to the right
#             ax.invert_xaxis()
#             ax.yaxis.set_ticks_position('right')
#             ax.yaxis.set_label_position('right')
        
#             # --- Zoom rectangle (note: main X axis is inverted) ---
#             rect = patches.Rectangle((x_high, y_low), (x_low - x_high), (y_high - y_low),
#                                      linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--')
#             ax.add_patch(rect)
        
#             # --- Inset ---
#             axins = inset_axes(ax, width=inset_size, height=inset_size, loc=inset_loc, borderpad=borderpad)
#             axins.scatter(self.vd_all, self.n_all, marker='.', color='blue', alpha=0.15)
#             axins.scatter(self.vd_all_possible, self.n_all_possible, marker='*', color='red')
        
#             axins.set_xlim(x_low, x_high)
#             axins.set_ylim(y_low, y_high)
#             axins.invert_xaxis()
#             axins.grid(True, alpha=0.2)
#             axins.tick_params(labelsize=8)
#             axins.yaxis.set_ticks_position('right')
#             axins.yaxis.set_label_position('right')
        
#             # --- Highlight the reference glass ---
#             x_ref = y_ref = label_ref = None
#             if getattr(self, "ref_i", None) is not None:
#                 # Search in possible → filtered → all
#                 search_sets = [
#                     (getattr(self, "Glass_idx_possible", np.array([])),
#                      getattr(self, "vd_all_possible",  np.array([])),
#                      getattr(self, "n_all_possible",   np.array([])),
#                      getattr(self, "Names_Glass_possible", np.array([]))),
#                 (getattr(self, "Glass_idx_filtered", np.array([])),
#                      getattr(self, "vd_all_filtered",    np.array([])),
#                      getattr(self, "n_all_filtered",     np.array([])),
#                      getattr(self, "Names_Glass_filtered", np.array([]))),
#                     (getattr(self, "Glass_idx", np.array([])),
#                      getattr(self, "vd_all",    np.array([])),
#                      getattr(self, "n_all",     np.array([])),
#                      getattr(self, "Names_Glass", np.array([]))),
#                 ]
#                 for idx_arr, vd_arr, n_arr, name_arr in search_sets:
#                     if getattr(idx_arr, "size", 0):
#                         m = (idx_arr == self.ref_i)
#                         if np.any(m):
#                             x_ref = float(vd_arr[m][0])  # v_d
#                             y_ref = float(n_arr[m][0])   # n
#                             label_ref = str(name_arr[m][0])
#                             break
        
#             # Draw the black dot and legends
#             if x_ref is not None and y_ref is not None:
#                 # black dot in the main plot (enters legend with name only)
#                 ax.scatter(x_ref, y_ref, c='k', s=s_ref, marker='o', zorder=6, label=label_ref)
        
#                 # black dot in the inset with detailed label (n and v_d)
#                 inset_label = rf"{label_ref}: $n={fmt_n.format(y_ref)}$, $v_d={fmt_vd.format(x_ref)}$"
#                 axins.scatter(x_ref, y_ref, c='k', s=s_ref, marker='o', zorder=7, label=inset_label)
#                 axins.legend(loc=inset_legend_loc, fontsize=8, frameon=True,
#                              borderpad=0.4, handlelength=1.0, handletextpad=0.5)
        
#             # --- Rebuild main axis legend (after everything) ---
#             handles, labels = ax.get_legend_handles_labels()
#             by_label = dict(zip(labels, handles))  # avoid duplicates
#             ax.legend(by_label.values(), by_label.keys(), loc=legend_loc, fontsize=12)
        
#             return fig, ax, axins
        
        
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

                

selector_BK7 = Glass_Selector(configuracion_1, WR=(0.34, 1.10),
                          name_glass="BK7", delta_n=0.025, delta_vd=5.0, pt_threshold=0.8)

print("Candidatos n/vd:", selector_BK7.Glass_idx_filtered.size)
print("Cumplen PT>=0.8:", selector_BK7.Glass_idx_possible.size)
print("Nombres posibles:", selector_BK7.Names_Glass_possible)

fig, ax, axins = selector_BK7.plot_nv_with_inset()
plt.show()




selector_F2 = Glass_Selector(configuracion_1, WR=(0.34, 1.10),
                          name_glass="F2", delta_n=0.025, delta_vd=5.0, pt_threshold=0.7)

print("Candidatos n/vd:", selector_F2.Glass_idx_filtered.size)
print("Cumplen PT>=0.7:", selector_F2.Glass_idx_possible.size)
print("Nombres posibles:", selector_F2.Names_Glass_possible)

fig, ax, axins = selector_F2.plot_nv_with_inset()
plt.show()

















