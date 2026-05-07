# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 17:56:12 2025

@author: MORGANRHAINAJERAROA
"""

"""
Utilities for optical glass selection and neighbor analysis in the (n, Vd) plane.

Functions:
- pick_name_universe: choose candidate name set ("possible", "filtered", "all")
- analyze_neighbors: compute left/center/right neighbors, n, Vd, r, Δr²
- pick_min_delta_r2: pick neighbor with minimal Δr²
"""



import numpy as np
from ..classes.Glass_selector import Glass_Selector


def set_pair_glass(sys, idx_a, name_a, idx_b, name_b):
    sys.SDT[idx_a].Glass = name_a
    sys.SDT[idx_b].Glass = name_b
    if hasattr(sys, "SetGlass"):
        sys.SetGlass()


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
    
def analyze_ranked(
    configuracion,
    refs,
    WR=(0.35, 1.0),
    delta_n=0.025,
    delta_vd=5.0,
    pt_threshold=0.8,
    prefer_set="possible",

):
    """
    Para cada vidrio 'ref' en refs:
      - arma el universo (names_arr, n_arr, vd_arr) según 'prefer_set'
      - ubica a ref como centro y calcula r_C = sqrt(n_c^2 + vd_c^2)
      - calcula r_i para todo el universo y Δr²_i = (r_i - r_C)^2
      - devuelve un ranking ascendente por Δr² (menor→mayor), con nombres, n, vd

    Retorna:
      dict[ref] = {
        'status', 'where', 'index', 'name_center',
        'rank_names': np.ndarray[str] ordenada,
        'rank_n':     np.ndarray[float] ordenada,
        'rank_vd':    np.ndarray[float] ordenada,
        'rank_delta_r2': np.ndarray[float] ordenada,
        'top':  [(name, n, vd, delta_r2), ...] si top_k se especifica
      }
    """
    out = {}

    for name in refs:
        sel = Glass_Selector(
            configuracion,
            WR=WR,
            name_glass=name,
            delta_n=delta_n,
            delta_vd=delta_vd,
            pt_threshold=pt_threshold
        )

        # universo de nombres según prefer_set
        names_arr, where = pick_name_universe(sel, prefer=prefer_set)
        names_arr = np.asarray(names_arr, dtype=object)

        # índice del ref en ese universo
        idxs = np.where(names_arr == name)[0]
        if idxs.size == 0:
            out[name] = {
                "status": "not_found",
                "where": where,
                "message": f"'{name}' no está en el set '{prefer_set}' ({where}).",
                "names": names_arr
            }
            continue
        i = int(idxs[-1])

        # arrays n, vd:
        if prefer_set == "possible":
            n_arr, vd_arr = sel.n_all_possible, sel.vd_all_possible
        elif prefer_set == "filtered":
            n_arr, vd_arr = sel.n_all_filtered, sel.vd_all_filtered
        elif prefer_set == "all":
            n_arr, vd_arr = sel.n_all, sel.vd_all
        else:
            # fallback: mapeo por nombre
            lookup = {nm: (float(vn), float(vd)) for nm, vn, vd
                      in zip(sel.Names_Glass, sel.n_all, sel.vd_all)}
            n_list, vd_list = [], []
            for nm in names_arr:
                nv, vv = lookup.get(str(nm), (np.nan, np.nan))
                n_list.append(nv); vd_list.append(vv)
            n_arr  = np.asarray(n_list, dtype=float)
            vd_arr = np.asarray(vd_list, dtype=float)

        # centro
        n_c, vd_c = float(n_arr[i]), float(vd_arr[i])
        r_c = np.sqrt(n_c*n_c + vd_c*vd_c)

        # r_i y Δr² para todo
        r_all = np.sqrt(n_arr*n_arr + vd_arr*vd_arr)
        delta_r2_all = (r_all - r_c)**2


        # orden ascendente por Δr²
        order = np.argsort(delta_r2_all)  # menor→mayor
        rank_names = names_arr[order]
        rank_n     = n_arr[order]
        rank_vd    = vd_arr[order]
        rank_dr2   = delta_r2_all[order]

        res = {
            "status": "ok",
            "where": where,
            "index": i,
            "name_center": names_arr[i],
            "rank_names": rank_names,
            "rank_n": rank_n,
            "rank_vd": rank_vd,
            "rank_delta_r2": rank_dr2
        }

        out[name] = res

    return out