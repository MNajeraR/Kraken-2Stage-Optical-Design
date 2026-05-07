
import numpy as np
import matplotlib.pyplot as plt
# import os
from pathlib import Path
from ..equations.tracing import trace_rays
import os
from scipy.stats import gaussian_kde
from scipy.special import j0, j1

def calculate_geometrical_center(X_all, Y_all):
    
    # -------------------------------------------------
    # Step 1: Compute the maximum and minimum values
    #         for the X and Y coordinates.
    # -------------------------------------------------
    
    
    x_max, x_min = np.max(X_all), np.min(X_all) # Find the X value
    y_max, y_min = np.max(Y_all), np.min(Y_all) # Find the maximum Y value
    
    # -------------------------------------------------
    # Step 2: Calculate the geometrical center (centroid)
    #         by averaging the maximum and minimum values.
    # -------------------------------------------------
    
    h = 0.5 * (x_max + x_min)   # X center
    k = 0.5 * (y_max + y_min)   # Y center
    
    return h, k


def encircled_energy(distance, weights=None):
    """
    Devuelve la curva EE(r) como función escalonada:
      r_sorted : radios ordenados (mismo tamaño que distance)
      EE       : fracción acumulada (0..1) para cada r_sorted
    Acepta 'weights' si cada rayo tiene potencia distinta.
    """
    distance = np.asarray(distance, dtype=float)
    if weights is None:
        weights = np.ones_like(distance, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != distance.shape:
            raise ValueError("weights debe tener la misma forma que distance")

    # Ordenar por radio e integrar acumulado
    idx = np.argsort(distance)
    r_sorted = distance[idx]
    w_sorted = weights[idx]
    Wtot = np.sum(w_sorted)
    EE = np.cumsum(w_sorted) / Wtot
    return r_sorted, EE

def airy_encircled_energy(r_over_airy):
    """
    Encircled energy teórica de un disco de Airy para un pupil circular.
    r_over_airy = r / r_Airy (adimensional).
    Devuelve EE_Airy entre 0 y 1.
    """
    r_over_airy = np.asarray(r_over_airy, dtype=float)
    x = np.pi * r_over_airy
    # Fórmula estándar
    EE = 1.0 - j0(x)**2 - j1(x)**2
    # Seguridad numérica
    EE = np.clip(EE, 0.0, 1.0)
    return EE

def encircled_energy_kde(distance, weights=None, *, n_points=512, r_factor=1.1, bw_method=0.05):
    """
    Versión SUAVE de EE(r) usando un kernel gaussiano.
    Se usa solo para graficar (no para las métricas exactas).

    Parámetros
    ----------
    distance : array-like
        Radios de cada rayo (>=0) en metros.
    weights : array-like o None
        Pesos de cada rayo. Si None, todos valen 1.
    n_points : int
        Número de puntos de la malla radial para la curva suave.
    r_factor : float
        Factor para extender el rango radial: r_max_plot = r_factor * max(distance).
        (Esto ayuda a que la cola se vea bien).
    bw_method : str o float
        Parámetro de ancho de banda de gaussian_kde ("scott", "silverman" o un número).

    Devuelve
    --------
    r_grid : ndarray
        Malla de radios (en las mismas unidades que `distance`).
    EE_smooth : ndarray
        Fracción de energía encerrada (0..1) en esos radios.
    """
    r = np.asarray(distance, dtype=float)
    if weights is None:
        w = np.ones_like(r, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != r.shape:
            raise ValueError("weights debe tener la misma forma que distance")

    # Forzar radios >= 0
    r = np.clip(r, 0.0, None)

    r_max = r.max()
    if r_max == 0.0:
        return np.array([0.0]), np.array([1.0])

    # KDE 1D del radio
    kde = gaussian_kde(r, weights=w, bw_method=bw_method)

    # Malla radial (extendida un poco más allá del máximo)
    r_grid = np.linspace(0.0, r_factor * r_max, n_points)

    # pdf(r)
    pdf = kde(r_grid)
    pdf = np.clip(pdf, 0.0, None)  # por seguridad numérica

    # CDF(r) ~ integral de pdf(r)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]  # normalizar para que llegue exactamente a 1

    return r_grid, cdf


def radius_at_fraction(r, EE, frac):
    if frac <= EE[0]:  return r[0]
    if frac >= EE[-1]: return r[-1]
    return np.interp(frac, EE, r)



# def plot_all_EE_for_fields(
#     Pup, Rays, Field_ccd, wavelengths, *,
#     fields=None, ptype="hexapolar", show_r50=True,
#     save=False,                     #  Guarda solo si True
#     save_dir=None,                  #  Carpeta destino (default EE_Diagrams)
#     filename=None,                  #  Nombre del archivo (sin ruta)
#     show=False,                     #  Muestra en pantalla si True
# ):
#     """
#     Grafica EE(r) para todos los campos en una sola figura (opcionalmente).
#     - fields: lista [(fx_deg, fy_deg, name), ...]; si None se usa set por defecto.
#     - show_r50: si True, marca r50 de cada campo con una marca y líneas guía.
#     - save: si True, guarda la figura en save_dir/filename (PDF/PNG según extensión).
#     - save_dir: carpeta de salida (por defecto <raiz_proyecto>/EE_Diagrams).
#     - filename: nombre del archivo (p.ej. 'EE_S-FPL51_F2HT_fields.pdf').
#     - show: si True, hace plt.show().
#     Retorna: lista de r50 por campo (en µm).
#     """

#     # 1) Campos por defecto (en grados)
#     theta_deg = float(np.rad2deg(Field_ccd))
#     if fields is None:
#         fields = [
#             (0.0,        0.0,        "Field_0"),
#             (theta_deg,  0.0,        "Field_1"),
#             (0.0,       -theta_deg,  "Field_2"),
#             (theta_deg, -theta_deg,  "Field_3"),
#         ]

#     # 2) Guardar estado original del pupil
#     _orig_ptype = getattr(Pup, "Ptype", None)
#     _orig_fx    = getattr(Pup, "FieldX", None)
#     _orig_fy    = getattr(Pup, "FieldY", None)

#     curves = []   # [(name, r_um, EE(0..1), r50_um)]
#     try:
#         Pup.Ptype = ptype
#         for fx_deg, fy_deg, name in fields:
#             Pup.FieldX, Pup.FieldY = float(fx_deg), float(fy_deg)

#             traced = trace_rays(Pup, wavelengths, Rays)
#             (Xa, Ya, *_), (Xb, Yb, *_), (Xc, Yc, *_) = traced

#             X_all = np.concatenate([Xa, Xb, Xc]).astype(float)
#             Y_all = np.concatenate([Ya, Yb, Yc]).astype(float)

#             # Centro geométrico y recentrado
#             h, k = calculate_geometrical_center(X_all, Y_all)
#             X_all -= h; Y_all -= k

#             # # EE
            
#             # Radios 
#             r_dist = np.hypot(X_all, Y_all)
        
#             # 1) EE exacta PARA MÉTRICAS (r50, etc.)
#             r_sorted_exact, EE_exact = encircled_energy(r_dist)
#             r_um_exact = 1000.0 * r_sorted_exact
#             r50_um = radius_at_fraction(r_um_exact, EE_exact, 0.50)
        
#             # 2) EE SUAVE 
#             r_grid, EE_smooth = encircled_energy_kde(r_dist, n_points=512, r_factor=1.1)
#             r_um_smooth = 1000.0 * r_grid
        
#             curves.append((name, r_um_smooth, EE_smooth, r50_um))

#     finally:
#         # restaurar estado del pupil
#         if _orig_ptype is not None: Pup.Ptype  = _orig_ptype
#         if _orig_fx    is not None: Pup.FieldX = _orig_fx
#         if _orig_fy    is not None: Pup.FieldY = _orig_fy

#     # Si no se va a mostrar ni guardar, no generes la figura
#     if not show and not save:
#         return [c[3] for c in curves]

#     # 3) Figura única con todas las curvas
#     # Encontrar el radio máximo global donde las curvas llegan a 100 %
#     r100_list = [c[1][-1] for c in curves]   # último r_um de cada curva
#     global_r100 = max(r100_list)
    
#     fig, ax = plt.subplots(figsize=(13, 9.0))
#     r_50_set = []
#     for name, r_um, EE, r50_um in curves:
#         # Copias para poder extender
#         r_plot  = np.array(r_um, copy=True)
#         EE_plot = np.array(EE, copy=True) * 100.0
    
#         # Si este campo llega a 100% antes que el máximo global,
#         # añadimos un tramo horizontal hasta global_r100.
#         if r_plot[-1] < global_r100:
#             r_plot  = np.append(r_plot,  global_r100)
#             EE_plot = np.append(EE_plot, EE_plot[-1])  # sigue al 100%
    
#         ax.plot(r_plot, EE_plot, label=fr"{r50_um:.2f} µm", lw=6)
    
#         if show_r50:
#             ax.plot([r50_um], [50], marker="o")
#             ax.hlines(y=50, xmin=0, xmax=r50_um,
#                       linestyles="--", linewidth=1.5, alpha=0.6)
#             ax.vlines(x=r50_um, ymin=0, ymax=50,
#                       linestyles="--", linewidth=1.5, alpha=0.6)
#         r_50_set.append(r50_um)
        
#     for spine in ax.spines.values():
#         spine.set_linewidth(2)

#     ax.tick_params(axis='both', which='major', labelsize=25, length=6, width=3)
#     ax.set_xlim(0, global_r100)
#     ax.set_ylim(0, 102)
#     ax.set_xlabel("Radius (µm)", fontsize=25, fontfamily="serif", fontname="Times New Roman")
#     ax.set_ylabel("Encircled Energy (%)", fontsize=25, fontfamily="serif", fontname="Times New Roman")
#     ax.grid(True, alpha=0.3)
#     ax.legend(title="Field", ncol=2,
#               prop={"family": "serif", "size": 20},
#               title_fontsize=22)

#     plt.tight_layout()

#     # Guardar si se pidió
#     if save:
#         if save_dir is None:
#             # raíz del proyecto / EE_Diagrams
#             script_dir = os.path.abspath(os.path.dirname(__file__))
#             base_path  = os.path.abspath(os.path.join(script_dir, '..', '..'))
#             save_dir   = os.path.join(base_path, 'Images\EE_Diagrams')
#         Path(save_dir).mkdir(parents=True, exist_ok=True)

#         if filename is None:
#             filename = "EE_fields.pdf"  # por defecto
#         save_path = os.path.join(save_dir, filename)
#         fig.savefig(save_path, dpi=150)
#         print(f"[saved] {save_path}")

#     if show:
#         plt.show()
#     else:
#         plt.close(fig)

#     return r_50_set



def plot_all_EE_for_fields(
    Pup, Rays, Field_ccd, wavelengths, *,
    fields=None, ptype="hexapolar", show_r50=True,
    save=False,                     #  Guarda solo si True
    save_dir=None,                  #  Carpeta destino (default EE_Diagrams)
    filename=None,                  #  Nombre del archivo (sin ruta)
    show=False,                     #  Muestra en pantalla si True
    airy_radius_um=None,
    multiply_by_diff_limit=False,   # Escalar EE geométrica por EE_Airy
    show_airy_limit=False,          # Dibujar (o no) la curva Airy teórica
):
   """
    Grafica EE(r) para todos los campos en una sola figura (opcionalmente).
    - fields: lista [(fx_deg, fy_deg, name), ...]; si None se usa set por defecto.
    - show_r50: si True, marca r50 de cada campo con una marca y líneas guía.
    - save: si True, guarda la figura en save_dir/filename (PDF/PNG según extensión).
    - save_dir: carpeta de salida (por defecto <raiz_proyecto>/EE_Diagrams).
    - filename: nombre del archivo (p.ej. 'EE_S-FPL51_F2HT_fields.pdf').
    - show: si True, hace plt.show().
    Retorna: lista de r50 por campo (en µm).
    """
   # 1) Campos por defecto (en grados)
   theta_deg = float(np.rad2deg(Field_ccd))
   if fields is None:
       fields = [
           (0.0,        0.0,        "Field_0"),
           (theta_deg,  0.0,        "Field_1"),
           (0.0,       -theta_deg,  "Field_2"),
           (theta_deg, -theta_deg,  "Field_3"),
        ]
    
   # 2) Guardar estado original del pupil
   _orig_ptype = getattr(Pup, "Ptype", None)
   _orig_fx    = getattr(Pup, "FieldX", None)
   _orig_fy    = getattr(Pup, "FieldY", None)
    
   # curves: (name, r_um_smooth, EE_geo_smooth, EE_airy, EE_eff, r50_um)
   curves = []
   try:
       Pup.Ptype = ptype
       for fx_deg, fy_deg, name in fields:
           Pup.FieldX, Pup.FieldY = float(fx_deg), float(fy_deg)
    
           traced = trace_rays(Pup, wavelengths, Rays)
           (Xa, Ya, *_), (Xb, Yb, *_), (Xc, Yc, *_) = traced
    
           X_all = np.concatenate([Xa, Xb, Xc]).astype(float)
           Y_all = np.concatenate([Ya, Yb, Yc]).astype(float)
    
           # Centro geométrico y recentrado
           h, k = calculate_geometrical_center(X_all, Y_all)
           X_all -= h
           Y_all -= k
    
           # Radios
           r_dist = np.hypot(X_all, Y_all)
        
           # EE suave geométrica (KDE)
           r_grid, EE_smooth_geo = encircled_energy_kde(
                r_dist,
                n_points=512,
                r_factor=1.2,
                bw_method=0.1
            )
           r_um_smooth = 1000.0 * r_grid
            
           # r50 geométrico (en µm) usando la curva suave
           r50_geo_um = radius_at_fraction(r_um_smooth, EE_smooth_geo, 0.50)
            
           # EE teórica de Airy en la misma malla radial
           EE_airy = None
           EE_eff  = EE_smooth_geo  # por defecto, solo geométrica
           r50_eff_um = None        # opcional: r50 "difractivo"
            
           if airy_radius_um is not None and airy_radius_um > 0:
                r_over_airy = r_um_smooth / airy_radius_um
                EE_airy = airy_encircled_energy(r_over_airy)
            
                if multiply_by_diff_limit:
                    # Geométrica × Airy
                    EE_eff = EE_smooth_geo * EE_airy
                    EE_eff = np.clip(EE_eff, 0.0, 1.0)
            
                    # r50 "difractivo" en µm 
                    r50_eff_um = radius_at_fraction(r_um_smooth, EE_eff, 0.50)
                else:
                    r50_eff_um = None
            
          
           r50_um = r50_eff_um if (multiply_by_diff_limit and r50_eff_um is not None) else r50_geo_um
            
           # Guardas todo en curves:
           curves.append((name, r_um_smooth, EE_smooth_geo, EE_airy, EE_eff, r50_um))
           
    
   finally:
       # Restaurar estado del pupil
        if _orig_ptype is not None: Pup.Ptype  = _orig_ptype
        if _orig_fx    is not None: Pup.FieldX = _orig_fx
        if _orig_fy    is not None: Pup.FieldY = _orig_fy
    
    # Si no se va a mostrar ni guardar, no generes la figura: solo regresa r50
   if not show and not save:
        return [c[5] for c in curves]  # c[5] = r50_um
    
   if multiply_by_diff_limit and (airy_radius_um is not None) and (airy_radius_um > 0):
        # Buscar el índice del campo de referencia (Field_0)
        ref_idx = None
        for i, (name, r_um, EE_geo, EE_airy, EE_eff, r50_um) in enumerate(curves):
            if name == "Field_0":
                ref_idx = i
                break

        if ref_idx is not None:
            ref_name, ref_r_um, ref_EE_geo, ref_EE_airy, ref_EE_eff, ref_r50 = curves[ref_idx]

            # Solo continuamos si el campo 0 tiene EE_eff válida
            if ref_EE_eff is not None:
                ref_EE_eff = np.asarray(ref_EE_eff, dtype=float)
                ref_limit = float(ref_EE_eff[-1])  # ~0.98782891

                new_curves = []
                for i, (name, r_um, EE_geo, EE_airy, EE_eff, r50_um) in enumerate(curves):
                    # Dejar el campo 0 tal cual
                    if i == ref_idx or EE_eff is None:
                        new_curves.append((name, r_um, EE_geo, EE_airy, EE_eff, r50_um))
                        continue

                    # Escalar EE_eff del campo externo para que su máximo coincida con ref_limit
                    EE_eff_arr = np.asarray(EE_eff, dtype=float)
                    limit_i = float(EE_eff_arr[-1])  # valor asintótico actual de este campo

                    if limit_i > 0:
                        scale = ref_limit / limit_i
                        EE_eff_scaled = EE_eff_arr * scale
                        EE_eff_scaled = np.clip(EE_eff_scaled, 0.0, 1.0)

                        # Recalcular r50 con la curva escalada (en µm)
                        r_um_arr = np.asarray(r_um, dtype=float)
                        r50_scaled = radius_at_fraction(r_um_arr, EE_eff_scaled, 0.50)

                        new_curves.append((name, r_um_arr, EE_geo, EE_airy, EE_eff_scaled, r50_scaled))
                    else:
                        # Caso raro: límite_i == 0, dejamos el campo sin cambios
                        new_curves.append((name, r_um, EE_geo, EE_airy, EE_eff, r50_um))

                curves = new_curves 
   
   # 3) Figura única con todas las curvas
   r100_list   = [c[1][-1] for c in curves]   # último radio de cada curva
   global_r100 = max(r100_list)
   r_50_set    = []
    
   fig, ax = plt.subplots(figsize=(13., 9.0))
    
   # 1) Graficar SOLO las curvas de cada campo
   for name, r_um, EE_geo, EE_airy, EE_eff, r50_um in curves:
        # Elegir qué EE graficar (geométrica o «multiplicada»)
        EE_plot = EE_eff if (multiply_by_diff_limit and EE_airy is not None) else EE_geo
    
        r_plot  = np.array(r_um, copy=True)
        EEp100  = np.array(EE_plot, copy=True) * 100.0
    
        # Extender horizontalmente hasta el radio máximo global
        if r_plot[-1] < global_r100:
            r_plot  = np.append(r_plot,  global_r100)
            EEp100  = np.append(EEp100, EEp100[-1])
    
        x_plot = r_plot      # SIEMPRE en micras
        # x_r50  = r50_um
    
        ax.plot(x_plot, EEp100, lw=6, label=fr"{r50_um:.2f} µm")
    
        # if show_r50:
        #     ax.plot([x_r50], [50], marker="o")
        #     ax.hlines(y=50, xmin=0, xmax=x_r50,
        #               linestyles="--", linewidth=1.5, alpha=0.6)
        #     ax.vlines(x=x_r50, ymin=0, ymax=50,
        #               linestyles="--", linewidth=1.5, alpha=0.6)
    
        r_50_set.append(r50_um)
    
   # 2) Curva de Airy global (opcional, solo una)
   if show_airy_limit and (airy_radius_um is not None) and (airy_radius_um > 0):
        x_airy = np.linspace(0, global_r100, 512)
        r_over_airy = x_airy / airy_radius_um
        EE_airy_global = airy_encircled_energy(r_over_airy)
    
        ax.plot(
            x_airy,
            100.0 * EE_airy_global,
            ls="--",
            lw=2,
            alpha=0.8,
            color="k"
        )
    
   # 3) Límites y etiquetas
   ax.tick_params(axis='both', which='major', labelsize=25, length=6, width=3)
   ax.set_xlim(0, global_r100)
   ax.set_ylim(0, 102)
   ax.set_xlabel("Radius (µm)", fontsize=25,
                  fontfamily="serif", fontname="Times New Roman")
   ax.set_ylabel("Encircled Energy (µm)", fontsize=25,
                  fontfamily="serif", fontname="Times New Roman")
   ax.grid(True, alpha=0.3)
   ax.legend(title="50% Encircled Energy", ncol=2,
              prop={"family": "serif", "size": 20},
              title_fontsize=22)
    
   plt.tight_layout()
    
   # Guardar si se pidió
   if save:
        if save_dir is None:
            # raíz del proyecto / EE_Diagrams
            script_dir = os.path.abspath(os.path.dirname(__file__))
            base_path  = os.path.abspath(os.path.join(script_dir, '..', '..'))
            save_dir   = os.path.join(base_path, 'Images', 'EE_Diagrams')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
        if filename is None:
            filename = "EE_fields.pdf"  # por defecto
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
    
   if show:
        plt.show()
   else:
        plt.close(fig)
    
   return r_50_set