# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:18:48 2025

@author: MORGANRHAINAJERAROA
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from ..equations.tracing import trace_rays
import matplotlib as mpl
from itertools import cycle
from pathlib import Path

# ======================================
#    Spot Diagram Plotting Functions
# ======================================
   

# def run_spots_for_fields(Pup, Rays, Field_ccd, wavelengths, xairy, yairy, name_save, save_dir=None):
    
#     # # Carpeta de salida
#     # if save_dir is None:
#     #     # raíz del proyecto / SPT_Diagrams
#     #     script_dir = os.path.abspath(os.path.dirname(__file__))
#     #     base_path  = os.path.abspath(os.path.join(script_dir, '..', '..'))
#     #     save_dir   = os.path.join(base_path, 'SPT_Diagrams')
#     # os.makedirs(save_dir, exist_ok=True)

#     fields = [
#         (0.0, 0.0, 'Field_0'),
#         (np.rad2deg(Field_ccd), 0.0, 'Field_1'),
#         (0.0, -np.rad2deg(Field_ccd), 'Field_2'),
#         (np.rad2deg(Field_ccd), -np.rad2deg(Field_ccd), 'Field_3')
#     ]

#     List_Radius = []

#     for fx, fy, name in fields:
#         Pup.Ptype = "hexapolar"
#         Pup.FieldX, Pup.FieldY = fx, fy

#         traced = trace_rays(Pup, wavelengths, Rays)
#         (Xa, Ya, *_), (Xb, Yb, *_), (Xc, Yc, *_) = traced
#         Spot_Set = [(Xa, Ya), (Xb, Yb), (Xc, Yc)]

#         # Guarda en SPT_Diagrams
#         geo_r, rms_r = plot_spot_diagram(
#             Spot_Set, [fx, fy], xairy, yairy,
#             field_name=name, custom_name=name_save,
#             save=True, output_folder=save_dir
#         )
#         List_Radius.append((geo_r, rms_r))

#     return np.array(List_Radius)


def run_spots_for_fields(
    Pup,
    Rays,
    Field_ccd,
    wavelengths,
    xairy,
    yairy,
    name_save,
    save_dir=None,
    *,
    fields=None,
    ptype="hexapolar",
    save=True,
    show=False, 
    show_geo_circle=True,
    show_rms_circle=True, 
    lock_box_across_fields=True,   # <<--- NUEVO: forzar caja común
    box_include_airy=True          # <<--- si el cálculo del límite considera Airy
):
    """
    Traza rayos y genera/guarda spot diagrams para varios campos, devolviendo
    radios geométricos y RMS por campo.

    Parámetros
    ----------
    Pup : objeto Pupil (mutable); se clona internamente por seguridad.
    Rays : objeto/estructura de rayos para el trazado.
    Field_ccd : float (radianes) o iterable; si es float se usa para armar campos por defecto.
    wavelengths : iterable de longitudes de onda o identificadores usados por trace_rays.
    xairy, yairy : arrays o escalares con escalado/normalización (según tu plot_spot_diagram).
    name_save : str, prefijo del nombre de archivo al guardar.
    save_dir : str | None, carpeta de salida. Si None, usa "<raiz>/SPT_Diagrams".
    fields : list[(fx_deg, fy_deg, name)] | None. Si None, usa 4 campos por defecto.
    ptype : str, tipo de muestreo pupilar (default "hexapolar").
    save : bool, si True llama a plot_spot_diagram con save=True.

    Retorna
    -------
    radii : np.ndarray shape (Nfields, 2) con columnas [geo_r, rms_r]
    meta  : dict con 'fields_deg' (lista de tuplas) y 'output_dir' usado
    """

    # 1) Campos por defecto (en grados) si no se proveen
    if fields is None:
        # Si Field_ccd es un escalar (en rad), convertir a deg
        if np.isscalar(Field_ccd):
            fdeg = float(np.rad2deg(Field_ccd))
            fields = [
                (0.0, 0.0, 'Field_0'),
                (fdeg, 0.0, 'Field_1'),
                (0.0, -fdeg, 'Field_2'),
                (fdeg, -fdeg, 'Field_3'),
            ]
        else:
            raise ValueError("Si 'fields' es None, 'Field_ccd' debe ser escalar (radianes).")

    # 2) Carpeta de salida
    if save_dir is None:
        # raíz del proyecto / SPT_Diagrams
        script_dir = os.path.abspath(os.path.dirname(__file__))
        base_path  = os.path.abspath(os.path.join(script_dir, '..', '..'))
        save_dir   = os.path.join(base_path, 'Images\SPT_Diagrams')

    if save:
        os.makedirs(save_dir, exist_ok=True)

    # ---------- PASO A: trazar y almacenar ----------
    traced_sets = []     # [(spot_set, (fx,fy,field_name)), ...]
    per_field_limits = []  # para calcular el límite global

    for fx_deg, fy_deg, field_name in fields:
        Pup.Ptype  = ptype
        Pup.FieldX, Pup.FieldY = fx_deg, fy_deg

        # 3a) Trazado
        traced = trace_rays(Pup, wavelengths, Rays)  # lista de tuples (X,Y,...)
        spot_set = []
        for t in traced:
            X, Y = t[0], t[1]
            spot_set.append((np.asarray(X), np.asarray(Y)))

        # Calcula un radio de referencia por campo (sin plot)
        X_all = np.concatenate([X for (X, _) in spot_set])
        Y_all = np.concatenate([Y for (_, Y) in spot_set])
        h, k = calculate_geometrical_center(X_all, Y_all)
        _, Geo_r, Rms_r = calculate_radius(X_all, Y_all, h, k)

        # radios de los puntos YA centrados para robustez
        r_pts = 0.0
        for (X, Y) in spot_set:
            r_pts = max(r_pts, float(np.max(np.hypot(X - h, Y - k))))

        r_airy = 0.0
        if xairy is not None and yairy is not None:
            r_airy = float(np.max(np.hypot(np.asarray(xairy), np.asarray(yairy))))

        r_ref = max(r_pts, r_airy if box_include_airy else 0.0, float(Geo_r), float(Rms_r))
        # pequeño margen (5%)
        per_field_limits.append(1.05 * r_ref)

        traced_sets.append((spot_set, (fx_deg, fy_deg, field_name)))

    # ---------- PASO B: fija el límite global y plotea ----------
    box_limit = max(per_field_limits) if (lock_box_across_fields and len(per_field_limits) > 0) else None

    list_radius = []
    for spot_set, (fx_deg, fy_deg, field_name) in traced_sets:
        geo_r, rms_r = plot_spot_diagram(
            Coordinates=spot_set,
            fields=[fx_deg, fy_deg],
            x_airy=xairy, y_airy=yairy,
            field_name=field_name, custom_name=name_save,
            show=show, save=save, output_folder=save_dir,
            show_geo_circle=show_geo_circle, show_rms_circle=show_rms_circle,
            box_limit=box_limit,               # <<--- LÍMITE GLOBAL
            box_include_airy=box_include_airy  # consistente con el cálculo
        )
        list_radius.append((float(geo_r), float(rms_r)))

    radii = np.asarray(list_radius, dtype=float)
    meta  = {
        "fields_deg": [(fx, fy, nm) for fx, fy, nm in fields],
        "output_dir": save_dir,
        "ptype": ptype,
        "name_save": name_save,
        "box_limit": box_limit
    }
    return radii, meta


"""
======================================
  Function: calculate_radius
======================================

This function calculates two important radius metrics for a set of spot points:
1. Geometrical Radius (GEO_Radius): The maximum Euclidean distance from the 
   center of the spot to the furthest point.
2. Root Mean Square Radius (RMS_Radius): The square root of the mean squared 
   distances of all points to the center.

Parameters:
- X_lists (list or np.array): List or array of x coordinates of the spot points.
- Y_lists (list or np.array): List or array of y coordinates of the spot points.
- center_x (float): The x-coordinate of the geometrical center.
- center_y (float): The y-coordinate of the geometrical center.

Steps:
1. Compute the Euclidean distance of each point to the center.
2. Find the maximum distance (Geometrical Radius).
3. Compute the root mean square of the distances (RMS Radius).
4. Return the center coordinates along with the two calculated radii.

Returns:
- (center_x, center_y) (tuple): The center coordinates of the spot diagram.
- GEO_Radius (float): The maximum distance from the center.
- RMS_Radius (float): The root mean square of all distances from the center.

======================================
"""

def calculate_radius(X_lists, Y_lists, center_x, center_y):
    
    # Compute Euclidean distances from each point
    # to the geometrical center (center_x, center_y).
    distances = np.sqrt((X_lists - center_x)**2 + (Y_lists - center_y)**2)
    
    # Calculate the Geometrical Radius (maximum distance).
    GEO_Radius = np.max(distances)  # Furthest point from the center
    
    # Calculate the RMS Radius (root of the mean squared distances).
    RMS_Radius = np.sqrt(np.mean(distances ** 2))  # Root Mean Square of distances
    
    # Return the center coordinates and the two radius metrics.
    return (center_x, center_y), GEO_Radius, RMS_Radius

#########################################################################################################



"""
======================================
  Function: calculate_geometrical_center
======================================

This function calculates the geometrical center (centroid) of a set of 
coordinates in the X and Y planes. It finds the extreme values (max and min) 
for both X and Y coordinates and computes the average to determine the central 
point. This is particularly useful for centering spot diagrams or optical 
fields in lens design and analysis.

Parameters:
- X_all (list or np.array): List or array of x coordinates.
- Y_all (list or np.array): List or array of y coordinates.

Steps:
1. Compute the maximum and minimum values for both X and Y coordinates.
2. Store these values in lists for maximum and minimum.
3. Calculate the geometrical center by averaging the maximum and minimum 
   for each axis (X and Y).
4. Return the calculated center coordinates (h, k).

Returns:
- h (float): Geometrical center coordinate for the X-axis.
- k (float): Geometrical center coordinate for the Y-axis.

======================================
"""

def calculate_geometrical_center(X_all, Y_all):
    
    # -------------------------------------------------
    # Step 1: Compute the maximum and minimum values
    #         for the X and Y coordinates.
    # -------------------------------------------------
    
    x_setmax = [max(X_all)]  # Find the maximum X value
    x_setmin = [min(X_all)]  # Find the minimum X value
    y_setmax = [max(Y_all)]  # Find the maximum Y value
    y_setmin = [min(Y_all)]  # Find the minimum Y value
    
    # -------------------------------------------------
    # Step 2: Calculate the geometrical center (centroid)
    #         by averaging the maximum and minimum values.
    # -------------------------------------------------
    
    h = (max(x_setmax) + min(x_setmin)) / 2  # X center
    k = (max(y_setmax) + min(y_setmin)) / 2  # Y center
    
    # -------------------------------------------------
    # Step 3: Return the calculated center coordinates.
    # -------------------------------------------------
    
    return h, k

#########################################################################################################

"""
======================================
  Function: plot_spot_diagram
======================================

This function generates and displays the spot diagram for a specified 
field configuration in an optical system. It represents the distribution 
of rays at the image plane for three different wavelengths, along with 
the Airy disk, the geometrical radius, and the RMS radius.

Parameters:
- Xa, Ya, Xb, Yb, Xc, Yc (list): Lists of coordinates representing 
  the intersection of rays with the image plane for three wavelengths:
  - Xa, Ya -> Wavelength 0.35 μm (blue)
  - Xb, Yb -> Wavelength 0.43 μm (green)
  - Xc, Yc -> Wavelength 0.55 μm (red)
- x_airy, y_airy (list): Coordinates for the Airy disk representation.
- field_name (str): Identifier for the field being analyzed. It determines 
  which labels (X, Y) are displayed in the plot.

Steps:
1. Concatenate the coordinates for all wavelengths into single lists.
2. Compute the geometrical center of the spot diagram.
3. Calculate the geometrical (GEO) radius and the root-mean-square (RMS) radius.
4. Recenter the spots to the origin (0, 0) based on the computed center.
5. Generate the spot diagram plot:
   - Plot the three wavelengths with different colors.
   - Plot the Airy disk as a dashed circle.
   - Plot the GEO and RMS radii as dashed and dotted circles, respectively.
6. Configure the plot aesthetics:
   - Increase label sizes for better readability.
   - Conditionally add axis labels based on the field being plotted.
7. Save the plot as a PNG file named after the field identifier.
8. Display the plot.

Returns:
- Geo_r (float): The geometrical radius of the spot diagram.
- Rms_r (float): The root-mean-square radius of the spot diagram.

======================================
"""

def plot_spot_diagram(
    Coordinates, fields, x_airy, y_airy, field_name, custom_name="",
    *, show=False, save=False, output_folder=None, dpi=300, show_geo_circle=True,     
    show_rms_circle=True,  box_limit=None, box_include_airy=True):
    
    # 1) Unpack campos (deg)
    field_x, field_y = float(fields[0]), float(fields[1])

    # 2) Asegura arrays + evita mutar entradas
    XY = [(np.asarray(X).copy(), np.asarray(Y).copy()) for (X, Y) in Coordinates]

    # 3) Centro geométrico y radios
    X_all = np.concatenate([X for (X, _) in XY])
    Y_all = np.concatenate([Y for (_, Y) in XY])

    h, k = calculate_geometrical_center(X_all, Y_all)
    _, Geo_r, Rms_r = calculate_radius(X_all, Y_all, h, k)

    # 4) Recentrado sin side-effects
    XYc = [(X - h, Y - k) for (X, Y) in XY]
    primary = ["blue", "green", "red"]

    # Si hay más de 3 sets, usa el ciclo de Matplotlib para los extra
    default_cycle = mpl.rcParams.get("axes.prop_cycle", None)
    cyc = cycle(default_cycle.by_key()["color"]) if default_cycle else cycle(["C0","C1","C2","C3","C4","C5"])
    
    colors = primary[:len(XYc)]
    if len(XYc) > 3:
        colors += [next(cyc) for _ in range(len(XYc) - 3)]

    # 5) Plot (opcional)
    if show or save:
        fig, ax = plt.subplots()
        for (Xc_i, Yc_i), c in zip(XYc, colors):
            ax.plot(Xc_i, Yc_i, 'x', markersize=9, color=c)

        # Disco de Airy
        if x_airy is not None and y_airy is not None:
            ax.plot(x_airy, y_airy, color="k", linestyle='solid', linewidth=3.0)

        # Círculos GEO y RMS (opcionales)
        if show_geo_circle:
            ax.add_patch(patches.Circle(
                (0.0, 0.0), radius=Geo_r, edgecolor='k',
                linestyle='-.', linewidth=3.0, facecolor='none'
            ))
        if show_rms_circle:
            ax.add_patch(patches.Circle(
                (0.0, 0.0), radius=Rms_r, edgecolor='k',
                linestyle=':',  linewidth=3.0, facecolor='none'
            ))

        # Estética
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(axis='both', which='both', labelsize=12, length=6, width=3.0)
        ax.set_title(f'OBJ: {field_x:.3f}, {field_y:.3f} deg', fontsize=25, fontfamily="serif")
        ax.set_xlabel(f'IMA: {h:.3f}, {k:.3f} mm', fontsize=25, color='black', labelpad=0.8, fontfamily="serif")
    
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        ax.relim(); ax.autoscale_view()
    
        # --- Ajuste de caja ---
        if box_limit is None:
            # Modo local: calcula el tamaño con el máximo del campo actual
            r_pts = 0.0
            for (Xc_i, Yc_i) in XYc:
                if len(Xc_i) > 0:
                    r_pts = max(r_pts, float(np.max(np.hypot(Xc_i, Yc_i))))
            r_airy = 0.0
            if x_airy is not None and y_airy is not None:
                r_airy = float(np.max(np.hypot(np.asarray(x_airy), np.asarray(y_airy))))
            r_ref = max(r_pts, r_airy if box_include_airy else 0.0, float(Geo_r), float(Rms_r))
            pad = 0.05 * r_ref if r_ref > 0 else 1e-6
            lim = r_ref + pad
        else:
            # Modo global: usa el límite común
            lim = float(box_limit)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal', adjustable='box')

        # Estética, regla/escala usando el ancho real:
        x_min, x_max = ax.get_xlim()
        box_size = (x_max - x_min)  # mm
        bar_x, tick_w, txt_x = 1.00, 0.02, 1.02
        ax.plot([bar_x, bar_x], [0, 1], transform=ax.transAxes, color='black', linewidth=2, clip_on=False, zorder=10)
        ax.plot([bar_x - tick_w, bar_x + tick_w], [0, 0], transform=ax.transAxes, color='black', linewidth=2, clip_on=False, zorder=10)
        ax.plot([bar_x - tick_w, bar_x + tick_w], [1, 1], transform=ax.transAxes, color='black', linewidth=2, clip_on=False, zorder=10)
        ax.text(txt_x, 0.5, f"{box_size*1000:.2f} μm", transform=ax.transAxes, rotation=90,
                va='center', ha='left', fontsize=25, color='black', fontfamily="serif", clip_on=False)

        # Guardado
        if save:
            if output_folder is None:
                script_dir = os.path.abspath(os.path.dirname(__file__))
                base_path  = os.path.abspath(os.path.join(script_dir, '..', '..'))
                output_folder = os.path.join(base_path, 'Images/SPT_diagram')
                
            # Crear las carpetas necesarias
            Path(output_folder).mkdir(parents=True, exist_ok=True)

            # Nombre limpio del archivo
            base_filename = f"spot_diagram_{field_name}_{custom_name}" if custom_name else f"spot_diagram_{field_name}"

            pdf_path = os.path.join(output_folder, base_filename + ".pdf")

            plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.01, dpi=dpi)
            print(f"[saved] {pdf_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return Geo_r, Rms_r