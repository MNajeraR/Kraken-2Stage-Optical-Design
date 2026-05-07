
# ===============================
#      Library Imports
# ===============================

import time
import numpy as np
import pkg_resources
import scipy
# import math
# from collections import OrderedDict


import matplotlib.pyplot as plt

# Import KrakenOS and custom modules
from utils import (Paraxial_Cal, ThirdOrder_Cal, analyze_ranked, 
                   Set_Initial_Radius, apply_system_actualization, configure_and_trace,  
                   Prin_Plane, run_spots_for_fields, seidel_terms, airy_data, 
                   configure_pupil_and_ab, set_pair_glass, plot_all_EE_for_fields) 
                    

# Check if KrakenOS is installed, otherwise append its path
required = {'KrakenOS'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    print("KrakenOS not installed")
    import sys
    sys.path.append("../..")

import KrakenOS as Kos

    # ===============================
    #    Constants Definition
    # ===============================
    

import os, re, json

def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)   
 
def safe_apply_actualization(
    *,  # forces keyword-only
    system, R1, R2, R3, R4, n_l1, d_lens1, n_l2, d_lens2,
    TO_data, d4, sup, AperType, AperVal, Field_ccd,
    W_ref, Rays, Kos, AB,
    # intentos en orden (de “fino” a “burdo”)
    tries=((3,6), (3,3), (1,3), (3,1), (1,1)),
    samp=7,
    results_dir="Results",
    glass_a=None, glass_b=None, W=None
):
    os.makedirs(results_dir, exist_ok=True)

    last_err = None
    for (n_nodes, n_arms) in tries:
        try:
            return apply_system_actualization(
                system=system,
                R1=R1, R2=R2, R3=R3, R4=R4,
                n_l1=n_l1, d_lens1=d_lens1,
                n_l2=n_l2, d_lens2=d_lens2,
                TO_data=TO_data, d4=d4,
                sup=sup, AperType=AperType, AperVal=AperVal, Field_ccd=Field_ccd,
                W_ref=W_ref,
                Rays=Rays, Kos=Kos, AB=AB,
                n_nodes=n_nodes, n_arms=n_arms, samp=samp
            )
        except ValueError as e:
            # Captura específica del reshape: “cannot reshape array of size ... into shape (1, N)”
            msg = str(e)
            if "cannot reshape array" in msg or "reshape" in msg:
                print(f"[warn] Quadrature reshape falló con n_nodes={n_nodes}, n_arms={n_arms}: {e}")
                last_err = e
                continue
            else:
                last_err = e
                break
        except Exception as e:
            last_err = e
            break

    # Si llegamos aquí, TODOS los intentos fallaron → registra y “skip”
    bad_path = os.path.join(results_dir, "bad_glasses.jsonl")
    rec = {
        "L1": str(glass_a), "L2": str(glass_b),
        "W_um": float(W) if W is not None else None,
        "error": f"{type(last_err).__name__}: {last_err}"
    }
    with open(bad_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[skip] Actualization falló para L1={glass_a}, L2={glass_b} -> {last_err}")
    return None, None  # señal de fallo        

def run_glasses(glass_a, glass_b,  *, save_prefix=None):  
    
    # Prefijo de archivos a partir de los vidrios
    if save_prefix is None:
        save_prefix = f"{safe_name(glass_a)}_{safe_name(glass_b)}"
    results_dir = "Results"
    ee_dir = os.path.join("Images", "EE_Diagrams")
    ensure_dirs(results_dir, ee_dir)
    
    # ---------- Paraxial setup ----------
    
    per_red = 50.05329978714254                       # Reduction percentage
    EFFL_JSTelescope = 18273.877041856547             # Effective Focal Length
    
    b_3 = -0.009626281725433176                       # Optical power of Lens L1
    b_4 = 0.01065114262935968                       # Optical power of Lens L2
    
    d_lens1 = 30.                                     # Thickness of Lens L1
    d_lens2 = 8.5                                     # Thickness of Lens L2
    
    F_Diameter = 105.                                 # Firts Lens Diameter
    S_Diameter = 90.0                                 # Second Lens Diameter
    
    # Initialize paraxial calculations for the system
    Prx_data = Paraxial_Cal(per_red)
    Prx_data.EFFL_Tel = EFFL_JSTelescope
    
    d_2 = Prx_data.d_1 + Prx_data.d_2 
    d_3 = Prx_data.d_3
    
    # Desired EFFL
    EFFL_Des = Prx_data.EFFL_Tr
    
    # Calculate the field angle for the CCD
    ccd_co = 11.25
    Field_ccd = ccd_co / Prx_data.EFFL_Tr
    
    # Initialize initial height 
    h_i = 1076.00
    
    # ===============================
    #    Starting Points Definition
    # ===============================
    
    # Initial radii setup with R2 = -R1 approximation (Lens-Maker)
    Phi_L1 = b_3  
    n_L1   = 1.50
    d_L1   = d_lens1  
    R1_initial, R2_initial= Set_Initial_Radius(Phi_L1, n_L1, d_L1)
    
    
    Phi_L2 = b_4   
    n_L2   = 1.6
    d_L2   = d_lens2  
    R3_initial, R4_initial = Set_Initial_Radius(Phi_L2, n_L2, d_L2)
    
    
    # Principal plane calculation
    H_1, H_2 = Prin_Plane(n_L1, R1_initial, R2_initial, d_L1)
    H_3, H_4 = Prin_Plane(n_L2, R3_initial, R4_initial, d_L2)
    
    
    # ======================================
    #    Optical Element Initialization
    # ======================================
    
    
    # Object surface configuration
    
    P_Obj = Kos.surf()
    P_Obj.Rc = 0
    P_Obj.Thickness = 1000 + Prx_data.d_1
    P_Obj.Glass = "AIR"
    P_Obj.Diameter = h_i * 2.0
    
    # ______________________________________#
    
    # ======================================
    #  Telescope Initialization
    # ======================================
    
    # Mirror M1 configuration
    
    M1 = Kos.surf()
    M1.Rc = -11.176E+003
    M1.Thickness = - Prx_data.d_1
    M1.k = -1.070110000000E+000
    M1.Glass = "MIRROR"
    M1.Diameter = h_i * 2.0
    
    
    # ______________________________________#
    
    # Mirror M2 configuration
    
    M2 = Kos.surf()
    M2.Rc = -4.4300E+003
    M2.Thickness = d_2
    M2.k = -4.32070000000000E+000
    M2.Glass = "MIRROR"
    M2.Diameter = 3.175E+002 * 2.0
    
    # ______________________________________#
    
    # ======================================
    #    Lens Initialization
    # ======================================
    
    # Lens L1a configuration
    
    L1a = Kos.surf()
    L1a.Rc = R1_initial
    L1a.Thickness = d_lens1
    L1a.Glass = glass_a
    L1a.Diameter = F_Diameter
    
    # ______________________________________#
    
    # Lens L1b configuration
    
    L1b = Kos.surf()
    L1b.Rc = R2_initial
    L1b.Thickness = Prx_data.d_3
    L1b.Glass = "AIR"
    L1b.Diameter = F_Diameter
    
    # ______________________________________#
    
    
    # Lens L2a configuration
    
    L2a = Kos.surf()
    L2a.Rc = R3_initial
    L2a.Thickness = d_lens2
    L2a.Glass = glass_b
    L2a.Diameter = S_Diameter + 0.1*S_Diameter
    
    # ______________________________________#
    
    # Lens L2b configuration
    
    L2b = Kos.surf()
    L2b.Rc = R4_initial
    L2b.Thickness = Prx_data.d_4 
    L2b.Diameter = S_Diameter + 0.1*S_Diameter
    
    # ______________________________________#
    
    # ======================================
    #  Image Plane Initialization
    # ======================================
    
    P_Ima = Kos.surf()
    P_Ima.Rc = 0.0
    P_Ima.Thickness = 0.0
    P_Ima.Glass = "AIR"
    P_Ima.Diameter = 35.0
    P_Ima.Name = "Plano imagen"
    
    # ______________________________________#
    
    # ======================================
    #  System Configuration Setup
    # ======================================
    
    A = [P_Obj, M1, M2, L1a, L1b, L2a, L2b, P_Ima]
    
    config_1 = Kos.Setup()
    
    # ======================================
    #  System Initialization
    # ======================================
    
    # Initialize the optical system with the defined elements and configuration
    Telescope_f85_FR = Kos.system(A, config_1)
    
    # Create a RayKeeper instance for storing and accessing ray tracing results.
    Rays = Kos.raykeeper(Telescope_f85_FR)
    
    
    # Consider the principal planes when accounting for lens thickness.
    Telescope_f85_FR.SDT[2].Thickness = d_2 - H_1
    Telescope_f85_FR.SDT[4].Thickness = d_3 - H_2 - H_3
    Telescope_f85_FR.SDT[6].Thickness = Prx_data.d_4 - H_4
    
    # Update of the optical system
    Telescope_f85_FR.SetData()
    
    
    # ======================================
    #  Paraxial Calculation Setup
    # ======================================
    
    # Define the reference wavelength:
    W = 0.43032015       # in micrometers
    # Define the wavelength range:
    RW = [0.35, W, 0.5499996]    
        
    # Perform a paraxial analysis of the system at the specified wavelength (W).
    try:
        Prx = Telescope_f85_FR.Parax(W)
    except Exception as e:
        bad_path = os.path.join(results_dir, "bad_glasses.txt")
        rec = {"L1": str(glass_a), "L2": str(glass_b), "W_um": float(W), "error": f"{type(e).__name__}: {e}"}
        with open(bad_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[skip] Parax falló con L1={glass_a}, L2={glass_b} -> {e}")
        # Devolver estado “saltado” para que el batch no truene
        return {"L1": str(glass_a), "L2": str(glass_b), "skipped": True, "error": str(e)}
    
    # Extract the list of refractive indices for each surface in the system.
    N = Prx[11]
    
    # Assign the refractive indices of the two main lenses for reference.
    n_l1i = N[3]          # Refractive index of the first lens.
    n_l2i = N[5]          # Refractive index of the second lens.
    
    # Extract the paraxial matrix (ABCD Matrix) of the optical system.
    M_System = np.array(Prx[0]) 
    
    # From the paraxial matrix, extract the D element.
    MS_d = M_System[1][1] 
    
    # ======================================
    #  Pupil Calculation Setup and Seidel Aberration Calculation
    # ======================================
    
    sup = 1              # Number of the surface representing the opening of the system
    AperType = "EPD"     # AperType sets the aperture ("STOP") or entrance pupil diameter ("EPD").
    AperVal = M1.Diameter # Diameter of the entrance pupil
    
    # Pupil setup and Seidel computation
    Pup, AB = configure_pupil_and_ab(Telescope_f85_FR, Kos, sup, RW, AperType,
                                     AperVal, Field_ccd, samp=7)
    
    # Extract individual aberration dimensionless terms
    Sph, Coma, Ast, CLon = seidel_terms(AB, W)
    
    # ======================================
    #  Display Initial Values
    # ====================================== 
    
    print("======================================")
    print("Initial Seidel Aberrations:")
    print(f"Spherical Aberration: {Sph:.2f}")
    print(f"Coma: {Coma:.2f}")
    print(f"Astigmatism: {Ast:.2f}")
    print(f"Longitudinal Chromatic Aberration: {CLon:.2f}")
    print(f"Matrix Element (D) from Paraxial Analysis: {MS_d:.2f}")
    print(f"Effective Focal Length: {Telescope_f85_FR.EFFL:.2f} mm")
    print("======================================")
    print('')
    
    # Return to initial configuration
    Telescope_f85_FR.RestoreData()
    
    # ======================================
    #  Process of Optimization
    # ======================================
    print("\nStarting Optimization...")
    # Start timing the optimization process
    start_time = time.time()
    
    # Initialize third-order aberration analysis with the current lens parameters
    TO_data = ThirdOrder_Cal(Telescope_f85_FR, [b_3, b_4])
    TO_data.W = W
    
    # Define the initial guess for the solver
    initial_guess = [L1a.Rc, L1b.Rc, L2a.Rc, L2b.Rc, Prx_data.d_4]
    
    # Define the boundaries for the solver
    LimInf = [-10000000, -100000000, -10000000, -100000000, -1000]
    LimSup = [ 100000000, 100000000,  100000000, 100000000, 1000]
    b = (LimInf, LimSup)
    
    # Perform the least-squares optimization to find optimal curvatures and thickness
    Curvatur_Rad = scipy.optimize.least_squares(TO_data.SeedPar, initial_guess, bounds=b, verbose=1)
    
    
    # Extract the results
    [R_1, R_2, R_3, R_4, d4] = Curvatur_Rad.x
    elapsed_time = time.time() - start_time
    Set_Initial_Opt = [R_1, R_2, R_3, R_4, d4]
    
    # ======================================
    #  System Update with Optimized Values
    # ======================================
    
    
    # Actualization of the system using Gaussian Quadrature 
    # --- Actualization of the system using Gaussian Quadrature ---
    Telescope_f85_FR, deltaZ = safe_apply_actualization(
        system=Telescope_f85_FR,
        R1=Set_Initial_Opt[0], R2=Set_Initial_Opt[1],
        R3=Set_Initial_Opt[2], R4=Set_Initial_Opt[3],
        n_l1=n_l1i, d_lens1=d_lens1,
        n_l2=n_l2i, d_lens2=d_lens2,
        TO_data=TO_data, d4=Set_Initial_Opt[4],
        sup=sup, AperType=AperType, AperVal=AperVal, Field_ccd=Field_ccd,
        W_ref=0.43032015,
        Rays=Rays, Kos=Kos, AB=AB,
        tries=((3,6), (3,3), (1,3), (3,1), (1,1)),  # puedes ajustar el orden/tamaño
        samp=7,
        results_dir=results_dir,        # asegúrate de tenerlo definido
        glass_a=glass_a, glass_b=glass_b, W=W
    )
    
    # Si falló todo, corta esta iteración y marca “skipped”
    if Telescope_f85_FR is None:
        return {
            "L1": str(glass_a),
            "L2": str(glass_b),
            "skipped": True,
            "error": "apply_system_actualization failed for all quadrature settings"
        }
    
    
    Set_Initial_Opt[4] =  d4 + deltaZ
    
    # ======================================
    #   Display Optimization Results
    # ======================================
    
    print("System parameters updated with optimized values.")
    print("======================================")
    print("Optimization Complete")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print("Optimized Parameters:")
    print(f"R1: {Set_Initial_Opt[0]:.2f} mm")
    print(f"R2: {Set_Initial_Opt[1]:.2f} mm")
    print(f"R3: {Set_Initial_Opt[2]:.2f} mm")
    print(f"R4: {Set_Initial_Opt[3]:.2f} mm")
    print(f"Lens Thickness (d4): {Set_Initial_Opt[4]:.2f} mm")
    print("======================================")
    
    # ======================================
    #  Paraxial Analysis Recalculation
    # ======================================
    
    # Perform a paraxial analysis of the system at the specified wavelength (W).
    Prx = Telescope_f85_FR.Parax(W)
    
    # Extract the paraxial matrix (ABCD Matrix) of the optical system.
    M_System = np.array(Prx[0]) 
    
    # From the paraxial matrix, extract the D element.
    MS_d = M_System[1][1] 
    
    
    # ======================================
    #  Seidel Aberration Analysis Recalculation
    # ======================================
    
    # Pupil setup and Seidel computation recalculation
    Pup, AB = configure_pupil_and_ab(Telescope_f85_FR, Kos, sup, RW, AperType,
                                     AperVal, Field_ccd, samp=7)
    
    # Extract individual aberration dimensionless terms
    Sph, Coma, Ast, CLon = seidel_terms(AB, W)
    
    
    #Calculates the Airy disk radius and its coordinates for plotting.
    Rairy, xairy, yairy = airy_data(Telescope_f85_FR, W, M1)
    
    # ======================================
    #    Plot Generation for All Fields
    # ======================================
    
    wavelengths = [AB.Wf, W, AB.Wc]
    
    List_Radius = run_spots_for_fields(
        Pup, Rays, Field_ccd, wavelengths, xairy, yairy,
        name_save=f"Third_Order_{save_prefix}"
    )
    List_Radius = np.array(List_Radius)
    GEO_R_average = float(np.average(List_Radius[:, 0] * 1000))
    RMS_R_average = float(np.average(List_Radius[:, 1] * 1000))

    # EE (PDF)
    ee_pdf = os.path.join(ee_dir, f"EE_{save_prefix}_fields.pdf")
    ee_name = f"EE_{save_prefix}_fields.pdf"
    EE_Example_information = plot_all_EE_for_fields(
    Pup, Rays, Field_ccd, RW,
    show_r50=True,
    save=True,                 
    save_dir=ee_dir,           
    filename=ee_name)           

    # Set_Initial_Opt (TXT)
    set_txt = os.path.join(results_dir, f"Optical_Parameters_{save_prefix}.txt")
    
    with open(set_txt, "w", encoding="utf-8") as f:
        f.write("Optical_Parameters = [\n")
        for v in Set_Initial_Opt:
            f.write(f"  {v:.6f},\n")
        f.write("]\n")

    # Métricas (TXT)
    metrics_txt = os.path.join(results_dir, f"metrics_{save_prefix}.txt")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"L1={glass_a}  L2={glass_b}\n")
        f.write("Final Seidel Aberrations:\n")
        f.write(
        f"  Spherical = {Sph:.2f}\n"
        f"  Coma      = {Coma:.2f}\n"
        f"  Astig     = {Ast:.2f}\n"
        f"  CLon      = {CLon:.2f}\n"
        )
        f.write(f"EFFL = {Telescope_f85_FR.EFFL:.6f} mm\n")
        f.write(f"MS_d = {MS_d:.6g}\n")
        f.write("Airy's disk comparison with GEO and RMS:\n")
        f.write(
            f"  GEO/Airy = {(GEO_R_average / (Rairy * 1000)):.2f}\n"
            f"  RMS/Airy = {(RMS_R_average / (Rairy * 1000)):.2f}\n"
           )
        f.write(f"GEO_R_avg_mm = {GEO_R_average:.6f}\n")
        f.write(f"RMS_R_avg_mm = {RMS_R_average:.6f}\n")
        f.write(f"EE_info = {EE_Example_information}\n")

  
    return {
        "L1": str(glass_a),
        "L2": str(glass_b),
        "Optical_Parameters": Set_Initial_Opt,
        "EFFL": float(Telescope_f85_FR.EFFL),
        "GEO_R_avg_mm": GEO_R_average,
        "RMS_R_avg_mm": RMS_R_average,
        "EE_info": EE_Example_information,
        "files": {
            "set_txt": set_txt,
            "ee_pdf": ee_pdf,
            "metrics_txt": metrics_txt,
        },
    }