# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 18:17:51 2025

@author: MORGANRHAINAJERAROA
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scipy.optimize import least_squares
import numpy as np
from utils import (matrix_refraction, matrix_trasmition)
from time import perf_counter

b_1 = -1 / (1.118E004 / 2)       # Optical power of the primary mirror
b_2 = -1 / (-4430. / 2)          # Optical power of the secondary mirror

# -------------------------------------
#    Distance Definitions
# -------------------------------------
d_1 = 4052.571043                # Distance between primary and secondary mirrors
d_2 = 605.3156776092883                       # Distance to the focal reducer
d_3 = 2.706860254305352E+001     # Distance for optical correction
d_4 = 13.86                      # Distance to the final image plane
d_5 = 107.56555422
d_Obj = 5.05257104e+03                     # Hypothetical object distance


M6 = np.array([1, 0, d_1 + d_2, 1]).reshape(2, 2)
M7 = np.array([1, b_2, 0, 1]).reshape(2, 2)
M8 = np.array([1, 0, d_1, 1]).reshape(2, 2)
M9 = np.array([1, b_1, 0, 1]).reshape(2, 2)
M10 = np.array([1, 0, d_Obj, 1]).reshape(2, 2)
# M10 = np.array([1, 0, d_Obj, 1]).reshape(2, 2)

# Compute the total ABCD matrix for the telescope system
MSJS = M6 @ M7 @ M8 @ M9 @ M10

b_t = -1 / 9127.198583362275 

Glass = 'K-PFK85 & ADF355 & K-PFK85'
#Glass = 'S-FPL51 & F2HT & S-FPL51'
Canal =  3

if Glass ==  'K-PFK85 & ADF355 & K-PFK85':
    ## Glass K-PFK85 & ADF355 & K-PFK85
    if Canal == 1:
        ### Canal 1 ###
        ### index of refraction ###
        n_L1  = 1.4930681817412375
        n_L2  = 1.6658229057857488
        ### bendings ###
        beta1, beta2, beta3 = 6., -0.5, -3.
    elif Canal == 2:
        ### Canal 2 ###
        ### index of refraction ###
        n_L1  = 1.4851387915076248
        n_L2  = 1.6431607438547413
        ### bendings ###
        beta1, beta2, beta3 = 8.5, -0.5, -3.
    elif Canal == 3:
        ### Canal 3 ###
        ### index of refraction ###
        n_L1  = 1.4813807774635017
        n_L2  = 1.6331091174536438
        ### bendings ###
        beta1, beta2, beta3 = 11.75, -0.5, -3.
    
elif Glass == 'S-FPL51 & F2HT & S-FPL51':
    ## Glass S-FPL51 & F2HT & S-FPL51
    if Canal == 1:
        ### Canal 1 ###
        ### index of refraction ###
        n_L1  = 1.504939667917171
        n_L2  = 1.6433733708672797
        ### bendings ###
        beta1, beta2, beta3 = 3.10, -0.5, -3.
        # beta1, beta2, beta3 = 2.75, -0.5, -3.
    if Canal == 2:
        ## Canal 2 ###
        ### index of refraction ###
        n_L1  = 1.49647373690684
        n_L2  = 1.618607772295224
        ### bendings ###
        beta1, beta2, beta3 = 5.5, -0.5, -3.
    if Canal == 3:
        ### Canal 3 ###
        ### index of refraction ###
        n_L1  = 1.4924304236621844
        n_L2  = 1.6081550533016609
        ### bendings ###
        beta1, beta2, beta3 = 13., -0.5, -3.
    
    
    

def R_from_phi_closed(phi, n, d, beta, sign_pattern=('+','+')):
    """
    Devuelve (R1, R2) dados: potencia phi, índice n, espesor d, y beta=R2/R1 (con signo).
    sign_pattern define la forma deseada: ('+','+'), ('-','+'), ('+','-'), ('-','-').
    """
    # signos objetivo
    s1 = +1 if sign_pattern[0] == '+' else -1
    s2 = +1 if sign_pattern[1] == '+' else -1

    # Si los signos objetivo son opuestos, beta debe ser NEGATIVO; si son iguales, POSITIVO.
    if s1 * s2 < 0 and beta > 0:
        beta = -beta
    if s1 * s2 > 0 and beta < 0:
        beta = -beta

    k = n - 1.0
    # Coeficientes de la cuadrática en x = 1/R1
    a = (k*k * d) / (n * beta)           # puede ser negativo si beta<0
    b = k * (1.0 - 1.0/beta)
    # Ecuación: a x^2 + b x - phi = 0

    # Caso lente delgada (a ~ 0): resolver lineal
    if abs(a) < 1e-18:
        denom = b
        if abs(denom) < 1e-18:
            raise ValueError("Degenerado: (1 - 1/beta) ~ 0 y a ~ 0; elige otro beta o d.")
        x = phi / denom
        R1 = 1.0 / x
        R2 = beta * R1
        # fuerza signos
        if np.sign(R1) != s1:
            R1 = -R1
            R2 = -R2
        if np.sign(R2) != s2:
            R1 = -R1
            R2 = -R2
        return float(R1), float(R2)

    disc = b*b + 4.0*a*phi
    if disc < 0:
        # por numeric safety: clamp a 0 si es pequeñamente negativo
        if disc > -1e-24:
            disc = 0.0
        else:
            raise ValueError(f"Discriminante negativo: {disc}. Revisa (phi,n,d,beta).")

    sqrt_disc = np.sqrt(disc)

    # Dos raíces para x=1/R1
    x1 = (-b + sqrt_disc) / (2.0*a)
    x2 = (-b - sqrt_disc) / (2.0*a)

    # Candidatos de R1
    R1_candidates = []
    
    for x in (x1, x2):
        if abs(x) < 1e-24:
            continue
        R1 = 1.0 / x
        R2 = beta * R1
        R1_candidates.append((R1, R2))

    if not R1_candidates:
        raise ValueError("No se obtuvieron candidatos válidos para R1.")

    # Selección: el que coincide con los signos deseados y con |R| razonables
    def score(R1, R2):
        s = 0.0
        s += 0.0 if np.sign(R1) == s1 else 1e3
        s += 0.0 if np.sign(R2) == s2 else 1e3
        # preferir magnitudes en rango 50..2000 mm
        s += (abs(R1) < 50 or abs(R1) > 5000) * 10.0
        s += (abs(R2) < 50 or abs(R2) > 5000) * 10.0
        return s

    R1_best, R2_best = min(R1_candidates, key=lambda t: score(*t))

    # Si aún no coincide el signo deseado, corrige invirtiendo ambos
    if np.sign(R1_best) != s1 or np.sign(R2_best) != s2:
        R1_best, R2_best = -R1_best, -R2_best

    return float(R1_best), float(R2_best)

def Complete_Matrix(v):
    
    Phi_1, Phi_2, Phi_3 = v
    
    Thhf_mar1 = np.array([[0.0], [1076.00]])
    
    R1, R2 = R_from_phi_closed(Phi_1, n_L1, 25., beta1, ('+','+'))
    R3, R4 = R_from_phi_closed(Phi_2, n_L2,  9., beta2, ('-','+'))
    R5, R6 = R_from_phi_closed(Phi_3, n_L1, 16., beta3, ('+','-'))
    
    M_TR3 = matrix_trasmition(68.00635914573589)
    
    M_L3b = matrix_refraction(n_L1, 1., R6)
    M_SL3 = matrix_trasmition(16.)
    M_L3a = matrix_refraction(1., n_L1, R5)
    
    M_TR2 = matrix_trasmition(13.86)
    
    M_L2b = matrix_refraction(n_L2, 1., R4)
    M_SL2 = matrix_trasmition(9.)
    M_L2a = matrix_refraction(1., n_L2, R3)
    
    M_TR1 = matrix_trasmition(2.706860254305352E+001)
    
    M_L1b = matrix_refraction(n_L1, 1., R2)
    M_SL1 = matrix_trasmition(25.)
    M_L1a = matrix_refraction(1., n_L1, R1)
    
    M_L3 = M_L3b@M_SL3@M_L3a
    M_L2 = M_L2b@M_SL2@M_L2a
    M_L1 = M_L1b@M_SL1@M_L1a
    
    M_T = M_TR3@M_L3b@M_SL3@M_L3a@M_TR2@M_L2b@M_SL2@M_L2a@M_TR1@M_L1b@M_SL1@M_L1a@MSJS
    
    b_3 = M_L1[0,1]
    b_4 = M_L2[0,1]
    b_5 = M_L3[0,1]
    
    exit_1 = MSJS @ Thhf_mar1
    exit_2 = M_TR1 @ M_L1 @ exit_1
    exit_3 = M_TR2 @ M_L2 @ exit_2
    
    M_RF = M_TR3 @ M_L3 @ M_TR2 @ M_L2 @ M_TR1 @ M_L1
    b_RF = M_RF[0, 1]
    
    exit_1_h = float(exit_1[1, 0])
    exit_2_h = float(exit_2[1, 0])
    exit_3_h = float(exit_3[1, 0])
    
    diff_H = exit_1_h*b_3 + exit_2_h*b_4 + exit_3_h*b_5 - exit_1_h*b_RF
    
    diff_P3 = float(M_L3[0, 1]/n_L1 + M_L2[0, 1]/n_L2 + M_L1[0, 1]/n_L1)
    b = float(np.abs(-1/M_T[0, 1] - (-1/b_t)))
    d = float(M_T[1, 1])
        
    return [b, diff_P3, diff_H, d]

bounds = ([0.00111, -0.0044, 0.00131],    # límites inferiores
          [0.00444, -0.00133, 0.0044])    # límites superiores

initial_guess = [0.0033, -0.003, 0.003]  # dentro de bounds y con signos correctos
start = perf_counter()
# Least Squares Optimization to solve paraxial equations
B_sistem = least_squares(Complete_Matrix, initial_guess, bounds=bounds, 
                                        verbose=0, ftol = 1e-12)
[b_3, b_4, b_5] = B_sistem.x

elapsed_time = perf_counter() - start
print(f"Optimization time: {elapsed_time:.3f} s")
# Display the solution
print("Power of lenses numerical Solution:", b_3, b_4, b_5)

R1, R2 = R_from_phi_closed(b_3, n_L1, 22.5, beta1, ('+','+'))
R3, R4 = R_from_phi_closed(b_4, n_L2,  9., beta2, ('-','+'))
R5, R6 = R_from_phi_closed(b_5, n_L1, 16., beta3, ('+','-'))

# First-order parameter list
First_Order_Parameters_Glass_ChCanal = [
    R1, R2, R3, R4, R5, R6
]

# Output directory
output_dir = PROJECT_ROOT / "optimized_parameters"
output_dir.mkdir(parents=True, exist_ok=True)

# Clean glass name for filename
glass_tag = Glass.replace(" ", "").replace("&", "_")

# Output file path
file_path = output_dir / f"First_Order_Parameters_{glass_tag}_Ch{Canal}.txt"

# Write file
with open(file_path, "w", encoding="utf-8") as f:
    f.write("# First-order parameters\n")
    f.write(f"# Glass : {Glass}\n")
    f.write(f"# Channel : {Canal}\n\n")
    f.write("First_Order_Parameters = [\n")
    for value in First_Order_Parameters_Glass_ChCanal:
        f.write(f"    {value:.12f},\n")
    f.write("]\n")

print(f"[ok] Archivo '{file_path}' guardado con los valores actuales.")

