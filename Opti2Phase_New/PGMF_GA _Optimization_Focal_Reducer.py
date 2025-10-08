# -*- coding: utf-8 -*-
"""
======================================
  Script: Genetic Algorithm Optimization
======================================

@author: MORGANRHAINAJERAROA

Description:
------------
This script implements an optimization routine for an optical system using KrakenOS 
and a Genetic Algorithm (GA). The optimization starts from an SPD defined through 
first- and third-order analysis and aims to refine surface curvatures and lens 
separation based on a physically grounded merit function. The fitness evaluation 
is based on principles such as Fermat’s principle, transverse coma cancellation, 
Coddington equations, and chromatic path equalization.

A GA is used to explore the solution space globally. The script configures the 
population, mutation, crossover, and selection strategies to drive the convergence 
toward optimal parameters. Final performance is 
evaluated through metrics such as GEO radius, RMS radius, EFFL deviation, 
convergence time  and spot diagram visualization.

Dependencies:
-------------
- NumPy: For numerical operations.
- time: For performance timing.
- pygad: Genetic algorithm framework.
- KrakenOS (Kos): Optical simulation environment.
- MOS_Class: Custom module containing the merit function, aberration 
             evaluation methods, and the `Optimizer` class.
- MOS_equation: Custom module for ray tracing routines and spot 
                diagram visualization.
"""
# ===============================
#      Library Imports
# ===============================
import time
import numpy as np
import pkg_resources

# Import KrakenOS and custom modules
from utils import MOS_Class as MOSCL
from utils import MOS_equation as Meq

# Check if KrakenOS is installed, otherwise append its path
required = {'KrakenOS'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("KrakenOS not installed")
    import sys
    sys.path.append("../..")

import KrakenOS as Kos
import pygad

# ===============================
#    Constants Definition
# ===============================

per_red = 27.879562
Prx_data = MOSCL.Paraxial_Cal(per_red)

d_1 = Prx_data.d_1
d_2 = 4674.568305701443
d_3 = 27.792111342523572

# First order    
d_lens1 = 13.584 
d_lens2 = 9.878

F_Diameter = 104.491 
S_Diameter = 75.988 

# Third Order
R_1 = 239.06104309576506
R_2 = -101987.90085962816
R_3 = 9998.050905262751
R_4 = 495.20317019855105

d_4 = 186.6409959040964

# Calculate the field angle for the CCD
ccd_co = 16.1
Field_ccd = ccd_co / Prx_data.EFFL_Tr

# Initialize initial height 
h_i  = 1076.00


# ______________________________________#

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
L1a.Rc = R_1
L1a.Thickness = d_lens1
L1a.Glass = "S-FPL51"
L1a.Diameter = F_Diameter

# ______________________________________#

# Lens L1b configuration

L1b = Kos.surf()
L1b.Rc = R_2
L1b.Thickness = d_3
L1b.Glass = "AIR"
L1b.Diameter = F_Diameter

# ______________________________________#

# Lens L2a configuration

L2a = Kos.surf()
L2a.Rc = R_3
L2a.Thickness = d_lens2
L2a.Glass = "F2HT"
L2a.Diameter = S_Diameter + 0.1*S_Diameter

# ______________________________________#

# Lens L2b configuration

L2b = Kos.surf()
L2b.Rc = R_4
L2b.Thickness = d_4
L2b.Glass = "AIR"
L2b.Diameter = S_Diameter + 0.1*S_Diameter

# ______________________________________#

# ======================================
#  Image Plane Initialization
# ======================================

P_Ima = Kos.surf()
P_Ima.Rc = 0.0
P_Ima.Thickness = 0.0
P_Ima.Glass = "AIR"
P_Ima.Diameter = 100.0
P_Ima.Name = "Plano imagen"

#______________________________________#

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


# ======================================
#  Pupil Calculation Setup
# ======================================

# Define the parameters for the pupil calculation:
W = 0.43032015             # Reference wavelength in micrometers
sup = 1                    # Number of the surface representing the opening of the system
AperType = "EPD"           # AperType sets the aperture ("STOP") or entrance pupil diameter ("EPD").
AperVal = M1.Diameter      # Diameter of the entrance pupil

# Initialize the Pupil calculation for the system
Pup = Kos.PupilCalc(Telescope_f85_FR , sup, W, AperType, AperVal)

# Configure the sampling for the pupil and its field representation:
Pup.Samp = 7              # Integer number for pupil ray sampling
Pup.FieldType = "angle"   # Field type, this in terms of object height and distance from the plane.
Pup.FieldX = np.rad2deg(Field_ccd) # Field value in degrees on the X-axis


# Third Order EFFL
EFFL_3O = Telescope_f85_FR.EFFL 

# ======================================
#    Aberration Analysis Setup
# ======================================
# Initializes the optical system for aberration analysis
InfSystem = [Telescope_f85_FR, Rays, Pup]
Aberration = MOSCL.Aberration_Info(InfSystem, W)

# Set design wavelengths for chromatic analysis
Aberration.dw_1 = 0.35
Aberration.dw_2 = 0.5499996

# Display surface radii

print("\nSurface Radii Configuration and working distance:")
print(f"R1: {R_1:.2f} mm")
print(f"R2: {R_2:.2f} mm")
print(f"R3: {R_3:.2f} mm")
print(f"R4: {R_4:.2f} mm")
print(f"Working distance (d4): {d_4:.2f} mm")
print(" ")

# ======================================
#  Display Initial Values
# ====================================== 

print("======================================")
print("Initial Aberrations and EFFL:")
print(f"Chromatic Aberration: {Aberration.Chromatic(1, [0.0, 0.0])[1]:.2f}")
print(f"Spherical Aberration: {Aberration.Spheric(1, [0.0, 0.0])[1]:.2f}")
print(f"Coma Aberration: {Aberration.Coma(1, Field_ccd)[1]:.2f}")
print(f"Astigmatism Aberration: {Aberration.Astigmatism(1, Field_ccd)[1]:.2f}")
print(f"Effective Focal Length: {EFFL_3O:.2f}")
print("======================================")
print('')

# ======================================
#  Genetic Algorithm Optimization Process
# ======================================

print("\nStarting Genetic Algorithm Optimization...")
start_time = time.time()

# Initialize the Optimizer class with system and ray information
GA_Result = MOSCL.Optimizer([Telescope_f85_FR, Rays])
GA_Result.Field = Field_ccd
GA_Result.ApVal = M1.Diameter

# Desired EFFL
EFFL_Des = GA_Result.EFFL_Tr

# Define the target output for the merit function (fitness is based on proximity to this value)
desired_output = 10

# Fitness function: evaluates how close a solution is to the desired merit value.
def fitness_func(ga_instance, solution, solution_idx):
    output = GA_Result.Set_RcValues(solution)
    fitness = 1.0 / np.abs(output - desired_output)
    return fitness

fitness_function = fitness_func

# Genetic algorithm parameters
num_generations = 100                  # Number of generations to evolve
num_parents_mating = 2                # Number of parents selected for mating
sol_per_pop = 50                      # Number of individuals per population
num_genes = 5                         # Number of variables (4 radii + 1 thickness)

# Define the search limits for each variable
Delta_search_Rc = 0.25
Delta_search_d = 2
limits = [
    {'low': R_1 - Delta_search_Rc, 'high': R_1 + Delta_search_Rc},
    {'low': R_2 - Delta_search_Rc, 'high': R_2 + Delta_search_Rc},
    {'low': R_3 - Delta_search_Rc, 'high': R_3 + Delta_search_Rc},
    {'low': R_4 - Delta_search_Rc, 'high': R_4 + Delta_search_Rc},
    {'low': d_4 - Delta_search_d,  'high': d_4 + Delta_search_d}
]

# Genetic operators and configuration
parent_selection_type = "sss"          # Steady-state selection
keep_parents = 2                       # Number of parents to retain for next generation
crossover_type = "uniform"             # Crossover strategy
mutation_type = "random"              # Mutation strategy
mutation_percent_genes = 10           # Percentage of genes to mutate

# Function to monitor generation-wise progress
last_fitness = 1.0
def on_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    print("Change     = {change}".format(
        change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

# Create and run the genetic algorithm instance
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    gene_space=limits,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes
)

ga_instance.run()
# Optionally plot the convergence curve
# ga_instance.plot_fitness()

# Extract the best solution from the final population
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
R1, R2, R3, R4, d4 = solution
elapsed_time = time.time() - start_time


# ======================================
#  Display Optimization Results
# ======================================

print("======================================")
print("Optimization Complete")
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
print("Optimized Parameters:")
print(f"R1: {R1:.2f} mm")
print(f"R2: {R2:.2f} mm")
print(f"R3: {R3:.2f} mm")
print(f"R4: {R4:.2f} mm")
print(f"Working distance (d4): {d4:.2f} mm")
print("======================================")

print("======================================")
print("Comparison with Initial Parameters:")
print(f"ΔR1: {np.abs(R1 - R_1):.2f} mm")
print(f"ΔR2: {np.abs(R2 - R_2):.2f} mm")
print(f"ΔR3: {np.abs(R3 - R_3):.2f} mm")
print(f"ΔR4: {np.abs(R4 - R_4):.2f} mm")
print(f"ΔWorking distance (d4): {np.abs(d4 - d_4):.2f} mm")
print("======================================")

# Update the optical system
Telescope_f85_FR.SDT[3].Rc = R1
Telescope_f85_FR.SDT[4].Rc = R2
Telescope_f85_FR.SDT[5].Rc = R3
Telescope_f85_FR.SDT[6].Rc = R4

Telescope_f85_FR.SDT[6].Thickness = d4

Telescope_f85_FR.SetData()
Telescope_f85_FR.SetSolid()


#Calculates the Airy disk radius and its coordinates for plotting.
    
#Parameters:
# W: # Reference wavelength
# Telescope_f85_FR.EFFL: Effective Focal Length
# M1.Diameter: Diameter of the primary mirror

Rairy = (1.22 * (Aberration.dw_2 / 1000) * Telescope_f85_FR.EFFL) / (M1.Diameter)
num_segmentos = 100
angulo = np.linspace(0, 2 * np.pi, num_segmentos + 1)
xairy = Rairy * np.cos(angulo)
yairy = Rairy * np.sin(angulo)


# ======================================
#    Plot Generation for All Fields
# ======================================

fields = [
    (0.0, 0.0, 'Field_0'),
    (np.rad2deg(Field_ccd), 0.0, 'Field_1'),
    (0.0, -np.rad2deg(Field_ccd), 'Field_2'),
    (0.035, -0.035, 'Field_3')
]

wavelengths = [Aberration.dw_1, W, Aberration.dw_2]

List_Radius = [] 

for fx, fy, name in fields:
    
    # Configure the pattern sampling for the pupil and fields:
    Pup.Ptype = "hexapolar"     
    Pup.FieldX = fx
    Pup.FieldY = fy
    
    # Perform ray tracing for all wavelengths
    traced_data = Meq.trace_rays(Pup, wavelengths, Rays)
    
    # Extract the data
    (Xa, Ya, Za, La, Ma, Na), (Xb, Yb, Zb, Lb, Mb, Nb), (Xc, Yc, Zc, Lc, Mc, Nc) = traced_data
    
    # Plot the diagram
    geo_r, rms_r = Meq.plot_spot_diagram(Xa, Ya, Xb, Yb, Xc, Yc, xairy, yairy, name, "GA_Method")
    
    # Append to the list of radii
    List_Radius.append((geo_r, rms_r))

# Convert to NumPy array for easy manipulation
List_Radius = np.array(List_Radius)

# Geo_radius in mm 
GEO_R = List_Radius[:, 0] * 1000
GEO_R_average = np.average(GEO_R)

# RMS_radius in mm 
RMS_R = List_Radius[:, 1] * 1000
RMS_R_average = np.average(RMS_R)


# ======================================
#    Plot for All the System
# ======================================

# ------------------------------------------------
# Field Configuration for Hexapolar Sampling
# ------------------------------------------------

field_configs = [
    (np.rad2deg(Field_ccd), 0.0, "hexapolar", 1),
    (0.0, 0.0, "hexapolar", 0),
    (-np.rad2deg(Field_ccd), 0.0, "hexapolar", 0)
]

traced_fields = []

for fx, fy, pattern, clean in field_configs:
    x, y, z, L, M, N = Meq.configure_and_trace(Pup, fx, fy, pattern, W, clean)
    Kos.TraceLoop(x * 0.99, y * 0.99, z * 0.99, 
                  L * 0.99, M * 0.99, N * 0.99, 
                  W, Rays, clean)
    traced_fields.append((x, y, z, L, M, N))

# ------------------------------------------------
# Display the 2D Optical System Layout
# ------------------------------------------------
Kos.display2d(Telescope_f85_FR, Rays, 1, arrow=0)

# Reinitialize the Pupil calculation
W = 0.43032015 
Pup = Kos.PupilCalc(Telescope_f85_FR, sup, W, AperType, AperVal)

# ======================================
#  Display Final Results
# ======================================

print("======================================")
print("\nFinal Aberration Analysis:")
print(f"Chromatic Aberration: {Aberration.Chromatic(1, [0.0, 0.0])[1]:.2f}")
print(f"Spherical Aberration: {Aberration.Spheric(1, [0.0, 0.0])[1]:.2f}")
print(f"Coma Aberration: {Aberration.Coma(1, Field_ccd)[1]:.2f}")
print(f"Astigmatism Aberration: {Aberration.Astigmatism(1, Field_ccd)[1]:.2f}")
print(f"Effective Focal Length: {Telescope_f85_FR.EFFL:.2f} mm")
print(f"EFFL deviation to Third Order's EFFL: {np.abs(Telescope_f85_FR.EFFL - EFFL_3O):.2f} mm")
print(f"EFFL deviation to Target: {np.abs(Telescope_f85_FR.EFFL - EFFL_Des):.2f} mm")
print(f"Airy disk radius: {Rairy * 1000:.2f}")
print(f"Average GEO radius: {GEO_R_average:.2f}")
print(f"Average RMS radius: {RMS_R_average:.2f}")
print(f"Airy's disk comparison with GEO and RMS: "
      f"GEO/Airy = {(GEO_R_average / (Rairy * 1000)):.2f}, "
      f"RMS/Airy = {(RMS_R_average / (Rairy * 1000)):.2f}")
print("======================================")
print('')



