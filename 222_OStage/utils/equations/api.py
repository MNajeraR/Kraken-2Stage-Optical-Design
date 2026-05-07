# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 12:31:42 2025

@author: MORGANRHAINAJERAROA
"""

from .ops import (Set_Initial_Radius, BestFocus, calculate_average, Min_Dis_point,
                  R_RMS_delta)  
from .MOS_equation import (BestRMS, Sphere_subtract, Astigmatism_Substraction,
                          Coma_Substraction)      
from .SPT_plotting import plot_spot_diagram, run_spots_for_fields, calculate_geometrical_center, calculate_radius
from .EE_plotting import plot_all_EE_for_fields
from .tracing import (configure_and_trace, trace_rays, ProcessPattern2Field)
from .generating import (generate_radios, generate_angles, generate_radiosc,
                        generate_anglesc, generate_radiosa, generate_anglesa)
from .opt_ecuation import Lens_Maker, Prin_Plane, seidel_terms, airy_data, configure_pupil_and_ab, seidel_terms_20, matrix_trasmition, matrix_refraction
from .Actualization_of_system import apply_system_actualization
from .glass_tools import analyze_ranked, set_pair_glass
from .actualization_of_system_3L import apply_system_actualization_3L




__all__ = [
    "Set_Initial_Radius", "calculate_average", "Min_Dis_point", "R_RMS_delta",
    "BestFocus", "Sphere_subtract", "Astigmatism_Substraction",
    "Coma_Substraction", "BestRMS", "plot_spot_diagram", "configure_and_trace", 
    "ProcessPattern2Field", 'analyze_ranked', "set_pair_glass",
    "trace_rays", "generate_radios", "generate_angles", "generate_radiosc",
    "generate_anglesc", "generate_radiosa", "generate_anglesa",
    "Lens_Maker",  "Prin_Plane", "apply_system_actualization",
    "run_spots_for_fields", "seidel_terms", "airy_data", "configure_pupil_and_ab",
    "plot_all_EE_for_fields", "calculate_geometrical_center", 
    "calculate_radius", "apply_system_actualization_3L", "seidel_terms_20", "matrix_refraction",
    "matrix_trasmition"
]