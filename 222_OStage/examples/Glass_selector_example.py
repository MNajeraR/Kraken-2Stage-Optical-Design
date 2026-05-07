#!/usr/bin/env python3

# -*- coding: utf-8 -*-

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pkg_resources


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

from utils import  Glass_Selector, analyze_ranked

        
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

L1b.Glass = "PBM18Y"

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

Doblet = Kos.system(A, configuracion_1) 


# glasses_2use = ["K-PFK85", "ADF355"]
glasses_2use = ["S-FPL51", "F2HT"]
glass_catalog = configuracion_1

Glass_S1 = Glass_Selector(configuracion_1,
    WR = [0.35, 1],
    name_glass = glasses_2use[0],
    delta_n =  0.025,
    delta_vd = 5.0,
    pt_threshold = 0.8
    ) 
Glass_S1.plot_nv_with_inset()

Glass_S2 = Glass_Selector(configuracion_1,
    WR = [0.35, 1],
    name_glass = glasses_2use[1],
    delta_n =  0.025,
    delta_vd = 5.0,
    pt_threshold = 0.8
    )

Glass_S2.plot_nv_with_inset()

lista = analyze_ranked(
    glass_catalog,
    glasses_2use,
    WR=(0.35, 1.0),
    delta_n=0.025,
    delta_vd=5.0,
    pt_threshold=0.8,
    prefer_set="possible",             
)


Vidrios_lista_1 = lista[glasses_2use[0]]['rank_names']
Vidrios_lista_2 = lista[glasses_2use[1]]['rank_names']

for i in range (len(Vidrios_lista_2)):
    print(i, Vidrios_lista_1[i], Vidrios_lista_2[i])
    print('')

















