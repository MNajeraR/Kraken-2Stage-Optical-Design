# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:51:48 2025

@author: MORGANRHAINAJERAROA
"""
import csv

#########################################################################################################

def guardar_lista_en_csv(lista, nombre_archivo):
    
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        
        writer = csv.writer(archivo_csv)
        writer.writerow(lista)
        
    print(f"La lista se ha guardado en el archivo {nombre_archivo}.")


#########################################################################################################
                
def data_exists(data):
    
    with open('aberration_data.csv', 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row == data:
                return True
    return False