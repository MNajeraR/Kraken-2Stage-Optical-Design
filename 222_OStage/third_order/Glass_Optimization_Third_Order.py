



import KrakenOS as Kos

from utils import  analyze_ranked

from Third_Order_Focal_Reducer_Glass_Opt import run_glasses


configuracion_1 = Kos.Setup()
glasses_2use = ["S-FPL51", "F2HT"]
glass_catalog = configuracion_1

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

# pairs = [("S-FPL51", "F2HT"), ("N-FK51A", "SF5")]
    
for i in range(len(Vidrios_lista_1)):
    for j in range(len(Vidrios_lista_2)):
        print('')
        print(Vidrios_lista_1[i],Vidrios_lista_2[j])
        out = run_glasses(Vidrios_lista_1[i], Vidrios_lista_2[j]) 
        if out.get("skipped") or "files" not in out:
            print(f"[skip]: {out.get('error')}")
            continue
        print(out["files"]["set_txt"], "-> ok")