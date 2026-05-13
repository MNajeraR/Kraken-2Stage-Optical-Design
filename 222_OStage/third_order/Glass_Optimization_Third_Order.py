
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import KrakenOS as Kos

from utils import analyze_ranked
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

Vidrios_lista_1 = lista[glasses_2use[0]]["rank_names"]
Vidrios_lista_2 = lista[glasses_2use[1]]["rank_names"]

batch_results = []

for i in range(len(Vidrios_lista_1)):
    for j in range(len(Vidrios_lista_2)):

        glass_1 = Vidrios_lista_1[i]
        glass_2 = Vidrios_lista_2[j]

        print("")
        print(glass_1, glass_2)

        out = run_glasses(glass_1, glass_2)
        batch_results.append(out)

        if out.get("skipped") or "files" not in out:
            print(f"[skip]: {out.get('error')}")
            continue

        print(out["files"]["set_txt"], "-> ok")

summary_path = RESULTS_DIR / "glass_optimization_batch_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(batch_results, f, indent=4, ensure_ascii=False, default=str)

print(f"[ok] Batch summary saved to: {summary_path}")