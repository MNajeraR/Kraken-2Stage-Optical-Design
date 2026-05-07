
import re
from pathlib import Path

# -------------------------
# Regex patterns (compiled)
# -------------------------
RE_LENSES   = re.compile(r"L1\s*=\s*([A-Za-z0-9\-\._]+)\s+L2\s*=\s*([A-Za-z0-9\-\._]+)")
RE_SEIDEL   = re.compile(r"Final Seidel Aberrations:\s*"
                         r"Spherical\s*=\s*([\-+eE0-9\.]+)\s*"
                         r"Coma\s*=\s*([\-+eE0-9\.]+)\s*"
                         r"Astig\s*=\s*([\-+eE0-9\.]+)\s*"
                         r"CLon\s*=\s*([\-+eE0-9\.]+)", re.DOTALL)
RE_EFFL     = re.compile(r"EFFL\s*=\s*([\-+eE0-9\.]+)\s*mm")
RE_MS_D     = re.compile(r"MS_d\s*=\s*([\-+eE0-9\.]+)")
RE_AIRY     = re.compile(r"Airy's disk.*?GEO/Airy\s*=\s*([\-+eE0-9\.]+)\s*"
                         r"RMS/Airy\s*=\s*([\-+eE0-9\.]+)", re.DOTALL)
RE_GEO_R    = re.compile(r"GEO_R_avg_mm\s*=\s*([\-+eE0-9\.]+)")
RE_RMS_R    = re.compile(r"RMS_R_avg_mm\s*=\s*([\-+eE0-9\.]+)")
RE_EE_INFO  = re.compile(r"EE_info\s*=\s*\[([^\]]+)\]")

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def parse_metrics_text(text: str) -> dict:
    out = {
        "L1": None, "L2": None,
        "Spherical": None, "Coma": None, "Astig": None, "CLon": None,
        "EFFL_mm": None, "MS_d": None,
        "GEO_over_Airy": None, "RMS_over_Airy": None,
        "GEO_R_avg_mm": None, "RMS_R_avg_mm": None,
        "EE_info": None
    }

    m = RE_LENSES.search(text)
    if m:
        out["L1"], out["L2"] = m.group(1), m.group(2)

    m = RE_SEIDEL.search(text)
    if m:
        out["Spherical"] = _to_float(m.group(1))
        out["Coma"]      = _to_float(m.group(2))
        out["Astig"]     = _to_float(m.group(3))
        out["CLon"]      = _to_float(m.group(4))

    m = RE_EFFL.search(text)
    if m:
        out["EFFL_mm"] = _to_float(m.group(1))

    m = RE_MS_D.search(text)
    if m:
        out["MS_d"] = _to_float(m.group(1))

    m = RE_AIRY.search(text)
    if m:
        out["GEO_over_Airy"] = _to_float(m.group(1))
        out["RMS_over_Airy"] = _to_float(m.group(2))

    m = RE_GEO_R.search(text)
    if m:
        out["GEO_R_avg_mm"] = _to_float(m.group(1))

    m = RE_RMS_R.search(text)
    if m:
        out["RMS_R_avg_mm"] = _to_float(m.group(1))

    m = RE_EE_INFO.search(text)
    if m:
        nums = [n.strip() for n in m.group(1).split(",")]
        out["EE_info"] = [_to_float(n) for n in nums if n.strip()]

    return out

def parse_metrics_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    data = parse_metrics_text(text)
    data["source_file"] = path.name
    data["source_path"] = str(path)
    return data

def main(folder="Results", recursive=False, expand_ee=True, save_csv=True, csv_name="metrics_consolidado.csv"):
    folder_path = Path(folder)
    files = sorted(folder_path.rglob("metrics_*.txt") if recursive else folder_path.glob("metrics_*.txt"))
    print(f"[INFO] Carpeta: {folder_path.resolve()}")
    print(f"[INFO] Archivos encontrados: {len(files)}")

    if not files:
        print("[WARN] No se encontraron archivos 'metrics_*.txt'.")
        return 0


    # Parsear todos
    rows = [parse_metrics_file(p) for p in files]
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        # Expansión opcional de EE_info
        if expand_ee:
            ee_lists = df.get("EE_info")
            if ee_lists is not None:
                max_len = max((len(lst) for lst in ee_lists if isinstance(lst, list)), default=0)
                for i in range(max_len):
                    df[f"EE{i+1}"] = df["EE_info"].apply(lambda lst: lst[i] if isinstance(lst, list) and len(lst) > i else None)
        if save_csv:
            out_csv = folder_path / csv_name
            df.to_csv(out_csv, index=False)
            print(f"\n[OK] CSV guardado en: {out_csv}")
        return 0
    except ModuleNotFoundError:
        print("\n[NOTE] pandas no está instalado. Imprimiendo los primeros 3 dicts parseados:")
        for r in rows[:3]:
            print(r)
        print("\nPara guardar a CSV instala pandas:  pip install pandas")
        return 0

if __name__ == "__main__":
    # Ajusta los flags según necesites
    main(folder="Results", recursive=False, expand_ee=True, save_csv=True, csv_name="metrics_consolidado.csv")
    
import pandas as pd
df = pd.read_csv("Results/metrics_consolidado.csv")

df["Total_Seidel"] = (
    df["Spherical"]**2 +
    df["Coma"]**2 +
    df["Astig"]**2 +
    df["CLon"]**2
)

df["EE50_mean"] = df[["EE1","EE2","EE3","EE4"]].mean(axis=1)

df["EE50_std"] = df[["EE1","EE2","EE3","EE4"]].std(axis=1)
df["EE50_range"] = df[["EE1","EE2","EE3","EE4"]].max(axis=1) - df[["EE1","EE2","EE3","EE4"]].min(axis=1)


# # Acceder a columnas específicas
# print(df[["L1","L2","Total_Seidel", "EE50_mean", "EE50_std","EE50_range", "EE50_weighted"]].head())


# Ordenar por la métrica que te interese (ej. menor RMS/Airy)
top = df.sort_values("EE50_mean", ascending=True)
print(top[["L1","L2","EE50_mean"]].head(5))



# # Orden combinado: minimizar Total_Seidel, minimizar EE50_mean
# ranked = df.sort_values(
#     by=["Total_Seidel", "EE50_mean"],
#     ascending=[True, True],          # ↓ TS (mejor), ↓ EE50_mean (mejor)
#     na_position="last"
# )

# print(ranked[["L1","L2","Total_Seidel","EE50_mean"]].head(5).to_string(index=False))


    


