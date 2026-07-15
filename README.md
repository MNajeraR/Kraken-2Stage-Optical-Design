# 222_OStage

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20187549.svg)](https://doi.org/10.5281/zenodo.20187549)

This repository contains the scripts and tools used to implement the optical design and optimization workflow for focal reducer systems. The workflow includes first-order modeling, third-order aberration-based optimization, glass selection, a physically grounded merit function (PGMF) optimization stage, and a second-stage refinement based on RMS-driven merit functions (RMSMF).

The folder `222_OStage` contains the Python scripts used to generate the results presented in the associated manuscript. For proper execution, this folder must be placed within the KrakenOS directory, at the same level as the `Examples` folder.

## Repository structure

### `examples/`

This folder contains basic usage examples.

- `EE_example.py`: example of enclosed energy calculation using KrakenOS.
- `Glass_selector_example.py`: example of the `GlassXtractor` tool for extracting and filtering S-FPL51 and F2HT glasses in the `(n, V_d)` space.

---

### `first_order/`

This folder contains the first-order paraxial models.

- `First_Order_Focal_Reducer.py`: first-order calculation for a two-lens focal reducer, applied to the blue channel.
- `Three_Lenses_Paraxial.py`: first-order calculation for a three-lens focal reducer, applied to the three spectral channels.

---

### `third_order/`

This folder contains the third-order optical models and glass optimization scripts.

- `Third_Order_Focal_Reducer.py`: third-order model for the two-lens focal reducer using the reference glasses S-FPL51 and F2HT.
- `Three_Lenses_Third_Order_Focal_Reducer_Ch1.py`, `Ch2.py`, and `Ch3.py`: third-order models for the three-lens focal reducer in each channel. These scripts allow switching between the optimized glass and the reference glass.
- `Third_Order_Focal_Reducer_Glass_Opt.py`: two-lens third-order model allowing glass substitution.
- `Glass_Optimization_Third_Order.py`: implementation of the glass selection and classification procedure used by the previous script.

---

### `pgmf_optimization/`

This folder contains the scripts associated with the physically grounded merit function (PGMF) optimization stage.

The scripts follow the naming convention:

```text
PGMF_k_Optimization_jL_Focal_Reducer_Chi.py
```

where:

- `k` indicates the optimization method: `LS`, `MC`, `GA`, or `ABC`.
- `j` indicates the number of lenses: `2L` or `3L`.
- `Chi` indicates the spectral channel for the three-lens configuration, with `i = 1, 2, 3`.

For the two-lens configuration, only the blue channel is considered. For the three-lens configuration, the optimization methods are applied to one channel at a time.

---

## rms_optimization/

This folder contains the scripts associated with the second optimization stage based on RMS-driven merit functions (RMSMF).

The primary optimization scripts follow the naming convention:

```
RMSMF_LS_Optimization_iL_Focal_Reducer_Chj.py
```

where:

- **i** indicates the number of lenses (2L or 3L).
- **Chj** indicates the spectral channel (j = 1, 2, 3).
- For the two-lens configuration, only the blue channel is considered.

In addition, the folder includes the script:

```
RMSMF_LS_Direct_From_Third_Order_3L_Focal_Reducer_Ch1.py
```

which performs RMS refinement directly from the third-order starting design, bypassing the intermediate PGMF stage. This script was developed to provide a direct comparison between the optimization workflows.

---

### `results/`

This folder stores the results of the glass optimization and filtering process. The script `Metrics_Glasses_Selector.py` is used to classify and evaluate candidate glasses.

---

### `optimized_parameters/`

This folder stores the optimized parameters obtained at each stage of the workflow.

---

### `images/`

This folder stores spot diagrams and enclosed energy plots generated during the glass optimization and the different optimization stages.

---

### `utils/`

This folder contains reusable classes and equations used across the scripts.

- `utils/equations/`: mathematical formulations and auxiliary equations.
- `utils/classes/`: reusable classes, including the `GlassXtractor` tool.

---

## Citation

If you use this repository, please cite:

Software DOI:
https://doi.org/10.5281/zenodo.20187549

---

## Related publication

Najera et al. (2026)

"Optical design of OPTICAM-ARG: a three channel high-time resolution camera for the Jorge Sahade telescope"

https://arxiv.org/abs/2605.11329

---

## Requirements

- NumPy 1.22.1
- SciPy 1.7.3
- PyGAD 3.0.1
- bees-algorithm 1.0.2
- KrakenOS 1.0.0.19
- matplotlib 3.5.2

---

## License

This project is licensed under the GNU GPL v3.0 License.
