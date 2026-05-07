# 222_OStage

This repository contains the scripts and tools used to implement the optical design and optimization workflow for a focal reducer system. The workflow includes first-order modeling, third-order aberration-based optimization, glass selection, a physically grounded merit function (PGMF) stage, and a second-stage image-quality refinement.

## Repository structure

### `examples/`

This folder contains basic usage examples.

- `EE_example.py`: example of enclosed energy calculation using KrakenOS.
- `Glass_selector_example.py`: example of the `GlassXtractor` tool for extracting and filtering S-FPL51 and F2HT glasses in the `(n, V_d)` space.

### `first_order/`

This folder contains the first-order paraxial models.

- `First_Order_Focal_Reducer.py`: first-order calculation for a two-lens focal reducer, applied to the blue channel.
- `Three_Lenses_Paraxial.py`: first-order calculation for a three-lens focal reducer, applied to the three spectral channels.

### `third_order/`

This folder contains the third-order optical models and glass optimization scripts.

- `Third_Order_Focal_Reducer.py`: third-order model for the two-lens focal reducer using the reference glasses S-FPL51 and F2HT.
- `Three_Lenses_Third_Order_Focal_Reducer_Ch1.py`, `Ch2.py`, and `Ch3.py`: third-order models for the three-lens focal reducer in each channel. These scripts allow switching between the optimized glass and the reference glass.
- `Third_Order_Focal_Reducer_Glass_Opt.py`: two-lens third-order model allowing glass substitution.
- `Glass_Optimization_Third_Order.py`: implementation of the glass selection and classification procedure used by the previous script.

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

### `second_stage/`

This folder contains the scripts for the second optimization stage, focused on image-quality refinement.

For the three-lens configuration, scripts are provided for the three channels. For the two-lens configuration, only the blue channel is considered.

### `Results/`

This folder stores the results of the glass optimization and filtering process. The script `Metrics_Glasses_Selector.py` is used to classify and evaluate candidate glasses.

### `Optimized_Parameters/`

This folder stores the optimized parameters obtained at each stage of the workflow.

### `images/`

This folder stores spot diagrams and enclosed energy plots generated during the glass optimization and the different optimization stages.

### `utils/`

This folder contains reusable classes and equations used across the scripts.

## License

This project is licensed under the GNU GPL v3.0 License.
- `utils/equations/`: mathematical formulations and auxiliary equations.
- `utils/classes/`: reusable classes, including the `GlassXtractor` tool.
