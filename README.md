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
