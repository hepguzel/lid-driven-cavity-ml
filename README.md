# lid-driven-cavity-ml

Machine-learning surrogate models for 2D lid-driven cavity flow. The CFD data are generated with OpenFOAM. Every model is trained on the same data and tested on the same unseen cases, so the results can be compared directly.

## The test

The models are trained on CFD solutions at Re = 100, 400, 1000 and 2000, and asked to predict the flow at **Re = 700**, which they never see during training. The neural network is also tested at **Re = 1500**.

Each model is compared against simple baselines that need no model at all:

- **copying** the nearest training solution, CFD(400) or CFD(1000),
- **averaging** the two neighbouring training solutions, which is the same as linear interpolation in Re.

## Results

- The **Random Forest** does not beat the copy baseline: at Re = 700 it returns the Re = 400 solution almost unchanged. See the [RF report](RF/README.md).
- The **neural network** interpolates: at the unseen Re = 700 its prediction matches the CFD solution with an RMSE of about 0.005, i.e. 0.5% of the lid velocity, and at Re = 1500 with about 0.006. Its largest errors are in the top corners of the cavity. Compared with the averaging baseline (the average of CFD(400) and CFD(1000)), its RMSE is about half at Re = 700, but no better at Re = 1500. See the [NN report](NN/README.md).
- A **physics-informed neural network (PINN)** is planned as the next model.


## Repository structure

| Folder | Contents |
|---|---|
| [`CFD/`](CFD/) | OpenFOAM cases (Re sweep and mesh independence study), the extracted dataset and its [description](CFD/README.md) |
| [`RF/`](RF/) | Random Forest model, report and figures |
| [`NN/`](NN/) | Neural network (MLP) model, report and figures |

Each model folder contains its own code, report (`README.md`) and figures.

## License

[MIT](LICENSE)
