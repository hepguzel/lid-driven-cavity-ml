# lid-driven-cavity-ml

Machine-learning surrogate models for 2D lid-driven cavity flow. The CFD data are generated with OpenFOAM. Every model is trained on the same data and tested on the same unseen case, so the results can be compared directly.

## The test

The models are trained on CFD solutions at Re = 100, 400, 1000 and 2000, and asked to predict the flow at **Re = 700**, which they never see during training. Each model is also compared against a simple baseline: copying the nearest training solution, CFD(400) or CFD(1000), without any model.

## Results

| Model | Ux R² at Re = 700 | Uy R² at Re = 700 | Report |
|---|---|---|---|
| Baseline: copy of CFD(1000) | 0.989 | 0.991 | - |
| Random Forest | 0.966 | 0.972 | [RF/](RF/) |
| Neural network (MLP) | | | in progress |
| Physics-Informed Neural Network (PINN) | | | planned |

The Random Forest does not beat the copy baseline: at Re = 700 it returns the Re = 400 solution almost unchanged. See the [RF report](RF/README.md) for details.

## Repository structure

| Folder | Contents |
|---|---|
| [`CFD/`](CFD/) | OpenFOAM cases (Re sweep and mesh independence study) and the extracted dataset |
| [`RF/`](RF/) | Random Forest model, report and figures |

Each model folder contains its own code, report (`README.md`) and figures.

## License

[MIT](LICENSE)
