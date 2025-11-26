# DAB-SMOTE: Density-Aware Borderline SMOTE

<div style="text-align: justify"> 

**Authors:** Unai Lalana Morales & José Antonio Sanz Delgado  

DAB-SMOTE is an advanced oversampling method for handling classification problems with imbalanced classes. Its goal is to improve classifier performance by generating synthetic samples of the minority class, taking into account the distribution, density, and clusters boundaries [1].

## Method Description
DAB-SMOTE combines noise detection techniques, density-based clustering (DBSCAN), boundary analysis, and synthetic sample generation guided by the data structure. The method includes:

- **Noise removal:** Identification and elimination of outliers in the minority class using statistical distances (euclidean, manhattan, chebyshev).
- **Clustering:** Grouping of the minority class using DBSCAN, with the option of centroid by mean or densest point.
- **Boundary detection:** Identification of samples at the boundary of each cluster.
- **Sample generation:** Creation of new synthetic instances from the boundary and centroid of each cluster, respecting the local structure.


## Repository Structure

- `classes/`
    - `DAB_SMOTE.py`: Main implementation of the DAB-SMOTE method.
    - `dataset.py`: Utilities for reading datasets in .dat format.
    - `Visual_DAB.py`: A visualization and debugging extension of the DAB_SMOTE
- `data/`
    - `benchmarks/`: Classic imbalanced datasets for benchmarking.
    - `initial_test/`: Example dataset for quick tests.
- `notebooks/`
    - `InitialTest.ipynb`: Basic usage and visualization example.
    - `DensityBasedCenters.ipynb`: Advanced example with density-based centers.
    - `Visualization.ipynb`: Visualization notebook.
    - `benchmarks/InitialBenchmarks.ipynb`: Simple benchmarking
- `docs/`
    - `DAB_SMOTE.html`: DAB_SMOTE class documentation.
    - `dataset.html`: Dataset class documentation.
    - `index.html`: Main index page.
    - `Visual_DAB.html`: Visual_DAB class documentation.
- `tests/`
    - `test_dab_smote.py`: Test cases for DAB-SMOTE class.
    - `test_visual_dab.py`: Test cases for Visual-DAB class.

## Installation and Environment (using **uv**)

This project uses **uv** to manage environments and dependencies
efficiently.

### 1. Sync the environment


``` bash
uv sync
```

This command will:

-   Automatically create a virtual environment (`.venv/`) if it does not
    exist.
-   Install all dependencies specified in `pyproject.toml`.
-   Generate/update the `uv.lock` lockfile.

### 2. Activate the environment

``` bash
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows PowerShell
```

### 3. Run tests
```bash
uv run python -m pytest -v
```
And test coverage with:

``` bash
uv run python -m pytest -v  --cov=classes --cov-config=pyproject.toml
```

## Basic Usage Example

```python
from classes.DAB_SMOTE import DAB_SMOTE
import pandas as pd
import numpy as np

# Load data (example with CSV)
df = pd.read_csv('data/initial_test/glass4.dat', header=None)
X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

# Instantiate and apply DAB-SMOTE
dab = DAB_SMOTE(dist_method='euclidean', k=2, progress=True)
X_res, y_res = dab.fit_resample(X, y)
dab.summary()
```

For complete examples and visualizations, see the notebooks in the `notebooks/` folder.

## Main Parameters
- `dist_method`: Distance method for noise detection (`'euclidean'`, `'manhattan'`, `'chebyshev'`).
- `k`: Borderline factor for borderline sample detection.
- `solver`: Centroid strategy (`'means'` for mean, `'density'` for densest point).
- `progress`: Shows progress bar.

## Datasets
The benchmarking datasets are in `data/benchmarks/` and follow the .dat format. Use the utilities in `classes/dataset.py` to read them.

## References and Citation
[1]	U. Lalana and J. A. S. Delgado, ‘Estudio, análisis e implementación de FSDR-SMOTE, técnica de sobremuestreo para problemas de clasificación desbalanceados’, Universidad Publica de Navarra.

---

For any questions or contributions, contact the authors.
</div>