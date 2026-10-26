"""
Random Forest surrogate for lid-driven cavity flow.
 
Trains a Random Forest on OpenFOAM results at Re = 100, 400, 1000, 2000
and tests it on an unseen Reynolds number (Re = 700).
 
Main finding: the forest does not interpolate in Re. For Re = 700 it returns
the Re = 400 field almost exactly; for Re = 701 it returns the Re = 1000 field.
"""
 
import os
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
 
FOAM_RUN = os.environ.get('FOAM_RUN')
COMPONENTS = ['Ux', 'Uy']
 
 
# ============================================================
# 1. Data
# ============================================================
 
df = pd.read_csv(f"{FOAM_RUN}/cavity_data.csv")
 
# Inputs: cell-centre position and Reynolds number
X = df[['x', 'y', 'Re']].values
# Outputs: velocity components (already non-dimensional, U_lid = 1)
y = df[['Ux', 'Uy']].values
 
 
# ============================================================
# 2. Hold out Re = 700 for testing
# ============================================================
 
mask = (df['Re'] == 700).values
 
X_train, y_train = X[~mask], y[~mask]   # Re = 100, 400, 1000, 2000
X_test, y_test = X[mask], y[mask]       # Re = 700 (never seen in training)
 
# CFD fields of the neighbouring training cases (same cell order as X_test)
y_400 = y[(df['Re'] == 400).values]
y_1000 = y[(df['Re'] == 1000).values]
y_2000 = y[(df['Re'] == 2000).values]
 
 
# ============================================================
# 3. Train the Random Forest and predict Re = 700
# ============================================================
 
rf = RandomForestRegressor(
    n_estimators=100,   # number of trees
    random_state=42,    # fix the randomness: same forest on every run
    n_jobs=-1           # use all CPU cores
)
 
# Each tree is trained on its own bootstrap sample of the training data
rf.fit(X_train, y_train)
 
y_pred = rf.predict(X_test)
 
 
# ============================================================
# 4. Error metrics at Re = 700
# ============================================================
 
print("=== RF prediction vs. true Re = 700 ===")
for i, name in enumerate(COMPONENTS):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    max_err = np.max(np.abs(y_test[:, i] - y_pred[:, i]))
    print(f"{name}: R2 = {r2:.4f}, RMSE = {rmse:.4f}, max |error| = {max_err:.4f}")
 
 
# ============================================================
# 5. Copy baselines: how good is simply copying a training field?
# ============================================================
 
print("\n=== Copy baselines vs. true Re = 700 (R2) ===")
for i, name in enumerate(COMPONENTS):
    r2_copy_400 = r2_score(y_test[:, i], y_400[:, i])
    r2_copy_1000 = r2_score(y_test[:, i], y_1000[:, i])
    print(f"{name}: copy Re=400 = {r2_copy_400:.4f}, copy Re=1000 = {r2_copy_1000:.4f}")
 
 
# ============================================================
# 6. Direct check: is the RF prediction just the Re = 400 field?
# ============================================================
 
print("\n=== RF (Re=700 query) vs. CFD Re = 400 ===")
for i, name in enumerate(COMPONENTS):
    diff = np.abs(y_pred[:, i] - y_400[:, i])
    print(f"{name}: max |RF - Re400| = {diff.max():.6f}, "
          f"mean = {diff.mean():.6f}, "
          f"R2 = {r2_score(y_400[:, i], y_pred[:, i]):.6f}")
 
 
# ============================================================
# 7. Step test: same points, different Re queries
# ============================================================
 
def predict_at_re(re_value):
    """Predict the velocity field at the test cell centres for a given Re."""
    X_query = X_test.copy()     # copy, so X_test itself is not modified
    X_query[:, 2] = re_value    # change only the Re column
    return rf.predict(X_query)
 
 
y_701 = predict_at_re(701)
y_1300 = predict_at_re(1300)
 
print("\n=== Step test (R2 against neighbouring CFD fields) ===")
for i, name in enumerate(COMPONENTS):
    print(f"Re=701,  {name}: vs Re=400 = {r2_score(y_400[:, i], y_701[:, i]):.4f}, "
          f"vs Re=1000 = {r2_score(y_1000[:, i], y_701[:, i]):.4f}")
          
for i, name in enumerate(COMPONENTS):
    print(f"Re=1300, {name}: vs Re=1000 = {r2_score(y_1000[:, i], y_1300[:, i]):.4f}, "
          f"vs Re=2000 = {r2_score(y_2000[:, i], y_1300[:, i]):.4f}")
 
 
# ============================================================
# 8. Centreline plot
# ============================================================
 
# Cell-centre (x, y) coordinates: identical for every Re
coords = X_test[:, :2]
 
 
def vertical_centreline(field, coords):
    """Cells on the vertical line closest to x = 0.5. Returns y and velocities."""
    x_c = coords[:, 0]
    y_c = coords[:, 1]
 
    # 1. Find the x value closest to 0.5 (0.49375 on an 80x80 mesh)
    x_line = x_c[np.argmin(np.abs(x_c - 0.5))]
 
    # 2. Select the 80 cells at this x
    sel = np.isclose(x_c, x_line)
 
    # 3. Sort from bottom to top
    order = np.argsort(y_c[sel])
    return y_c[sel][order], field[sel][order]
 
 
def horizontal_centreline(field, coords):
    """Cells on the horizontal line closest to y = 0.5. Returns x and velocities."""
    x_c = coords[:, 0]
    y_c = coords[:, 1]
 
    # 1. Find the y value closest to 0.5
    y_line = y_c[np.argmin(np.abs(y_c - 0.5))]
 
    # 2. Select the 80 cells at this y
    sel = np.isclose(y_c, y_line)
 
    # 3. Sort from left to right
    order = np.argsort(x_c[sel])
    return x_c[sel][order], field[sel][order]
 
 
fields = {
    'OpenFOAM Re=400': y_400,
    'OpenFOAM Re=700': y_test,
    'OpenFOAM Re=1000': y_1000,
    'RF Re=700': y_pred,
    'RF Re=701': y_701,
}
 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
for label, field in fields.items():
    if label.startswith('RF'):
        style = dict(linestyle='--', linewidth=1.5)
    else:
        # Thick, semi-transparent CFD curves so the RF curves on top stay visible
        style = dict(linestyle='-', linewidth=3, alpha=0.5)
 
    y_line, U = vertical_centreline(field, coords)
    axes[0].plot(U[:, 0], y_line, label=label, **style)
 
    x_line, U = horizontal_centreline(field, coords)
    axes[1].plot(x_line, U[:, 1], label=label, **style)
 
axes[0].set_xlabel('Ux')
axes[0].set_ylabel('y')
axes[0].set_title('Ux along the vertical centreline (x ≈ 0.5)')
 
axes[1].set_xlabel('x')
axes[1].set_ylabel('Uy')
axes[1].set_title('Uy along the horizontal centreline (y ≈ 0.5)')
 
for ax in axes:
    ax.grid(True)
    ax.legend()
 
plt.tight_layout()
plt.savefig(f"{FOAM_RUN}/rf_centerline_Re700.png", dpi=150)
plt.show()
