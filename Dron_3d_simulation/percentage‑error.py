import os, numpy as np, pyvista as pv

VTP = "outputs/__main__/validators/foam_validator.vtp"
u_in = 10.0            # [m/s]
rho  = 1.0             # [kg/m3]
ref  = {"u": u_in, "v": u_in, "w": u_in, "p": 0.5*rho*u_in**2}
thr  = {"u": 0.05, "v": 0.05, "w": 0.05, "p": 1e-2}

mesh = pv.read(VTP)

def stats(var):
    pred = mesh.point_data[f"pred_{var}"].astype(float)
    true = mesh.point_data[f"true_{var}"].astype(float)
    mask = np.abs(true) > thr[var]
    if mask.sum() == 0:
        print(f"{var:>4s}  — всі значення < порога, пропускаю")
        return
    err = np.abs(pred[mask] - true[mask]) / ref[var] * 100
    print(f"{var:>4s}  mean={err.mean():8.3f}%  95‑pct={np.percentile(err,95):8.3f}%  max={err.max():8.3f}%")

for v in ("p", "u", "v", "w"):
    stats(v)
