import os
import numpy as np
import pyvista as pv

# Шлях до VTP-файла
VTP = "outputs/navier_and_zero/validators/foam_validator.vtp"

# Завантажуємо сітку з даними pred_* та true_*
mesh = pv.read(VTP)

def stats(var):
    pred = mesh.point_data[f"pred_{var}"].astype(float)
    true = mesh.point_data[f"true_{var}"].astype(float)
    # MSE по всіх точках
    mse = np.mean((pred - true)**2)
    # Вивід у фіксованому форматі з 6-ма знаками після коми
    print(f"{var:>4s}  MSE = {mse:.6f}")
    return mse

# Обчислюємо MSE для p, u, v, w
mses = [stats(v) for v in ("p", "u", "v", "w")]

# Додатково: загальне середнє MSE
overall_mse = np.mean(mses)
print(f"\nOverall MSE (p,u,v,w): {overall_mse:.6f}")
