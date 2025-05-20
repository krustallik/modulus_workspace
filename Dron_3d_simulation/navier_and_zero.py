#!/usr/bin/env python3
"""
PINN with Zero Equation turbulence (drone + propellers inside cube)
---------------------------------------------------------------------
This script uses the Zero Equation turbulence closure without any mesh normalization.
"""
import torch
# ---------------------------------------------------------------------
# 1) Import standard Python libraries
# ---------------------------------------------------------------------
import os           # for file path operations
import numpy as np  # for numerical arrays and math
import pyvista as pv  # for reading VTK files and geometries
import re
import csv
import scipy.interpolate  # for griddata interpolation
from modulus.sym.domain.monitor import PointwiseMonitor
from scipy.interpolate import griddata
import matplotlib.pyplot as plt  # for plotting
from sympy import Symbol, Abs, Max
# ---------------------------------------------------------------------
# 2) Import Modulus Sym libraries
# ---------------------------------------------------------------------
import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig, to_absolute_path
from modulus.sym.geometry.tessellation import Tessellation
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from modulus.sym.solver import Solver
from modulus.sym.key import Key
from modulus.sym.utils.io.plotter import ValidatorPlotter
from sympy import Eq, Symbol, Abs


class SliceValidatorPlotter(ValidatorPlotter):
    """
    Custom plotter: for each variable (u, v, w, p)
    we build a single figure with three vertical panels: true, pred, difference.
    Slice always at X ≈ 0.
    """
    def __call__(self, invar, true_outvar, pred_outvar):
        # Extract coordinates and build mask around x ≈ 0
        x = invar["x"][:, 0]
        y = invar["y"][:, 0]
        z = invar["z"][:, 0]
        x_mid = 0.0
        dx = (x.max() - x.min()) / 200.0
        tol = 2 * dx
        mask = np.abs(x - x_mid) <= tol
        if mask.sum() < 20:
            tol *= 5
            mask = np.abs(x - x_mid) <= tol

        # Prepare grid for interpolation
        yi, zi = y[mask], z[mask]
        extent = (y.min(), y.max(), z.min(), z.max())
        yi_lin = np.linspace(extent[0], extent[1], 200)
        zi_lin = np.linspace(extent[2], extent[3], 200)
        YI, ZI = np.meshgrid(yi_lin, zi_lin, indexing="xy")
        pts2d = (yi, zi)

        figs = []
        for var in ("u", "v", "w", "p"):
            # Gather true, pred, and diff values
            tvals = true_outvar[var][mask, 0]
            pvals = pred_outvar[var][mask, 0]
            dvals = np.abs(pvals - tvals)

            # Interpolate onto regular grid
            Tg = griddata(pts2d, tvals, (YI, ZI), method="linear")
            Pg = griddata(pts2d, pvals, (YI, ZI), method="linear")
            Dg = griddata(pts2d, dvals, (YI, ZI), method="linear")

            # Choose colormaps
            cmap_base = "coolwarm" if var == "p" else "viridis"

            # Create a single figure with 3 vertical subplots
            fig, axes = plt.subplots(3, 1, figsize=(6, 15), dpi=100)
            for ax, data, title, cmap_name in zip(
                axes,
                (Tg, Pg, Dg),
                (
                    f"True {var.upper()} slice at X≈{x_mid:.3f}",
                    f"Pred {var.upper()} slice at X≈{x_mid:.3f}",
                    f"Diff {var.upper()} slice at X≈{x_mid:.3f}",
                ),
                (cmap_base, cmap_base, "plasma"),
            ):
                im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap_name)
                ax.set_xlabel("Y")
                ax.set_ylabel("Z")
                ax.set_title(title)
                fig.colorbar(im, ax=ax, shrink=0.8)

            plt.tight_layout()
            figs.append((fig, var))

        return figs

def analyze_loss(log_path, bin_size, out_csv, out_png):
    """
    Parse training log, compute average loss per bin, save CSV and plot.
    """
    pattern = re.compile(r'\[step:\s*(\d+)\].*?loss:\s*([\d\.e\+\-]+)')
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                data.append((int(m.group(1)), float(m.group(2))))
    sums, counts = {}, {}
    for step, loss in data:
        bin_start = ((step-1)//bin_size)*bin_size if step>0 else 0
        sums.setdefault(bin_start,0.0)
        counts.setdefault(bin_start,0)
        sums[bin_start] += loss
        counts[bin_start] += 1
    bins = sorted(sums.keys())
    avg_losses = [sums[b]/counts[b] for b in bins]
    # CSV
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['step_bin_start','avg_loss'])
        for b, avg in zip(bins, avg_losses):
            writer.writerow([b, f"{avg:.6f}"])
    print(f"Average loss per {bin_size}-step bins saved to {out_csv}")
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(bins, avg_losses, marker='o', linestyle='-')
    plt.title(f"Average Loss per {bin_size} Steps")
    plt.xlabel("Step (bin start)"); plt.ylabel("Average Loss")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Loss plot saved to {out_png}")


def compute_mse_stats(vtp_path):
    """
    Load validator VTP, compute and print MSE for u,v,w,p plus overall.
    """
    mesh = pv.read(vtp_path)
    def stats(var):
        pred = mesh.point_data[f"pred_{var}"].astype(float)
        true = mesh.point_data[f"true_{var}"].astype(float)
        mse = np.mean((pred-true)**2)
        print(f"{var:>4s}  MSE = {mse:.6f}")
        return mse
    mses = [stats(v) for v in ("p","u","v","w")]
    overall = np.mean(mses)
    print(f"\nOverall MSE (p,u,v,w): {overall:.6f}")


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig):
    # --- A) Physical parameters ---
    nu_phys  = 1.5e-5    # kinematic viscosity [m^2/s]
    rho_phys = 1.225     # density [kg/m^3]
    u_in     = 10.0      # inlet speed [m/s]

    # --- B) STL meshes (no normalization) ---
    stl_dir = to_absolute_path("./stl_files")

    # Cube mesh
    cube_stl = Tessellation.from_stl(
        os.path.join(stl_dir, "cube_without_drone.stl"), airtight=True
    )
    # Cube mesh
    inlet_stl = Tessellation.from_stl(
        os.path.join(stl_dir, "inlet.stl"), airtight=False
    )
    # Cube mesh
    outlet_stl = Tessellation.from_stl(
        os.path.join(stl_dir, "outlet.stl"), airtight=False
    )
    # Cube mesh
    walls_stl = Tessellation.from_stl(
        os.path.join(stl_dir, "walls.stl"), airtight=False
    )

    # Drone body mesh
    drone_stl = Tessellation.from_stl(
        os.path.join(stl_dir, "drone.stl"), airtight=True
    )

    # --- C) Define PDE: steady incompressible Navier–Stokes ---
    ze_eq = ZeroEquation(nu=nu_phys, dim=3, time=False, max_distance=0.1 ,rho=rho_phys)
    ns_eq = NavierStokes(nu=ze_eq.equations["nu"], rho=rho_phys, dim=3, time=False)


    # --- D) PINN architecture (MLP) ---
    input_keys  = [Key("x"), Key("y"), Key("z")]
    output_keys = [Key("u"), Key("v"),Key("w"), Key("p")]


    # 1) Fully-connected MLP

    # flow_net = instantiate_arch(
    #     input_keys=input_keys,
    #     output_keys=output_keys,
    #     cfg=cfg.arch.fully_connected
    # )

    # 2) Modified Fourier-feature MLP

    flow_net = instantiate_arch(
        input_keys=input_keys,
        output_keys=output_keys,
        cfg=cfg.arch.modified_fourier,
        frequencies=("axis,diagonal", [i/2.0 for i in range(6)]),
    )

    # 3) SIREN: sinusoidal activation network

    # flow_net = instantiate_arch(
    #     input_keys=input_keys,
    #     output_keys=output_keys,
    #     cfg=cfg.arch.siren
    # )


    nodes = (ns_eq.make_nodes()+
             ze_eq.make_nodes()+
             [flow_net.make_node(name="flow_network")])



    # --- E) Create domain and add constraints ---
    domain = Domain()



    # Символьні координати на грані inlet (площина y = +10)
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

    # Півширина прямокутника по осях X і Z (наприклад, cube з -10 до +10 → половина = 10)
    Rx = 3
    Rz = 2.6

    # Лінійне зважування: максимум у центрі, спад до 0 на краях по X та Z
    w_expr = Max(1 - Abs(x) / Rx, 0) * Max(1 - Abs(z) / Rz, 0)

    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=inlet_stl,
            outvar={"u": 0.0, "v": -u_in, "w": 0.0},
            lambda_weighting={
                "u": 1.0,
                "v": w_expr,
                "w": 1.0,
            },
            batch_size=cfg.batch_size.inlet,
        ),
        "inlet"
    )

    # 2) Outlet
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=outlet_stl,
            outvar={"p": 0.0},
            batch_size=cfg.batch_size.outlet
        ),
        "outlet"
    )

    # walls
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=walls_stl,
            outvar={"u": 0.0, "v": 0.0, "w": 0.0},
            batch_size=cfg.batch_size.walls
        ),
        "walls_noslip"
    )

    # 4) Drone body no-slip
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=drone_stl,
            outvar={"u": 0.0, "v": 0.0, "w": 0.0},
            batch_size=cfg.batch_size.drone
        ),
        "drone_noslip"
    )

    # 6) Interior: PDE residuals
    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=cube_stl,
            outvar={
                "continuity":  0.0,
                "momentum_x":  0.0,
                "momentum_y":  0.0,
                "momentum_z":  0.0,
            },
            batch_size=cfg.batch_size.interior,

            #for zeroEquation
            compute_sdf_derivatives=True,
            lambda_weighting={
                "continuity": Symbol("sdf"),
                "momentum_x": Symbol("sdf"),
                "momentum_y": Symbol("sdf"),
                "momentum_z": Symbol("sdf"),
            },

        ),
        "interior"
    )




    # # # VIII. Validator: compare PINN output vs OpenFOAM reference
    try:
        vtk_path = os.path.join(os.path.dirname(__file__),
                                "foamValidationData/myWindTunnelCase_1500.vtk")
        vmesh = pv.read(vtk_path)
        vmesh.cell_data.clear()  # drop cell_data
        coords = vmesh.points

        invar = {
            "x": coords[:, 0:1],
            "y": coords[:, 1:2],
            "z": coords[:, 2:3],
        }

        # Ground truth from OpenFOAM
        true_vals = {
            "u": vmesh.point_data["U"][:, 0:1],
            "v": vmesh.point_data["U"][:, 1:2],
            "w": vmesh.point_data["U"][:, 2:3],
            "p": vmesh.point_data["p"].reshape(-1, 1),
        }

        # Add validator with our custom slice plotter
        domain.add_validator(
            PointwiseValidator(
                nodes=nodes,
                invar=invar,
                true_outvar=true_vals,
                batch_size=cfg.batch_size.interior,
                plotter=SliceValidatorPlotter(),  # use our beginner plotter
            ),
            "foam_validator"
        )
        print("[INFO] Validator with custom plotter added.")
    except Exception as e:
        print("[WARN] Validator skipped:", e)


    # --- F) Create solver and run training ---
    solver = Solver(cfg, domain)
    solver.solve()
    print("[INFO] Training complete!")


    analyze_loss(
        log_path=os.path.join(os.path.dirname(__file__),"log","log.txt"),
        bin_size=2000,
        out_csv="loss_avg.csv",
        out_png="loss_avg_plot.png"
    )
    compute_mse_stats(
        vtp_path=os.path.join(os.path.dirname(__file__),
                              "outputs","navier_and_zero","validators","foam_validator.vtp")
    )


if __name__ == "__main__":
    run()

