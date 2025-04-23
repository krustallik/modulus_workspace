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
import scipy.interpolate  # for griddata interpolation
from modulus.sym.domain.monitor import PointwiseMonitor
from scipy.interpolate import griddata
import matplotlib.pyplot as plt  # for plotting

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
from sympy import Eq, Symbol



class SliceValidatorPlotter(ValidatorPlotter):
    """
    Custom plotter: для кожної змінної (u, v, w, p)
    будуємо 1×3 підграфіки: true, pred, difference.
    Зріз завжди по площині X ≈ 0.
    """

    def __call__(self, invar, true_outvar, pred_outvar):
        # Витягаємо координати X, Y, Z
        x = invar["x"][:, 0]
        y = invar["y"][:, 0]
        z = invar["z"][:, 0]

        # Фіксуємо plane X ≈ 0
        x_mid = 0.0
        # Крок сітки по X для товщини зрізу
        dx = (x.max() - x.min()) / 200.0
        tol = 2 * dx

        # Маска точок близько до X=0
        mask = np.abs(x - x_mid) <= tol
        if mask.sum() < 20:
            tol *= 5
            mask = np.abs(x - x_mid) <= tol

        # координати зрізу у площині Y–Z
        yi = y[mask]
        zi = z[mask]

        # extent для imshow: (y_min, y_max, z_min, z_max)
        extent = (y.min(), y.max(), z.min(), z.max())

        # регулярна 200×200 сітка у Y–Z
        yi_lin = np.linspace(extent[0], extent[1], 200)
        zi_lin = np.linspace(extent[2], extent[3], 200)
        YI, ZI = np.meshgrid(yi_lin, zi_lin, indexing="xy")
        pts2d = (yi, zi)

        figs = []
        for var in ("u", "v", "w", "p"):
            tvals = true_outvar[var][mask, 0]
            pvals = pred_outvar[var][mask, 0]
            dvals = np.abs(pvals - tvals)

            Tg = griddata(pts2d, tvals, (YI, ZI), method="linear")
            Pg = griddata(pts2d, pvals, (YI, ZI), method="linear")
            Dg = griddata(pts2d, dvals, (YI, ZI), method="linear")

            fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
            fig.suptitle(f"{var.upper()} slice at X≈{x_mid:.3f}", fontsize=14)

            cmap_base = "coolwarm" if var == "p" else "viridis"
            for data, title, cmap_name, ax in [
                (Tg, f"True {var}", cmap_base, axs[0]),
                (Pg, f"Pred {var}", cmap_base, axs[1]),
                (Dg, f"Diff {var}",    "plasma",     axs[2]),
            ]:
                im = ax.imshow(data.T, origin="lower",
                               extent=extent, cmap=cmap_name)
                ax.set_xlabel("Y")
                ax.set_ylabel("Z")
                ax.set_title(title)
                fig.colorbar(im, ax=ax, shrink=0.8)

            plt.tight_layout(rect=[0,0,1,0.92])
            figs.append((fig, f"slice_{var}"))

        return figs




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
        os.path.join(stl_dir, "cube.stl"), airtight=True
    )
    # Cube mesh
    inlet_stl = Tessellation.from_stl(
        os.path.join(stl_dir, "inlet.stl"), airtight=True
    )
    # Cube mesh
    outlet_stl = Tessellation.from_stl(
        os.path.join(stl_dir, "outlet.stl"), airtight=True
    )
    # Cube mesh
    walls_stl = Tessellation.from_stl(
        os.path.join(stl_dir, "walls.stl"), airtight=True
    )

    # Drone body mesh
    drone_stl = Tessellation.from_stl(
        os.path.join(stl_dir, "drone.stl"), airtight=True
    )


    # Domain bounds (matching STL coordinates)
    ymin, ymax = -10, 10

    # --- C) Define PDE: steady incompressible Navier–Stokes ---
    ze_eq = ZeroEquation(nu=nu_phys, dim=3, time=False, max_distance=0.5,rho=rho_phys)
    ns_eq = NavierStokes(nu=ze_eq.equations["nu"], rho=rho_phys, dim=3, time=False)

    # --- D) PINN architecture (MLP) ---
    input_keys  = [Key("x"), Key("y"), Key("z")]
    output_keys = [Key("u"), Key("v"),Key("w"), Key("p")]
    flow_net    = instantiate_arch(
        input_keys=input_keys,
        output_keys=output_keys,
        cfg=cfg.arch.fully_connected
    )
    flow_node   = flow_net.make_node(name="flow_network")

    nodes = ns_eq.make_nodes()+ns_eq.make_nodes()+ [flow_node]
    # nodes = ns_eq.make_nodes() + [flow_node]

    # --- E) Create domain and add constraints ---
    domain = Domain()

    # 1) Inlet
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=inlet_stl,
            outvar={"u": 0.0, "v": -u_in, "w": 0.0},
            batch_size=cfg.batch_size.inlet
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




    # # VIII. Validator: compare PINN output vs OpenFOAM reference
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


if __name__ == "__main__":
    run()

