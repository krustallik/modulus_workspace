#!/usr/bin/env python3
import os
import sys
import numpy as np
import pyvista as pv
from sympy import symbols, Function

# Modulus imports
import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig, to_absolute_path
from modulus.sym.geometry.tessellation import Tessellation
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.solver import Solver
from modulus.sym.key import Key

# Import custom steady k-epsilon PDE class
from turbulence.custom_k_ep_3D import kEpsilon

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    """
    3D-симуляція потоку через короб із дроном і пропелерами з RANS k–ε.
    Геометрію центрируємо, масштабуємо; ν відмасштабовано для збереження того ж Re.
    PINN прогнозує u,v,w,p,k,epsilon у фізичних одиницях для порівняння з OpenFOAM.
    """
    # I. Фізичні параметри
    nu_phys = 1.5e-5    # кінематична в’язкість [м²/с]
    rho_phys = 1.0      # густина [кг/м³]
    u_in = 10.0         # вхідна швидкість [м/с]

    # II. Геометричне нормування
    center = np.array([0.0, 0.0, 0.0])
    scale = 0.1

    def normalize_mesh(mesh: Tessellation) -> Tessellation:
        return mesh.translate((-center).tolist()).scale(scale)

    def normalize_invar(invar: dict) -> dict:
        invar['x'] = (invar['x'] - center[0]) * scale
        invar['y'] = (invar['y'] - center[1]) * scale
        invar['z'] = (invar['z'] - center[2]) * scale
        return invar

    # III. Завантаження та нормалізація STL
    stl_dir = to_absolute_path('./stl_files')
    names = ['inlet','outlet','walls','drone','blade1','blade2','blade3','blade4','cube']
    stls = {}
    for name in names:
        path = os.path.join(stl_dir, f"{name}.stl")
        airtight = name in ('drone','blade1','blade2','blade3','blade4','cube')
        stls[name] = normalize_mesh(
            Tessellation.from_stl(path, airtight=airtight)
        )
    inlet_stl  = stls['inlet']
    outlet_stl = stls['outlet']
    walls_stl  = stls['walls']
    drone_stl  = stls['drone']
    blade_stls = [stls[f'blade{i}'] for i in range(1,5)]
    cube_stl   = stls['cube']

    # IV. PDE: Navier-Stokes + custom steady k-ε model
    # Масштабована ν для збереження того ж Re (ν_phys * scale)
    nu_scaled = nu_phys * scale
    ns_eq = NavierStokes(nu=nu_scaled, rho=rho_phys, dim=3, time=False)
    ke_eq = kEpsilon(nu=nu_scaled, rho=rho_phys)

    # V. Архітектура PINN
    input_keys = [Key('x'), Key('y'), Key('z')]
    output_keys = [Key(var) for var in ('u','v','w','p','k','ep')]
    flow_net = instantiate_arch(
        input_keys=input_keys,
        output_keys=output_keys,
        cfg=cfg.arch.fully_connected
    )
    flow_node = flow_net.make_node(name='flow_network')

    # Combine all PDE nodes
    nodes = ns_eq.make_nodes() + ke_eq.make_nodes() + [flow_node]
    domain = Domain()

    # VI. Граничні умови: швидкість та тиск
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes, geometry=inlet_stl,
            outvar={'u':0.0,'v':-u_in,'w':0.0},
            batch_size=cfg.batch_size.inlet
        ), 'inlet_bc'
    )
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes, geometry=outlet_stl,
            outvar={'p':0.0},
            batch_size=cfg.batch_size.outlet
        ), 'outlet_bc'
    )
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes, geometry=walls_stl,
            outvar={'u':0.0,'v':0.0,'w':0.0},
            batch_size=cfg.batch_size.walls
        ), 'walls_bc'
    )
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes, geometry=drone_stl,
            outvar={'u':0.0,'v':0.0,'w':0.0},
            batch_size=cfg.batch_size.drone
        ), 'drone_bc'
    )
    for i, blade in enumerate(blade_stls, start=1):
        domain.add_constraint(
            PointwiseBoundaryConstraint(
                nodes=nodes, geometry=blade,
                outvar={'u':0.0,'v':0.0,'w':0.0},
                batch_size=cfg.batch_size.get(f'blade{i}',400)
            ), f'blade{i}_bc'
        )

    # VII. Турбулентні умови: k та ε на вході та виході
    turb_vals = {'k':0.19, 'ep':0.01}
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes, geometry=inlet_stl,
            outvar=turb_vals,
            batch_size=cfg.batch_size.inlet
        ), 'inlet_turb_bc'
    )
    # domain.add_constraint(
    #     PointwiseBoundaryConstraint(
    #         nodes=nodes, geometry=outlet_stl,
    #         outvar=turb_vals,
    #         batch_size=cfg.batch_size.outlet
    #     ), 'outlet_turb_bc'
    # )

    # VIII. Турбулентні умови на стінках: спрощення k=0, ε=0
    # zero_turb = {'k':0.0,'ep':0.0}
    # domain.add_constraint(
    #     PointwiseBoundaryConstraint(
    #         nodes=nodes, geometry=walls_stl,
    #         outvar=zero_turb,
    #         batch_size=cfg.batch_size.walls
    #     ), 'walls_turb_bc'
    # )
    # domain.add_constraint(
    #     PointwiseBoundaryConstraint(
    #         nodes=nodes, geometry=drone_stl,
    #         outvar=zero_turb,
    #         batch_size=cfg.batch_size.drone
    #     ), 'drone_turb_bc'
    # )
    # for i, blade in enumerate(blade_stls, start=1):
    #     domain.add_constraint(
    #         PointwiseBoundaryConstraint(
    #             nodes=nodes, geometry=blade,
    #             outvar=zero_turb,
    #             batch_size=cfg.batch_size.get(f'blade{i}',400)
    #         ), f'blade{i}_turb_bc'
    #     )

    # IX. Внутрішні умови: решта PDE residuals у кубі
    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=cube_stl,
        outvar={
            'continuity':0.0,
            'momentum_x':0.0,'momentum_y':0.0,'momentum_z':0.0,
            'k_equation':0.0,'ep_equation':0.0
        },
        batch_size=cfg.batch_size.interior
    )
    domain.add_constraint(interior, 'interior')

    # X. Validator: порівняння з VTK даними OpenFOAM
    try:
        vtk_path = os.path.join(os.path.dirname(__file__), 'foamValidationData/myWindTunnelCase_1500.vtk')
        mesh = pv.read(vtk_path)
        mesh.cell_data.clear()
        coords = mesh.points
        U_of = mesh.point_data['U']
        P_of = mesh.point_data['p']
        invar_sub = {'x':coords[:,0:1],'y':coords[:,1:2],'z':coords[:,2:3]}
        invar_sub = normalize_invar(invar_sub)
        true_sub = {
            'u':U_of[:,0:1],'v':U_of[:,1:2],'w':U_of[:,2:3],
            'p':P_of.reshape(-1,1)
        }
        domain.add_validator(
            PointwiseValidator(
                nodes=nodes,
                invar=invar_sub,
                true_outvar=true_sub,
                batch_size=cfg.batch_size.interior
            ), 'foam_validator'
        )
        print('[INFO] Validator added.')
    except Exception as e:
        print('[ERROR] Validator failed:', e)

    # XI. Запуск Solver
    solver = Solver(cfg, domain)
    print('Eval metrics:', solver.eval())
    solver.solve()
    print('[INFO] Training complete!')

if __name__ == '__main__':
    run()
