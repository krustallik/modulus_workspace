
from sympy import Symbol, Eq, Abs


# == Modulus imports ==
import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint
)
import torch
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt



from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.key import Key
from modulus.sym.solver import Solver


# Новий механізм параметризації
from modulus.sym.geometry.parameterization import Parameterization

# Геометричні примітиви з Modulus
from modulus.sym.geometry.primitives_2d import (
    Rectangle,
    Polygon as SymPolygon,
    Circle as SymCircle
)

from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation


def plot_geometry():
    """
    Візуалізує геометрію симуляції з деталями для дрона, пропелерів, будівель,
    тунелю з нерівностями та додаванням вихрових генераторів.
    """
    # Визначення геометрії
    width = 10.0
    height = 5.0

    # Масштабування дрона
    drone_scale = 0.8  # Зменшення розміру дрона на 50%

    # Вітрова труба з нерівностями (полігон)
    tunnel_points = [
        (-width / 2, -height / 2),
        (-width / 2,  height / 2),
        (-3,  height / 2),
        (-2,  height / 2 - 0.5),  # Нерівність на верхній стінці
        ( 0,  height / 2),
        ( 2,  height / 2 - 0.5),
        ( 3,  height / 2),
        ( width / 2,  height / 2),
        ( width / 2, -height / 2),
        ( 3, -height / 2),
        ( 2, -height / 2 + 0.5),  # Нерівність на нижній стінці
        ( 0, -height / 2),
        (-2, -height / 2 + 0.5),
        (-3, -height / 2),
    ]

    # Дрон (головний корпус)
    drone_body_points = [
        (-0.5 * drone_scale, -0.1 * drone_scale + 0.7),
        (-0.5 * drone_scale, 0.1 * drone_scale + 0.7),
        (0.5 * drone_scale, 0.1 * drone_scale + 0.7),
        (0.5 * drone_scale, -0.1 * drone_scale + 0.7),
    ]
    left_wing_points = [
        (-0.5 * drone_scale, 0.1 * drone_scale + 0.7),
        (-1.5 * drone_scale, 0.5 * drone_scale + 0.7),
        (-1.5 * drone_scale, 0.6 * drone_scale + 0.7),
        (-0.5 * drone_scale, 0.2 * drone_scale + 0.7),
    ]
    right_wing_points = [
        (0.5 * drone_scale, 0.1 * drone_scale + 0.7),
        (1.5 * drone_scale, 0.5 * drone_scale + 0.7),
        (1.5 * drone_scale, 0.6 * drone_scale + 0.7),
        (0.5 * drone_scale, 0.2 * drone_scale + 0.7),
    ]
    front_propeller_center = (-1.0 * drone_scale, 0.55 * drone_scale + 0.7)
    rear_propeller_center = (1.0 * drone_scale, 0.55 * drone_scale + 0.7)

    # Лопаті пропелерів (4 лопаті для кожного)
    propeller_length = 0.3 * drone_scale
    propeller_width = 0.05 * drone_scale

    def create_blades(center, length, width):
        cx, cy = center
        return [
            [
                (cx - length / 2, cy - width / 2),
                (cx + length / 2, cy - width / 2),
                (cx + length / 2, cy + width / 2),
                (cx - length / 2, cy + width / 2),
            ],
            [
                (cx - width / 2, cy - length / 2),
                (cx + width / 2, cy - length / 2),
                (cx + width / 2, cy + length / 2),
                (cx - width / 2, cy + length / 2),
            ],
        ]

    front_blades = create_blades(front_propeller_center, propeller_length, propeller_width)
    rear_blades = create_blades(rear_propeller_center, propeller_length, propeller_width)

    # Будівлі
    building1_points = [(-4, -2.5), (-3.5, -2.5), (-3.5, -1), (-4, -1)]
    building2_points = [( 3.5, -2.5), ( 4, -2.5), ( 4,  1.0), ( 3.5,  1.0)]

    # Вихрові генератори
    vortex_generators = [
        [(-4.5, -1.8), (-4.2, -1.5), (-4.5, -1.2)],  # Малий трикутник зліва
        [( 2.0,  1.2), ( 2.3,  1.5), ( 2.0,  1.8)],  # Малий трикутник справа
    ]

    # Перешкоди
    obstacle1_points = [
        (-3, -1.5),  # Опущена нижче
        (-2.5, -1.5),
        (-2.5, -0.7),  # Зменшена висота
        (-3, -0.7)
    ]
    obstacle2_points = [
        (1, -1.5),
        (1.5, -1.5),
        (1.5, -1.0),
        (1, -1.0)
    ]

    # Візуалізація
    fig, ax = plt.subplots(figsize=(10, 5))

    # Вітрова труба
    tunnel_patch = plt.Polygon(tunnel_points, closed=True, edgecolor='black', facecolor='none', linewidth=1.5)
    ax.add_patch(tunnel_patch)

    # Дрон
    drone_body_patch = plt.Polygon(drone_body_points,  closed=True, edgecolor='blue', facecolor='none', linewidth=1.5)
    left_wing_patch  = plt.Polygon(left_wing_points,   closed=True, edgecolor='blue', facecolor='none', linewidth=1.5)
    right_wing_patch = plt.Polygon(right_wing_points,  closed=True, edgecolor='blue', facecolor='none', linewidth=1.5)
    ax.add_patch(drone_body_patch)
    ax.add_patch(left_wing_patch)
    ax.add_patch(right_wing_patch)

    # Пропелери
    for blade in front_blades + rear_blades:
        blade_patch = plt.Polygon(blade, closed=True, edgecolor='purple', facecolor='none', linewidth=1.5)
        ax.add_patch(blade_patch)

    # Будівлі
    building1_patch = plt.Polygon(building1_points, closed=True, edgecolor='red', facecolor='none', linewidth=1.5)
    building2_patch = plt.Polygon(building2_points, closed=True, edgecolor='red', facecolor='none', linewidth=1.5)
    ax.add_patch(building1_patch)
    ax.add_patch(building2_patch)

    # Вихрові генератори
    for vortex in vortex_generators:
        vortex_patch = plt.Polygon(vortex, closed=True, edgecolor='green', facecolor='lightgreen', linewidth=1.5)
        ax.add_patch(vortex_patch)

    # Перешкоди
    obstacle1_patch = plt.Polygon(obstacle1_points, closed=True, edgecolor='orange', facecolor='none', linewidth=1.5)
    obstacle2_patch = plt.Polygon(obstacle2_points, closed=True, edgecolor='orange', facecolor='none', linewidth=1.5)
    ax.add_patch(obstacle1_patch)
    ax.add_patch(obstacle2_patch)

    # Налаштування графіка
    ax.set_aspect('equal')
    ax.set_xlim([-6, 6])
    ax.set_ylim([-3, 3])
    ax.set_title("Simulation Geometry with Vortex Generators, Obstacles, and Buildings")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


def visualize_results_and_save_vtp(
    solver,
    flow_net,                 # <-- об’єкт архітектури (PyTorch-модель)
    flow_network_node,        # <-- об’єкт Node
    x_min, x_max, y_min, y_max,
    t_value=0.0,
    nx=100,
    ny=100,
    vtp_filename="results.vtp"
):
    """
    Візуалізує результати (u, v, p) на прямокутній сітці [x_min..x_max]×[y_min..y_max]
    при часі t_value, а також зберігає ці дані у форматі .vtp.
    """

    # 1) Створюємо сітку точок (NumPy)
    x_vals = np.linspace(x_min, x_max, nx)
    y_vals = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    T = np.full_like(X, t_value)

    invar_np = {
        "x": X.flatten()[:, None],
        "y": Y.flatten()[:, None],
        "t": T.flatten()[:, None],
    }

    # 2) Переводимо flow_network_node в eval-режим (вимикає dropout, batchnorm тощо)
    flow_net.eval()

    # 3) Формуємо invar_torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    invar_torch = {}
    for k, v in invar_np.items():
        # k тут буде рядок "x", "y", або "t"
        invar_torch[k] = torch.tensor(v, dtype=torch.float32, device=device)

    # 4) Прямий прохід (forward pass)
    with torch.no_grad():
        outvar_torch = flow_network_node.evaluate(invar_torch)

    # 5) Перетворюємо виходи назад у NumPy і надаємо формату (ny, nx)
    u = outvar_torch["u"].cpu().numpy().reshape(X.shape)
    v = outvar_torch["v"].cpu().numpy().reshape(X.shape)
    p = outvar_torch["p"].cpu().numpy().reshape(X.shape)
    speed = np.sqrt(u**2 + v**2)

    # ---------------------------
    # 6) ВІЗУАЛІЗАЦІЯ MATPLOTLIB
    # ---------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Векторне поле (u,v)
    axs[0].quiver(X, Y, u, v, scale=5)
    axs[0].set_title("Velocity Field (u, v)")
    axs[0].set_aspect("equal", "box")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    # Поле тиску (p)
    cont1 = axs[1].contourf(X, Y, p, levels=50, cmap="viridis")
    fig.colorbar(cont1, ax=axs[1])
    axs[1].set_title("Pressure Field (p)")
    axs[1].set_aspect("equal", "box")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    # Модуль швидкості |v|
    cont2 = axs[2].contourf(X, Y, speed, levels=50, cmap="plasma")
    fig.colorbar(cont2, ax=axs[2])
    axs[2].quiver(X, Y, u, v, color="white", scale=5, alpha=0.8)
    axs[2].set_title("Speed Magnitude |v|")
    axs[2].set_aspect("equal", "box")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")

    plt.tight_layout()
    plt.show()

    # ---------------------------------
    # 7) ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ У .VTP
    # ---------------------------------
    points = np.zeros((nx*ny, 3), dtype=np.float32)
    points[:, 0] = X.flatten()
    points[:, 1] = Y.flatten()
    points[:, 2] = 0.0  # 2D

    grid = pv.StructuredGrid()
    grid.dimensions = (nx, ny, 1)  # якщо X.shape == (ny, nx)
    grid.points = points

    grid.point_data["u"] = u.flatten()
    grid.point_data["v"] = v.flatten()
    grid.point_data["p"] = p.flatten()

    grid.save(vtp_filename)
    print(f"[INFO] Results saved to '{vtp_filename}'")

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    """
    Основна функція симуляції потоку повітря довкола дрона
    з використанням Navier-Stokes та Modulus, з урахуванням турбулентності та обертання пропелерів.
    """
    # ---------------------
    # І. ЗАДАННЯ ГЕОМЕТРІЇ
    # ---------------------
    width = 10.0
    height = 5.0
    drone_scale = 0.8# Зменшення розміру дрона на 50%

    # Параметри часу (кутова швидкість пропелерів)
    omega = 20.0
    t_symbol = Symbol("t")

    # Збільшення часу в 3 рази
    time_range = (0.0, 3 * (2.0 * np.pi / omega))
    time_param = Parameterization({t_symbol: time_range})

    # === Дрон ===
    drone_body = SymPolygon([
        (-0.5 * drone_scale, -0.1 * drone_scale + 0.7),
        (-0.5 * drone_scale, 0.1 * drone_scale + 0.7),
        (0.5 * drone_scale, 0.1 * drone_scale + 0.7),
        (0.5 * drone_scale, -0.1 * drone_scale + 0.7),
    ])
    left_wing = SymPolygon([
        (-0.5 * drone_scale, 0.1 * drone_scale + 0.7),
        (-1.5 * drone_scale, 0.5 * drone_scale + 0.7),
        (-1.5 * drone_scale, 0.6 * drone_scale + 0.7),
        (-0.5 * drone_scale, 0.2 * drone_scale + 0.7),
    ])
    right_wing = SymPolygon([
        (0.5 * drone_scale, 0.1 * drone_scale + 0.7),
        (1.5 * drone_scale, 0.5 * drone_scale + 0.7),
        (1.5 * drone_scale, 0.6 * drone_scale + 0.7),
        (0.5 * drone_scale, 0.2 * drone_scale + 0.7),
    ])

    # Пропелери (дві окружності)
    front_propeller_center = (-1.0 * drone_scale, 0.55 * drone_scale + 0.7)
    rear_propeller_center = (1.0 * drone_scale, 0.55 * drone_scale + 0.7)
    front_propeller = SymCircle(center=front_propeller_center, radius=0.1 * drone_scale)
    rear_propeller = SymCircle(center=rear_propeller_center, radius=0.1 * drone_scale)

    # Сукупна геометрія дрона
    drone = drone_body + left_wing + right_wing + front_propeller + rear_propeller

    # === Тунель з нерівностями (полігон) ===
    tunnel_points = [
        (-width / 2, -height / 2),
        (-width / 2,  height / 2),
        (-3,  height / 2),
        (-2,  height / 2 - 0.5),
        ( 0,  height / 2),
        ( 2,  height / 2 - 0.5),
        ( 3,  height / 2),
        ( width / 2,  height / 2),
        ( width / 2, -height / 2),
        ( 3, -height / 2),
        ( 2, -height / 2 + 0.5),
        ( 0, -height / 2),
        (-2, -height / 2 + 0.5),
        (-3, -height / 2),
    ]
    tunnel = SymPolygon(tunnel_points)

    # === Перешкоди всередині тунелю ===
    obstacle1 = SymPolygon([
        (-3, -1.5),  # Опущена нижче
        (-2.5, -1.5),
        (-2.5, -0.7),  # Зменшена висота
        (-3, -0.7)
    ])
    obstacle2 = SymPolygon([
        (1, -1.5),
        (1.5, -1.5),
        (1.5, -1.0),
        (1, -1.0)
    ])

    # === Вихрові генератори (трикутники) ===
    vortex_generator1 = SymPolygon([
        (-4.5, -1.8),
        (-4.2, -1.5),
        (-4.5, -1.2),
    ])
    vortex_generator2 = SymPolygon([
        (2.0, 1.2),
        (2.3, 1.5),
        (2.0, 1.8),
    ])

    # === Будівлі (два полігони) ===
    building1 = SymPolygon([
        (-4, -2.5),
        (-3.5, -2.5),
        (-3.5, -1),
        (-4, -1)
    ])
    building2 = SymPolygon([
        (2, -2.5),
        (2.5, -2.5),
        (2.5,  0.5),
        (2,    0.5)
    ])

    # ---------------------
    # ІІ. ВИЗНАЧЕННЯ РІВНЯНЬ
    # ---------------------
    nu  = 1.48e-5  # кінематична в'язкість
    rho = 1.293    # густина повітря

    navier_stokes = NavierStokes(nu=nu, rho=rho, dim=2, time=True)
    zero_eq       = ZeroEquation(nu=nu, max_distance=0.5, dim=2, rho=rho)

    # Нейронна мережа
    flow_net = instantiate_arch(
        input_keys = [Key("x"), Key("y"), Key("t")],
        output_keys= [Key("u"), Key("v"), Key("p"), Key("nu_t")],
        cfg=cfg.arch.fully_connected,
    )

    # ОТРИМУЄМО ВУЗОЛ, зберігаємо його у змінну
    flow_network_node = flow_net.make_node(name="flow_network")

    # Збираємо всі ноди
    nodes = (
            navier_stokes.make_nodes()
            + zero_eq.make_nodes()
            + [flow_network_node]  # Додаємо наш flow_network_node
    )

    # ---------------------
    # ІІІ. СТВОРЕННЯ DOMAIN ТА ДОДАВАННЯ ОБМЕЖЕНЬ
    # ---------------------
    domain = Domain()

    # 1) Вхід (Inlet), x = -width/2
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tunnel,
        outvar={"u": 1.0, "v": 0.0},  # вхідна швидкість
        batch_size=cfg.batch_size.inlet,
        criteria=Eq(Symbol("x"), -width / 2),
        parameterization=time_param,
    )
    domain.add_constraint(inlet, "inlet")

    # 2) Вихід (Outlet), x = width/2
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tunnel,
        outvar={"p": 0.0},  # тиск = 0 на виході
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(Symbol("x"), width / 2),
        parameterization=time_param,
    )
    domain.add_constraint(outlet, "outlet")

    # 3) Стіни тунелю (no-slip), окрім inlet/outlet
    def not_inlet_outlet(invar, param):
        x_ = invar["x"]
        tol = 1e-5  # невеликий допуск
        return (x_ > -width / 2 + tol) & (x_ < width / 2 - tol)

    no_slip_walls = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tunnel,
        outvar={"u": 0.0, "v": 0.0},
        batch_size=cfg.batch_size.walls,
        criteria=not_inlet_outlet,  # Використовуємо нашу функцію замість Abs(...)
        parameterization=time_param,
    )
    domain.add_constraint(no_slip_walls, "no_slip_walls")

    # 4) Поверхня дрона (окрім пропелерів) — no-slip
    drone_main_body = drone_body + left_wing + right_wing
    drone_surface = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=drone_main_body,
        outvar={"u": 0.0, "v": 0.0},
        batch_size=cfg.batch_size.drone,
        parameterization=time_param,
    )
    domain.add_constraint(drone_surface, "drone_surface")

    # 5) Пропелери (задаємо обертання)
    X_sym = Symbol("x")
    Y_sym = Symbol("y")

    # Передній пропелер
    u_rot_front = -omega * (Y_sym - front_propeller_center[1])
    v_rot_front =  omega * (X_sym - front_propeller_center[0])
    front_propeller_velocity = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=front_propeller,
        outvar={
            "u": u_rot_front,
            "v": v_rot_front,
        },
        batch_size=cfg.batch_size.rotors,
        parameterization=time_param,
    )
    domain.add_constraint(front_propeller_velocity, "front_propeller_velocity")

    # Задній пропелер
    u_rot_rear = -omega * (Y_sym - rear_propeller_center[1])
    v_rot_rear =  omega * (X_sym - rear_propeller_center[0])
    rear_propeller_velocity = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rear_propeller,
        outvar={
            "u": u_rot_rear,
            "v": v_rot_rear,
        },
        batch_size=cfg.batch_size.rotors,
        parameterization=time_param,
    )
    domain.add_constraint(rear_propeller_velocity, "rear_propeller_velocity")

    # 6) Будівлі (no-slip)
    buildings_surface = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=building1 + building2,
        outvar={"u": 0.0, "v": 0.0},
        batch_size=cfg.batch_size.buildings,
        parameterization=time_param,
    )
    domain.add_constraint(buildings_surface, "buildings_surface")

    # 7) Перешкоди всередині тунелю (no-slip)
    obstacles_surface = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=obstacle1 + obstacle2,
        outvar={"u": 0.0, "v": 0.0},
        batch_size=cfg.batch_size.obstacles,
        parameterization=time_param,
    )
    domain.add_constraint(obstacles_surface, "obstacles_surface")

    # 8) Вихрові генератори (no-slip)
    vortex_generators_surface = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=vortex_generator1 + vortex_generator2,
        outvar={"u": 0.0, "v": 0.0},
        batch_size=cfg.batch_size.obstacles,
        parameterization=time_param,
    )
    domain.add_constraint(vortex_generators_surface, "vortex_generators")

    # 9) Інтер’єр (continuity, momentum_x, momentum_y, ...)
    interior_geometry = (
        tunnel
        - drone_body
        - left_wing
        - right_wing
        - front_propeller
        - rear_propeller
        - building1
        - building2
        - obstacle1
        - obstacle2
        - vortex_generator1
        - vortex_generator2
    )

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_geometry,
        outvar={
            "continuity": 0.0,
            "momentum_x": 0.0,
            "momentum_y": 0.0,
        },
        batch_size=cfg.batch_size.interior,
        parameterization=time_param,
    )
    domain.add_constraint(interior, "interior")

    # # ---------------------
    # # IV. ВАЛІДАЦІЯ (за бажанням)
    # # ---------------------
    # x_vals = np.linspace(-width/2, width/2, 50)
    # y_vals = np.linspace(-height/2, height/2, 50)
    # t_vals = np.linspace(time_range[0], time_range[1], 10)
    # X, Y, T = np.meshgrid(x_vals, y_vals, t_vals)
    #
    # validation_points = {
    #     "x": X.flatten()[:, None],
    #     "y": Y.flatten()[:, None],
    #     "t": T.flatten()[:, None],
    # }
    #
    # # Тут умовні "true" дані (або експериментальні).
    # validator = PointwiseValidator(
    #     nodes=nodes,
    #     invar=validation_points,
    #     true_outvar={
    #         "u":    np.zeros_like(validation_points["x"]),
    #         "v":    np.zeros_like(validation_points["x"]),
    #         "p":    np.zeros_like(validation_points["x"]),
    #         "nu_t": np.zeros_like(validation_points["x"]),
    #     },
    #     batch_size=1024,
    # )
    # domain.add_validator(validator)

    # ---------------------
    # V. СТВОРЕННЯ РОЗВ’ЯЗУВАЧА ТА НАВЧАННЯ
    # ---------------------
    solver = Solver(cfg, domain)
    solver.solve()
    print("[INFO] Training complete!")

    # ---------------------
    # VI. ВІЗУАЛІЗАЦІЯ + ЗАПИС У .VTP
    # ---------------------
    # Наприклад, подивимось на момент часу t = 0.0
    # та виведемо поле в прямокутнику x: [-5..5], y: [-2.5..2.5]
    visualize_results_and_save_vtp(
        solver=solver,
        flow_net=flow_net,  # <-- передаємо
        flow_network_node=flow_network_node,  # <-- передаємо
        x_min=-5.0,
        x_max=5.0,
        y_min=-2.5,
        y_max=2.5,
        t_value=0.0,
        nx=400,
        ny=200,
        vtp_filename = "all_domain.vts"
    )


if __name__ == "__main__":
    # За бажанням показуємо початкову геометрію (статичний малюнок без результатів):
    plot_geometry()
    run()