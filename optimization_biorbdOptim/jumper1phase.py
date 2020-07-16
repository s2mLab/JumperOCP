from time import time

import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    Constraint,
    Objective,
    ProblemType,
    BidirectionalMapping,
    Mapping,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    Data,
    PlotType,
)

# TODO: No custom constraint 1 contact to 0 added here yet

# def from_2contacts_to_1(ocp, nlp, t, x, u, p):
#     return ocp.nlp[0]["contact_forces_func"](x[0], u[0], p)[[2, 5], -1]

# def custom_func_anatomical_constraint(ocp, nlp, t, x, u, p):
#     val = x[0][7:14]
#     return val


def prepare_ocp(model_path, phase_time, number_shooting_points, use_symmetry=True, use_actuators=True):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(model_path)

    if use_actuators:
        torque_activation_min, torque_activation_max, torque_activation_init = -1, 1, 0
    else:
        torque_min, torque_max, torque_init = -500, 500, 0

    if use_symmetry:
        q_mapping = BidirectionalMapping(
            Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
        )
        q_mapping = q_mapping
        tau_mapping = BidirectionalMapping(
            Mapping([-1, -1, -1, -1, 0, -1, 0, 1, 2, 3, 1, 2, 3], [5]), Mapping([4, 7, 8, 9])
        )
        tau_mapping = tau_mapping

    # Add objective functions
    objective_functions = (
        {"type": Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, "weight": -1},
        # {"type": Objective.Lagrange.MINIMIZE_ALL_CONTROLS, "weight": 1 / 100},
        # {"type": Objective.Lagrange.MINIMIZE_ALL_CONTROLS, "weight": -1 / 100},
    )

    # Dynamics
    # problem_type = (
    #     {"type": ProblemType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT},
    # )
    problem_type = ({"type": ProblemType.TORQUE_DRIVEN_WITH_CONTACT},)
    constraints_second_phase = []

    # Positivity constraints of the normal component of the reaction forces
    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        constraints_second_phase.append(
            {
                "type": Constraint.CONTACT_FORCE_INEQUALITY,
                "direction": "GREATER_THAN",
                "instant": Instant.ALL,
                "contact_force_idx": i,
                "boundary": 0,
            }
        )

    # Non-slipping constraints
    # N.B.: Application on only one of the two feet is sufficient, as the slippage cannot occurs on only one foot.
    # constraints_first_phase.append(
    #     {
    #         "type": Constraint.NON_SLIPPING,
    #         "instant": Instant.ALL,
    #         "normal_component_idx": (1, 2),
    #         "tangential_component_idx": 0,
    #         "static_friction_coefficient": 0.5,
    #     }
    # )
    constraints_second_phase.append(
        {
            "type": Constraint.NON_SLIPPING,
            "instant": Instant.ALL,
            "normal_component_idx": 1,
            "tangential_component_idx": 0,
            "static_friction_coefficient": 0.5,
        }
    )

    constraints = (constraints_second_phase,)

    # for i, constraints_phase in enumerate(constraints):
    #     constraints_phase.append({"type": Constraint.TIME_CONSTRAINT, "minimum": time_min, "maximum": time_max})

    # Path constraint
    if use_symmetry:
        nb_q = q_mapping.reduce.len
        nb_qdot = nb_q
        pose_at_first_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]

    # Initialize X_bounds
    X_bounds = QAndQDotBounds(biorbd_model, all_generalized_mapping=q_mapping)
    X_bounds.min[:, 0] = pose_at_first_node + [0] * nb_qdot
    X_bounds.max[:, 0] = pose_at_first_node + [0] * nb_qdot

    # X_bounds = ProblemType.slicing_bounds("q", X_bounds)
    # q_bounds = X_bounds[0][:nq]

    # Initial guess
    X_init = InitialConditions(pose_at_first_node + [0] * nb_qdot)

    # Define control path constraint
    if use_actuators:
        U_bounds = [
            Bounds(
                min_bound=[torque_activation_min] * tau_mapping.reduce.len,
                max_bound=[torque_activation_max] * tau_mapping.reduce.len,
            )
        ]
        U_init = [InitialConditions([torque_activation_init] * tau_mapping.reduce.len)]
    else:
        U_bounds = [
            Bounds(min_bound=[torque_min] * tau_mapping.reduce.len, max_bound=[torque_max] * tau_mapping.reduce.len)
        ]
        U_init = [InitialConditions([torque_init] * tau_mapping.reduce.len)]

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        phase_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        q_mapping=q_mapping,
        q_dot_mapping=q_mapping,
        tau_mapping=tau_mapping,
    )


def plot_CoM(x):
    m = biorbd.Model("../models/jumper2contacts.bioMod")
    q_mapping = BidirectionalMapping(
        Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
    )
    q_reduced = x[:7, :]
    q_expanded = q_mapping.expand.map(q_reduced)
    from casadi import Function, MX
    import numpy as np

    q_sym = MX.sym("q", m.nbQ(), 1)
    CoM_func = Function("Compute_CoM", [q_sym], [m.CoM(q_sym).to_mx()], ["q"], ["CoM"],).expand()
    CoM = np.array(CoM_func(q_expanded[:, :]))
    return CoM[2]


def plot_CoM_dot(x):
    m = biorbd.Model("../models/jumper2contacts.bioMod")
    q_mapping = BidirectionalMapping(
        Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
    )
    q_reduced = x[:7, :]
    qdot_reduced = x[7:, :]
    q_expanded = q_mapping.expand.map(q_reduced)
    qdot_expanded = q_mapping.expand.map(qdot_reduced)
    from casadi import Function, MX
    import numpy as np

    q_sym = MX.sym("q", m.nbQ(), 1)
    qdot_sym = MX.sym("q_dot", m.nbQdot(), 1)
    CoM_dot_func = Function(
        "Compute_CoM_dot", [q_sym, qdot_sym], [m.CoMdot(q_sym, qdot_sym).to_mx()], ["q", "q_dot"], ["CoM_dot"],
    ).expand()
    CoM_dot = np.array(CoM_dot_func(q_expanded[:, :], qdot_expanded[:, :]))
    return CoM_dot[2]


# def run_and_save_ocp(model_path, phase_time, number_shooting_points):
#     ocp = prepare_ocp(
#         model_path=model_path, phase_time=phase_time, number_shooting_points=number_shooting_points, use_symmetry=True, use_actuators=True
#     )
#
#     sol = ocp.solve(options_ipopt={"hessian_approximation": "exact", "max_iter": 1000}, show_online_optim=False)
#
#     OptimalControlProgram.save(ocp, sol, "../Results/jumper2phases_sol")


if __name__ == "__main__":
    model_path = "../models/jumper2contacts.bioMod"
    # time_min = 0.4
    # time_max = 0.75
    phase_time = 0.32
    number_shooting_points = 30

    tic = time()
    # run_and_save_ocp(model_path, phase_time=phase_time, number_shooting_points=number_shooting_points)
    # ocp, sol = OptimalControlProgram.load("../Results/jumper2phases_sol.bo")

    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=phase_time,
        number_shooting_points=number_shooting_points,
        use_symmetry=True,
        use_actuators=False,
    )
    ocp.add_plot("CoM", lambda x, u, p: plot_CoM(x), phase_number=0, plot_type=PlotType.PLOT)
    ocp.add_plot("CoM_dot", lambda x, u, p: plot_CoM_dot(x), phase_number=0, plot_type=PlotType.PLOT)

    sol = ocp.solve(show_online_optim=True, options_ipopt={"hessian_approximation": "exact", "max_iter": 1000})

    # --- Show results --- #
    # param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    # print(
    #     f"The optimized phases times are: {param['time'][0, 0]}s."
    # )

    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate(nb_frames=61)
