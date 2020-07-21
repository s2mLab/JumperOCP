from time import time

import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ConstraintList,
    Constraint,
    Objective,
    ObjectiveList,
    DynamicsTypeList,
    DynamicsType,
    BidirectionalMapping,
    Mapping,
    BoundsList,
    QAndQDotBounds,
    InitialConditions,
    InitialConditionsList,
    InterpolationType,
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

def CoM_dot_Z_last_node_positivity(ocp, nlp, t, x, u, p):
    from casadi import Function, MX
    q_reduced = x[0][:nlp["nbQ"]]
    qdot_reduced = x[0][nlp["nbQ"]:]

    q_mapping = BidirectionalMapping(
        Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
    )
    q_expanded = q_mapping.expand.map(q_reduced)
    qdot_expanded = q_mapping.expand.map(qdot_reduced)

    q_sym = MX.sym("q", q_expanded.size()[0], 1)
    qdot_sym = MX.sym("q_dot", qdot_expanded.size()[0], 1)
    CoM_dot_func = Function(
        "Compute_CoM_dot", [q_sym, qdot_sym], [nlp["model"].CoMdot(q_sym, qdot_sym).to_mx()], ["q", "q_dot"], ["CoM_dot"],
    ).expand()
    CoM_dot = CoM_dot_func(q_expanded, qdot_expanded)
    return CoM_dot[2]


def prepare_ocp(model_path, phase_time, number_shooting_points, use_symmetry=True, use_actuators=True):
    # --- Options --- #
    # Model path
    biorbd_model = [biorbd.Model(elt) for elt in model_path]

    nb_phases = len(biorbd_model)
    if use_actuators:
        torque_activation_min, torque_activation_max, torque_activation_init = -1, 1, 0
    else:
        torque_min, torque_max, torque_init = -500, 500, 0

    if use_symmetry:
        q_mapping = BidirectionalMapping(
            Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
        )
        q_mapping = q_mapping, q_mapping
        tau_mapping = BidirectionalMapping(
            Mapping([-1, -1, -1, -1, 0, -1, 0, 1, 2, 3, 1, 2, 3], [5]), Mapping([4, 7, 8, 9])
        )
        tau_mapping = tau_mapping, tau_mapping

    else:
        q_mapping = BidirectionalMapping(
            Mapping([i for i in range(biorbd_model[0].nbQ())]), Mapping([i for i in range(biorbd_model[0].nbQ())])
        )
        q_mapping = q_mapping, q_mapping
        tau_mapping = q_mapping

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1, phase=1)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)
    dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)

    # dynamics.add(DynamicsType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT)
    # dynamics.add(DynamicsType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=0, second_marker_idx=1, phase=0)

    # constraints_first_phase = []
    # constraints_second_phase = []

    # Positivity constraints of the normal component of the reaction forces
    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        # constraints_first_phase.append(
        #     {
        #         "type": Constraint.CONTACT_FORCE_INEQUALITY,
        #         "direction": "GREATER_THAN",
        #         "instant": Instant.ALL,
        #         "contact_force_idx": i,
        #         "boundary": 0,
        #     }
        # )
        constraints.add(Constraint.CONTACT_FORCE_INEQUALITY, phase=0, direction="GREATER_THAN", instant=Instant.ALL, contact_force_idx=i, boundary=0)
    contact_axes = (1, 3)
    for i in contact_axes:
        # constraints_second_phase.append(
        #     {
        #         "type": Constraint.CONTACT_FORCE_INEQUALITY,
        #         "direction": "GREATER_THAN",
        #         "instant": Instant.ALL,
        #         "contact_force_idx": i,
        #         "boundary": 0,
        #     }
        # )
        constraints.add(Constraint.CONTACT_FORCE_INEQUALITY, phase=1, direction="GREATER_THAN", instant=Instant.ALL, contact_force_idx=i, boundary=0)


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
    constraints.add(Constraint.NON_SLIPPING, phase=0, instant=Instant.ALL, normal_component_idx=(1, 2), tangential_component_idx=0, static_friction_coefficient=0.5)

    # constraints_second_phase.append(
    #     {
    #         "type": Constraint.NON_SLIPPING,
    #         "instant": Instant.ALL,
    #         "normal_component_idx": 1,
    #         "tangential_component_idx": 0,
    #         "static_friction_coefficient": 0.5,
    #     }
    # )
    constraints.add(Constraint.NON_SLIPPING, phase=1, instant=Instant.ALL, normal_component_idx=1, tangential_component_idx=0, static_friction_coefficient=0.5)


    # Custom constraints for contact forces at transitions
    # constraints_second_phase.append(
    #     {"type": Constraint.CUSTOM, "function": from_2contacts_to_1, "instant": Instant.START}
    # )

    # Custom constraints for positivity of CoM_dot on z axis just before the take-off
    constraints.add(CoM_dot_Z_last_node_positivity, phase=1, instant=Instant.END, min_bound=0, max_bound=np.inf)

    # if not use_symmetry:
    #     first_dof = (3, 4, 7, 8, 9)
    #     second_dof = (5, 6, 10, 11, 12)
    #     coef = (-1, 1, 1, 1, 1)
    #     for i in range(len(first_dof)):
    #         for elt in [
    #             constraints_first_phase,
    #             constraints_second_phase,
    #         ]:
    #             elt.append(
    #                 {
    #                     "type": Constraint.PROPORTIONAL_STATE,
    #                     "instant": Instant.ALL,
    #                     "first_dof": first_dof[i],
    #                     "second_dof": second_dof[i],
    #                     "coef": coef[i],
    #                 }
    #             )

    # constraints = (
    #     constraints_first_phase,
    #     constraints_second_phase,
    # )
    # for i, constraints_phase in enumerate(constraints):
    #     constraints_phase.append({"type": Constraint.TIME_CONSTRAINT, "minimum": time_min[i], "maximum": time_max[i]})

    # Path constraint
    if use_symmetry:
        nb_q = q_mapping[0].reduce.len
        nb_qdot = nb_q
        pose_at_first_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]
    else:
        nb_q = q_mapping[0].reduce.len
        nb_qdot = nb_q
        pose_at_first_node = [0, 0, -0.5336, 0, 1.4, 0, 1.4, 0.8, -0.9, 0.47, 0.8, -0.9, 0.47]

    # Initialize x_bounds (Interpolation type is CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    x_bounds = BoundsList()
    for i in range(nb_phases):
        x_bounds.add(QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping[i]))
    x_bounds[0].min[:, 0] = pose_at_first_node + [0] * nb_qdot
    x_bounds[0].max[:, 0] = pose_at_first_node + [0] * nb_qdot

    # X_bounds = [QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping[i]) for i in range(nb_phases)]
    # X_bounds[0].min[:, 0] = pose_at_first_node + [0] * nb_qdot
    # X_bounds[0].max[:, 0] = pose_at_first_node + [0] * nb_qdot

    # X_bounds = ProblemType.slicing_bounds("q", X_bounds)
    # q_bounds = X_bounds[0][:nq]

    # Initial guess (Interpolation type is CONSTANT)
    x_init = InitialConditionsList()
    for i in range(nb_phases):
        x_init.add(pose_at_first_node + [0] * nb_qdot)

    # X_init = [InitialConditions(pose_at_first_node + [0] * nb_qdot) for i in range(nb_phases)]

    # Define control path constraint (
    if use_actuators:
        u_bounds = BoundsList()
        for tau_m in tau_mapping:
            u_bounds.add([[torque_activation_min] * tau_m.reduce.len, [torque_activation_max] * tau_m.reduce.len], interpolation=InterpolationType.CONSTANT)    # This precision of the CONSTANT type is for informative purposes only

    # if use_actuators:
    #     U_bounds = [
    #         Bounds(
    #             min_bound=[torque_activation_min] * tau_m.reduce.len,
    #             max_bound=[torque_activation_max] * tau_m.reduce.len,
    #         )
    #         for tau_m in tau_mapping
    #     ]
    #     U_init = [InitialConditions([torque_activation_init] * tau_m.reduce.len) for tau_m in tau_mapping]

        u_init = InitialConditionsList()
        for tau_m in tau_mapping:
            u_init.add([torque_activation_init] * tau_m.reduce.len)     # Interpolation type is CONSTANT (default value)

    else:
        u_bounds = BoundsList()
        for tau_m in tau_mapping:
            u_bounds.add([[torque_min] * tau_m.reduce.len, [torque_max] * tau_m.reduce.len], interpolation=InterpolationType.CONSTANT)  # This precision of the CONSTANT type is for informative purposes only
        u_init = InitialConditionsList()
        for tau_m in tau_mapping:
            u_init.add([torque_init] * tau_m.reduce.len)  # Interpolation type is CONSTANT (default value)

    # else:
    #     U_bounds = [
    #         Bounds(min_bound=[torque_min] * tau_m.reduce.len, max_bound=[torque_max] * tau_m.reduce.len)
    #         for tau_m in tau_mapping
    #     ]
    #     U_init = [InitialConditions([torque_init] * tau_m.reduce.len) for tau_m in tau_mapping]

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        phase_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        q_mapping=q_mapping,
        q_dot_mapping=q_mapping,
        tau_mapping=tau_mapping,
    )


def plot_CoM(x, model_path):
    m = biorbd.Model(model_path)
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


def plot_CoM_dot(x, model_path):
    m = biorbd.Model(model_path)
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
#         model_path=model_path, phase_time=phase_time, number_shooting_points=number_shooting_points, use_symmetry=True
#     )
#
#     sol = ocp.solve(options_ipopt={"hessian_approximation": "limited-memory", "max_iter": 1000}, show_online_optim=False)
#
#     OptimalControlProgram.save(ocp, sol, "../Results/jumper2phases_sol")


if __name__ == "__main__":
    model_path = (
        "../models/jumper2contacts.bioMod",
        "../models/jumper1contacts.bioMod",
    )
    time_min = [0.1, 0.3]
    time_max = [0.4, 0.5]
    phase_time = [0.2, 0.32]
    number_shooting_points = [20, 20]

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
    ocp.add_plot(
        "CoM", lambda x, u, p: plot_CoM(x, "../models/jumper2contacts.bioMod"), phase_number=0, plot_type=PlotType.PLOT
    )
    ocp.add_plot(
        "CoM", lambda x, u, p: plot_CoM(x, "../models/jumper1contacts.bioMod"), phase_number=1, plot_type=PlotType.PLOT
    )
    ocp.add_plot(
        "CoM_dot",
        lambda x, u, p: plot_CoM_dot(x, "../models/jumper2contacts.bioMod"),
        phase_number=0,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "CoM_dot",
        lambda x, u, p: plot_CoM_dot(x, "../models/jumper1contacts.bioMod"),
        phase_number=1,
        plot_type=PlotType.PLOT,
    )

    sol = ocp.solve(show_online_optim=True, solver_options={"hessian_approximation": "exact", "max_iter": 20})

    # --- Show results --- #
    # param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    # print(
    #     f"The optimized phases times are: {param['time'][0, 0]}s and {param['time'][1, 0]}s."
    # )

    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    result = ShowResult(ocp, sol)
    result.graphs(adapt_graph_size_to_bounds=False)
    result.animate(nb_frames=61)
