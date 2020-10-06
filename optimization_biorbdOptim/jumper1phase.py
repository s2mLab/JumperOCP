from time import time

import numpy as np
import biorbd

from bioptim import (
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
    InitialGuess,
    InitialGuessList,
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
    q_reduced = x[0][:nlp.shape["q"]]
    qdot_reduced = x[0][nlp.shape["q"]:]

    q_mapping = BidirectionalMapping(
        Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
    )
    q_expanded = q_mapping.expand.map(q_reduced)
    qdot_expanded = q_mapping.expand.map(qdot_reduced)

    q_sym = MX.sym("q", q_expanded.size()[0], 1)
    qdot_sym = MX.sym("q_dot", qdot_expanded.size()[0], 1)
    CoM_dot_func = Function(
        "Compute_CoM_dot", [q_sym, qdot_sym], [nlp.model.CoMdot(q_sym, qdot_sym).to_mx()], ["q", "q_dot"], ["CoM_dot"],
    ).expand()
    CoM_dot = CoM_dot_func(q_expanded, qdot_expanded)
    return CoM_dot[2]




def prepare_ocp(model_path, phase_time, number_shooting_points, use_symmetry=True, use_actuators=True):
    # --- Options --- #
    # Model path
    biorbd_model = [biorbd.Model(elt) for elt in model_path]

    nb_phases = len(biorbd_model)
    if use_actuators:
        tau_min, tau_max, tau_init = -1, 1, 0
    else:
        tau_min, tau_max, tau_init = -500, 500, 0

    if use_symmetry:
        q_mapping = BidirectionalMapping(
            Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
        )
        q_mapping = q_mapping
        tau_mapping = BidirectionalMapping(
            Mapping([-1, -1, -1, -1, 0, -1, 0, 1, 2, 3, 1, 2, 3], [5]), Mapping([4, 7, 8, 9])
        )
        tau_mapping = tau_mapping

    else:
        q_mapping = BidirectionalMapping(
            Mapping([i for i in range(biorbd_model[0].nbQ())]), Mapping([i for i in range(biorbd_model[0].nbQ())])
        )
        q_mapping = q_mapping
        tau_mapping = q_mapping

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1, phase=0)

    # Dynamics
    dynamics = DynamicsTypeList()
    if use_actuators:
        dynamics.add(DynamicsType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT)
    else:
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)

    # --- Constraints --- #
    constraints = ConstraintList()

    # Positivity constraints of the normal component of the reaction forces
    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        constraints.add(Constraint.CONTACT_FORCE_INEQUALITY, phase=0, direction="GREATER_THAN", instant=Instant.ALL, contact_force_idx=i, boundary=0)

    # Non-slipping constraints
    # N.B.: Application on only one of the two feet is sufficient, as the slippage cannot occurs on only one foot.
    constraints.add(Constraint.NON_SLIPPING, phase=0, instant=Instant.ALL, normal_component_idx=(1, 2), tangential_component_idx=0, static_friction_coefficient=0.5)

    # Custom constraints for contact forces at transitions
    # constraints_second_phase.append(
    #     {"type": Constraint.CUSTOM, "function": from_2contacts_to_1, "instant": Instant.START}
    # )

    # Custom constraints for positivity of CoM_dot on z axis just before the take-off
    constraints.add(CoM_dot_Z_last_node_positivity, phase=0, instant=Instant.END, min_bound=0, max_bound=np.inf)

    # TODO: Make it works also with no symmetry (adapt to refactor)
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

    # # Time constraint
    for i in range(nb_phases):
        constraints.add(Constraint.TIME_CONSTRAINT, phase=i, minimum=time_min[i], maximum=time_max[i])

    # --- Path constraints --- #
    if use_symmetry:
        nb_q = q_mapping.reduce.len
        nb_qdot = nb_q
        pose_at_first_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]
    else:
        nb_q = q_mapping.reduce.len
        nb_qdot = nb_q
        pose_at_first_node = [0, 0, -0.5336, 0, 1.4, 0, 1.4, 0.8, -0.9, 0.47, 0.8, -0.9, 0.47]

    # Initialize x_bounds (Interpolation type is CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    x_bounds = BoundsList()
    for i in range(nb_phases):
        x_bounds.add(QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping))
    x_bounds[0].min[:, 0] = pose_at_first_node + [0] * nb_qdot
    x_bounds[0].max[:, 0] = pose_at_first_node + [0] * nb_qdot

    # # Initial guess for states (Interpolation type is CONSTANT)
    # x_init = InitialGuessList()
    # for i in range(nb_phases):
    #     x_init.add(pose_at_first_node + [0] * nb_qdot)

    # Initial guess for states (Interpolation type is CONSTANT for 1st phase and SPLINE with 3 key positions for 2nd phase)
    x_init = InitialGuessList()
    t_spline = np.hstack((0, 0.34, phase_time))
    p0 = np.array([pose_at_first_node + [0] * nb_qdot]).T
    p_flex = np.array([[-0.12, -0.23, -1.10, 1.85, 2.06, -1.67, 0.55, 0, 0, 0, 0, 0, 0, 0]]).T
    p_end = p0
    key_positions = np.hstack((p0, p_flex, p_end))
    x_init.add(key_positions, t=t_spline, interpolation=InterpolationType.SPLINE)       # x_init phase 1 type SPLINE

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([[tau_min] * tau_mapping.reduce.len, [tau_max] * tau_mapping.reduce.len], interpolation=InterpolationType.CONSTANT)    # This precision of the CONSTANT type is for informative purposes only

    u_init = InitialGuessList()
    u_init.add([tau_init] * tau_mapping.reduce.len)     # Interpolation type is CONSTANT (default value)

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
        nb_threads=2,
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
    )
    time_min = [0.2]
    time_max = [1]
    phase_time = 0.4
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
    ocp.add_plot(
        "CoM", lambda x, u, p: plot_CoM(x, "../models/jumper2contacts.bioMod"), phase_number=0, plot_type=PlotType.PLOT
    )
    ocp.add_plot(
        "CoM_dot",
        lambda x, u, p: plot_CoM_dot(x, "../models/jumper2contacts.bioMod"),
        phase_number=0,
        plot_type=PlotType.PLOT,
    )

    sol = ocp.solve(show_online_optim=True, solver_options={"hessian_approximation": "exact", "max_iter": 10000})

    #--- Show results --- #
    # param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    # print(
    #     f"The optimized phase time are: {param['time'][0, 0]}s."
    # )

    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    result = ShowResult(ocp, sol)
    result.graphs(adapt_graph_size_to_bounds=False)
    result.animate(nb_frames=61)
