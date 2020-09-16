from time import time

import numpy as np
import biorbd
from casadi import Function, MX

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    Constraint,
    ConstraintList,
    Objective,
    ObjectiveList,
    DynamicsType,
    DynamicsTypeList,
    BidirectionalMapping,
    Mapping,
    StateTransition,
    StateTransitionList,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsList,
    InterpolationType,
    ShowResult,
    Data,
    PlotType,
)

# def constraint_2c_to_1c_transition(ocp, nlp, t, x, u, p):
#     val = ocp.nlp[0]["contact_forces_func"](x[0], u[0], p)[[2, 5], -1]
#     return val


def from_2contacts_to_1(ocp, nlp, t, x, u, p):
    return ocp.nlp[0]["contact_forces_func"](x[0], u[0], p)[[2, 5], -1]


def from_1contact_to_0(ocp, nlp, t, x, u, p):
    return ocp.nlp[1]["contact_forces_func"](x[0], u[0], p)[:, -1]


def from_0contact_to_1(ocp, nlp, t, x, u, p):
    nbQ_reduced = nlp["nbQ"]
    q_reduced = nlp["X"][0][:nbQ_reduced]
    q = nlp["q_mapping"].expand.map(q_reduced)
    toeD_marker_z = nlp["model"].marker(q, 2).to_mx()[2]
    return (
        toeD_marker_z + 0.77865438
    )  # -0.77865438 is the value returned with Eigen by the 3rd dim. of toeD_marker CoM at pose_at_first_node


def from_1contact_to_2(ocp, nlp, t, x, u, p):
    nbQ_reduced = nlp["nbQ"]
    q_reduced = nlp["X"][0][:nbQ_reduced]
    q = nlp["q_mapping"].expand.map(q_reduced)
    talD_marker_z = nlp["model"].marker(q, 3).to_mx()[2]
    return (
        talD_marker_z + 0.77865829
    )  # -0.77865829 is the value returned with Eigen by the 3rd dim. of toeD_marker CoM at pose_at_first_node


# def phase_transition_1c_to_2c(nlp_pre, nlp_post):
#     nbQ = nlp_post["nbQ"]
#     q_reduced = nlp_post["X"][0][:nbQ]
#     q = nlp_post["q_mapping"].expand.map(q_reduced)
#     talD_marker = nlp_post["model"].marker(q, 3).to_mx()
#     return talD_marker


# def custom_func_anatomical_constraint(ocp, nlp, t, x, u, p):
#     val = x[0][7:14]
#     return val


def prepare_ocp(model_path, phase_time, number_shooting_points, use_symmetry=True, use_torque_activation=True):
    # --- Options --- #
    # Model path
    biorbd_model = [biorbd.Model(elt) for elt in model_path]

    nb_phases = len(biorbd_model)
    if use_torque_activation:
        tau_min, tau_max, tau_init = -1, 1, 0
    else:
        tau_min, tau_max, tau_init = -500, 500, 0

    if use_symmetry:
        q_mapping = BidirectionalMapping(
            Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
        )
        q_mapping = q_mapping, q_mapping, q_mapping, q_mapping, q_mapping
        tau_mapping = BidirectionalMapping(
            Mapping([-1, -1, -1, -1, 0, -1, 0, 1, 2, 3, 1, 2, 3], [5]), Mapping([4, 7, 8, 9])
        )
        tau_mapping = tau_mapping, tau_mapping, tau_mapping, tau_mapping, tau_mapping

    else:
        q_mapping = BidirectionalMapping(
            Mapping([i for i in range(biorbd_model[0].nbQ())]), Mapping([i for i in range(biorbd_model[0].nbQ())])
        )
        q_mapping = q_mapping, q_mapping, q_mapping, q_mapping, q_mapping
        tau_mapping = q_mapping

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1, phase=1)

    # Dynamics
    dynamics = DynamicsTypeList()
    if use_torque_activation:
        dynamics.add(DynamicsType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsType.TORQUE_DRIVEN)
        dynamics.add(DynamicsType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT)
    else:
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsType.TORQUE_DRIVEN)
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)

    # --- Constraints --- #
    constraints = ConstraintList()

    # Positivity constraints of the normal component of the reaction forces
    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        constraints.add(Constraint.CONTACT_FORCE_INEQUALITY, phase=0, direction="GREATER_THAN", instant=Instant.ALL, contact_force_idx=i, boundary=0)
        constraints.add(Constraint.CONTACT_FORCE_INEQUALITY, phase=4, direction="GREATER_THAN", instant=Instant.ALL, contact_force_idx=i, boundary=0)
    contact_axes = (1, 3)
    for i in contact_axes:
        constraints.add(Constraint.CONTACT_FORCE_INEQUALITY, phase=1, direction="GREATER_THAN", instant=Instant.ALL, contact_force_idx=i, boundary=0)
        constraints.add(Constraint.CONTACT_FORCE_INEQUALITY, phase=3, direction="GREATER_THAN", instant=Instant.ALL, contact_force_idx=i, boundary=0)

    # Non-slipping constraints
    # N.B.: Application on only one of the two feet is sufficient, as the slippage cannot occurs on only one foot.
    constraints.add(Constraint.NON_SLIPPING, phase=0, instant=Instant.ALL, normal_component_idx=(1, 2), tangential_component_idx=0, static_friction_coefficient=0.5)
    constraints.add(Constraint.NON_SLIPPING, phase=1, instant=Instant.ALL, normal_component_idx=1, tangential_component_idx=0, static_friction_coefficient=0.5)
    constraints.add(Constraint.NON_SLIPPING, phase=3, instant=Instant.ALL, normal_component_idx=1, tangential_component_idx=0, static_friction_coefficient=0.5)
    constraints.add(Constraint.NON_SLIPPING, phase=4, instant=Instant.ALL, normal_component_idx=(1, 2), tangential_component_idx=0, static_friction_coefficient=0.5)

    # Custom constraints for contact forces at transitions
    constraints.add(from_2contacts_to_1, phase=1, instant=Instant.START)
    constraints.add(from_1contact_to_0, phase=2, instant=Instant.START)
    constraints.add(from_0contact_to_1, phase=3, instant=Instant.START)
    constraints.add(from_1contact_to_2, phase=4, instant=Instant.START)

    if not use_symmetry:
        raise NotImplementedError("Need to adapt to recent refactors")
        # first_dof = (3, 4, 7, 8, 9)
        # second_dof = (5, 6, 10, 11, 12)
        # coef = (-1, 1, 1, 1, 1)
        # for i in range(len(first_dof)):
        #     for elt in [
        #         constraints_first_phase,
        #         constraints_second_phase,
        #         constraints_third_phase,
        #         constraints_fourth_phase,
        #         constraints_fifth_phase,
        #     ]:
        #         elt.append(
        #             {
        #                 "type": Constraint.PROPORTIONAL_STATE,
        #                 "instant": Instant.ALL,
        #                 "first_dof": first_dof[i],
        #                 "second_dof": second_dof[i],
        #                 "coef": coef[i],
        #             }
        #         )

    # # Time constraint
    # for i in range(nb_phases):
    #     constraints.add(Constraint.TIME_CONSTRAINT, phase=i, minimum=time_min[i], maximum=time_max[i])

    # State transitions
    state_transitions = StateTransitionList()
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=2)

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
    x_bounds[4].min[:, -1] = pose_at_first_node + [0] * nb_qdot
    x_bounds[4].max[:, -1] = pose_at_first_node + [0] * nb_qdot

    # # Initial guess for states (Interpolation type is CONSTANT)
    # x_init = InitialConditionsList()
    # for i in range(nb_phases):
    #     x_init.add(pose_at_first_node + [0] * nb_qdot)

    # Initial guess for states (Interpolation type is CONSTANT all phases except the SPLINE type with 3 key positions for 2nd phase)
    x_init = InitialConditionsList()
    x_init.add(pose_at_first_node + [0] * nb_qdot)  # x_init phase 0 type CONSTANT
    t_spline = np.hstack((0, 0.34, phase_time[1]))
    p0 = np.array([pose_at_first_node + [0] * nb_qdot]).T
    p_flex = np.array([[-0.12, -0.23, -1.10, 1.85, 2.06, -1.67, 0.55, 0, 0, 0, 0, 0, 0, 0]]).T
    p_end = p0
    key_positions = np.hstack((p0, p_flex, p_end))
    x_init.add(key_positions, t=t_spline, interpolation=InterpolationType.SPLINE)
    x_init.add(pose_at_first_node + [0] * nb_qdot)  # x_init phase 2 type CONSTANT
    x_init.add(pose_at_first_node + [0] * nb_qdot)  # x_init phase 3 type CONSTANT
    x_init.add(pose_at_first_node + [0] * nb_qdot)  # x_init phase 4 type CONSTANT

    # Define control path constraint
    u_bounds = BoundsList()
    for tau_m in tau_mapping:
        u_bounds.add([[tau_min] * tau_m.reduce.len, [tau_max] * tau_m.reduce.len], interpolation=InterpolationType.CONSTANT)  # This precision of the CONSTANT type is for informative purposes only

    # Define initial guess for controls
    u_init = InitialConditionsList()
    for tau_m in tau_mapping:
        u_init.add([tau_init] * tau_m.reduce.len)  # Interpolation type is CONSTANT (default value)

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
        state_transitions=state_transitions,
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

    q_sym = MX.sym("q", m.nbQ(), 1)
    qdot_sym = MX.sym("q_dot", m.nbQdot(), 1)
    CoM_dot_func = Function(
        "Compute_CoM_dot", [q_sym, qdot_sym], [m.CoMdot(q_sym, qdot_sym).to_mx()], ["q", "q_dot"], ["CoM_dot"],
    ).expand()
    CoM_dot = np.array(CoM_dot_func(q_expanded[:, :], qdot_expanded[:, :]))
    return CoM_dot[2]


def run_and_save_ocp(model_path, phase_time, number_shooting_points):
    ocp = prepare_ocp(
        model_path=model_path, phase_time=phase_time, number_shooting_points=number_shooting_points, use_symmetry=True, use_torque_activation=False
    )
    for i in range(len(model_path)):
        ocp.add_plot("CoM", lambda x, u, p: plot_CoM(x, model_path[i]), phase_number=i, plot_type=PlotType.PLOT)
        ocp.add_plot("CoM_dot", lambda x, u, p: plot_CoM_dot(x, model_path[i]), phase_number=i, plot_type=PlotType.PLOT)
    sol = ocp.solve(solver_options={"hessian_approximation": "exact", "max_iter": 10000}, show_online_optim=True)

    OptimalControlProgram.save(ocp, sol, "../Results/jumper5phases_exact_sol")
    return ocp, sol


if __name__ == "__main__":
    use_saved_results = True

    model_path = (
        "../models/jumper2contacts.bioMod",
        "../models/jumper1contacts.bioMod",
        "../models/jumper1contacts.bioMod",
        "../models/jumper1contacts.bioMod",
        "../models/jumper2contacts.bioMod",
    )
    # time_min = [0.1, 0.3, 0.2, 0.1, 0.1]
    # time_max = [0.4, 0.6, 2, 0.4, 0.4]
    phase_time = [0.2, 0.4, 1, 0.3, 0.3]
    number_shooting_points = [20, 20, 20, 20, 20]

    tic = time()

    if use_saved_results:
        ocp, sol = OptimalControlProgram.load("../Results/jumper5phases_exact_sol.bo")
    else:
        ocp, sol = run_and_save_ocp(model_path, phase_time=phase_time, number_shooting_points=number_shooting_points)

    # # --- Show results --- #
    # param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    # print(
    #     f"The optimized phases times are: {param['time'][0, 0]}s, {param['time'][1, 0]}s, {param['time'][2, 0]}s, {param['time'][3, 0]}s and {param['time'][4, 0]}s."
    # )

    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate(nb_frames=61)
