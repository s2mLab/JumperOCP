from time import time

import numpy as np
import biorbd
from casadi import vertcat, MX, Function

from bioptim import (
    Node,
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
        "Compute_CoM_dot",
        [q_sym, qdot_sym],
        [nlp.model.CoMdot(q_sym, qdot_sym).to_mx()],
        ["q", "q_dot"],
        ["CoM_dot"],
    ).expand()
    CoM_dot = CoM_dot_func(q_expanded, qdot_expanded)
    return CoM_dot[2]


def Torque_Constraint(ocp, nlp, t, x, u, p):
    nq = nlp.mapping["q"].reduce.len
    q = [nlp.mapping["q"].expand.map(mx[:nq]) for mx in x]
    qdot = [nlp.mapping["q_dot"].expand.map(mx[nq:]) for mx in x]

    min_bound = []
    max_bound = []

    func = biorbd.to_casadi_func("torqueMax", nlp.model.torqueMax, nlp.q, nlp.q_dot)
    for i in range(len(u)):
        bound = func(q[i], qdot[i])
        min_bound.append(nlp.mapping["tau"].reduce.map(bound[:, 1]))
        max_bound.append(nlp.mapping["tau"].reduce.map(bound[:, 0]))

    obj = vertcat(*u)
    min_bound = vertcat(*min_bound)
    max_bound = vertcat(*max_bound)

    return vertcat(np.zeros(min_bound.shape), np.ones(max_bound.shape) * -np.inf),\
            vertcat(obj + min_bound, obj - max_bound),\
            vertcat(np.ones(min_bound.shape) * np.inf, np.zeros(max_bound.shape))


def prepare_ocp(model_path, phase_time, number_shooting_points, time_min, time_max, use_symmetry=True, use_actuators=True):
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
        q_mapping = q_mapping, q_mapping, q_mapping
        tau_mapping = BidirectionalMapping(
            Mapping([-1, -1, -1, -1, 0, -1, 0, 1, 2, 3, 1, 2, 3], [5]), Mapping([4, 7, 8, 9])
        )
        tau_mapping = tau_mapping, tau_mapping, tau_mapping

    else:
        q_mapping = BidirectionalMapping(
            Mapping([i for i in range(biorbd_model[0].nbQ())]), Mapping([i for i in range(biorbd_model[0].nbQ())])
        )
        q_mapping = q_mapping, q_mapping, q_mapping
        tau_mapping = q_mapping

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1, phase=1)

    # Dynamics
    dynamics = DynamicsTypeList()
    if use_actuators:
        dynamics.add(DynamicsType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsType.TORQUE_DRIVEN)
    else:
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # --- Constraints --- #
    constraints = ConstraintList()

    # Positivity constraints of the normal component of the reaction forces
    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        constraints.add(
            Constraint.CONTACT_FORCE,
            phase=0,
            node=Node.ALL,
            contact_force_idx=i,
            min_bound=0,
            max_bound=np.inf
        )
    contact_axes = (1, 3)
    for i in contact_axes:
        constraints.add(Constraint.CONTACT_FORCE, phase=1, node=Node.ALL, contact_force_idx=i, min_bound=0, max_bound=np.inf)

    # Non-slipping constraints
    # N.B.: Application on only one of the two feet is sufficient, as the slippage cannot occurs on only one foot.
    constraints.add(
        Constraint.NON_SLIPPING,
        phase=0,
        node=Node.ALL,
        normal_component_idx=(1, 2),
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
    )
    constraints.add(
        Constraint.NON_SLIPPING,
        phase=1,
        node=Node.ALL,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5
    )

    # Custom constraints for contact forces at transitions
    # constraints_second_phase.append(
    #     {"type": Constraint.CUSTOM, "function": from_2contacts_to_1, "node": Node.START}
    # )

    # Custom constraints for positivity of CoM_dot on z axis just before the take-off
    constraints.add(CoM_dot_Z_last_node_positivity, phase=1, node=Node.END, min_bound=0, max_bound=np.inf)

    for i in range(nb_phases):
        constraints.add(Torque_Constraint, phase=i, node=Node.ALL)

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
    #                     "node": Node.ALL,
    #                     "first_dof": first_dof[i],
    #                     "second_dof": second_dof[i],
    #                     "coef": coef[i],
    #                 }
    #             )

    # constraints = (
    #     constraints_first_phase,
    #     constraints_second_phase,
    # )

    # Time constraint
    for i in range(nb_phases):
        #objective_functions.add(Objective.Lagrange.MINIMIZE_TIME, weight=0.0001, phase=i)#, min_bound=time_min[i], max_bound=time_max[i])
        objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=0.0001, phase=i)

    # --- Path constraints --- #
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

    # # Initial guess for states (Interpolation type is CONSTANT)
    # x_init = InitialGuessList()
    # for i in range(nb_phases):
    #     x_init.add(pose_at_first_node + [0] * nb_qdot)

    # Initial guess for states (Interpolation type is CONSTANT for 1st phase and SPLINE with 3 key positions for 2nd phase)
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * nb_qdot)      # x_init phase 0 type CONSTANT
    t_spline = np.hstack((0, 0.34, phase_time[1]))
    p0 = np.array([pose_at_first_node + [0] * nb_qdot]).T
    p_flex = np.array([[-0.12, -0.23, -1.10, 1.85, 2.06, -1.67, 0.55, 0, 0, 0, 0, 0, 0, 0]]).T
    p_end = p0
    key_positions = np.hstack((p0, p_flex, p_end))
    x_init.add(key_positions, t=t_spline, interpolation=InterpolationType.SPLINE)       # x_init phase 1 type SPLINE
    x_init.add(pose_at_first_node + [0] * nb_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    for tau_m in tau_mapping:
        u_bounds.add([[tau_min] * tau_m.reduce.len, [tau_max] * tau_m.reduce.len], interpolation=InterpolationType.CONSTANT)    # This precision of the CONSTANT type is for informative purposes only

    u_init = InitialGuessList()
    for tau_m in tau_mapping:
        u_init.add([tau_init] * tau_m.reduce.len)     # Interpolation type is CONSTANT (default value)

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


def plot_torque_bounds(x, min_or_max, model_path):
    model = biorbd.Model(model_path)
    q_mapping = BidirectionalMapping(
        Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
    )

    q_sym = MX.sym("q", model.nbQ(), 1)
    qdot_sym = MX.sym("qdot", model.nbQdot(), 1)
    func = biorbd.to_casadi_func("TorqueMax", model.torqueMax, q_sym, qdot_sym)

    q_reduced = x[:7, :]
    qdot_reduced = x[7:, :]
    q_expanded = q_mapping.expand.map(q_reduced)
    qdot_expanded = q_mapping.expand.map(qdot_reduced)

    res = []
    for dof in [6, 7, 8, 9]:
        bound = []

        for i in range(len(x[0])):
            tmp = func(q_expanded[:, i], qdot_expanded[:, i])
            bound.append(tmp[dof, min_or_max])
        res.append(np.array(bound))

    return np.array(res)

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
        "../models/jumper1contacts.bioMod",
    )
    time_min = [0.1, 0.0]
    time_max = [1, 1]
    phase_time = [0.6, 0.2, 1]
    number_shooting_points = [30, 30, 30]

    tic = time()
    # run_and_save_ocp(model_path, phase_time=phase_time, number_shooting_points=number_shooting_points)
    # ocp, sol = OptimalControlProgram.load("../Results/jumper2phases_sol.bo")

    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=phase_time,
        number_shooting_points=number_shooting_points,
        time_min=time_min,
        time_max=time_max,
        use_symmetry=True,
        use_actuators=False,
    )
    # Plot Torque Bounds
    ocp.add_plot(
        "tau", lambda x, u, p: plot_torque_bounds(x, 0, "../models/jumper2contacts.bioMod"), phase_number=0, plot_type=PlotType.STEP, color='g'
    )
    ocp.add_plot(
        "tau", lambda x, u, p: -plot_torque_bounds(x, 1, "../models/jumper2contacts.bioMod"), phase_number=0, plot_type=PlotType.STEP, color='g'
    )
    ocp.add_plot(
        "tau", lambda x, u, p: plot_torque_bounds(x, 0, "../models/jumper2contacts.bioMod"), phase_number=1, plot_type=PlotType.STEP, color='g'
    )
    ocp.add_plot(
        "tau", lambda x, u, p: -plot_torque_bounds(x, 1, "../models/jumper2contacts.bioMod"), phase_number=1, plot_type=PlotType.STEP, color='g'
    )
    ocp.add_plot(
        "tau", lambda x, u, p: plot_torque_bounds(x, 0, "../models/jumper1contacts.bioMod"), phase_number=2, plot_type=PlotType.STEP, color='g'
    )
    ocp.add_plot(
        "tau", lambda x, u, p: -plot_torque_bounds(x, 1, "../models/jumper1contacts.bioMod"), phase_number=2, plot_type=PlotType.STEP, color='g'
    )
    # Plot CoM pos and speed
    ocp.add_plot(
        "CoM", lambda x, u, p: plot_CoM(x, "../models/jumper2contacts.bioMod"), phase_number=0, plot_type=PlotType.PLOT
    )
    ocp.add_plot(
        "CoM", lambda x, u, p: plot_CoM(x, "../models/jumper1contacts.bioMod"), phase_number=1, plot_type=PlotType.PLOT
    )
    ocp.add_plot(
        "CoM", lambda x, u, p: plot_CoM(x, "../models/jumper1contacts.bioMod"), phase_number=2, plot_type=PlotType.PLOT
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
    ocp.add_plot(
        "CoM_dot",
        lambda x, u, p: plot_CoM_dot(x, "../models/jumper1contacts.bioMod"),
        phase_number=2,
        plot_type=PlotType.PLOT,
    )
    # Plot q and qdot ranges
    ocp.add_plot(
        "q",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * -10,
            np.ones(len(x[0])) * -10,
            np.ones(len(x[0])) * -30,
            np.ones(len(x[0])) * -0.7,
            np.ones(len(x[0])) * -0.4,
            np.ones(len(x[0])) * -2.3,
            np.ones(len(x[0])) * -0.7
            ]),
        phase_number=0,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * 10,
            np.ones(len(x[0])) * 10,
            np.ones(len(x[0])) * 30,
            np.ones(len(x[0])) * 3.1,
            np.ones(len(x[0])) * 2.6,
            np.ones(len(x[0])) * 0.02,
            np.ones(len(x[0])) * 0.7
            ]),
        phase_number=0,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q_dot",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * -20,
            np.ones(len(x[0])) * -20,
            np.ones(len(x[0])) * -15,
            np.ones(len(x[0])) * -17,
            np.ones(len(x[0])) * -10,
            np.ones(len(x[0])) * -13,
            np.ones(len(x[0])) * -17
            ]),
        phase_number=0,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q_dot",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * 20,
            np.ones(len(x[0])) * 20,
            np.ones(len(x[0])) * 15,
            np.ones(len(x[0])) * 17,
            np.ones(len(x[0])) * 8,
            np.ones(len(x[0])) * 20,
            np.ones(len(x[0])) * 17
            ]),
        phase_number=0,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * -10,
            np.ones(len(x[0])) * -10,
            np.ones(len(x[0])) * -30,
            np.ones(len(x[0])) * -0.7,
            np.ones(len(x[0])) * -0.4,
            np.ones(len(x[0])) * -2.3,
            np.ones(len(x[0])) * -0.7
            ]),
        phase_number=1,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * 10,
            np.ones(len(x[0])) * 10,
            np.ones(len(x[0])) * 30,
            np.ones(len(x[0])) * 3.1,
            np.ones(len(x[0])) * 2.6,
            np.ones(len(x[0])) * 0.02,
            np.ones(len(x[0])) * 0.7
            ]),
        phase_number=1,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q_dot",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * -20,
            np.ones(len(x[0])) * -20,
            np.ones(len(x[0])) * -15,
            np.ones(len(x[0])) * -17,
            np.ones(len(x[0])) * -10,
            np.ones(len(x[0])) * -13,
            np.ones(len(x[0])) * -17
            ]),
        phase_number=1,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q_dot",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * 20,
            np.ones(len(x[0])) * 20,
            np.ones(len(x[0])) * 15,
            np.ones(len(x[0])) * 17,
            np.ones(len(x[0])) * 8,
            np.ones(len(x[0])) * 20,
            np.ones(len(x[0])) * 17
            ]),
        phase_number=1,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * -10,
            np.ones(len(x[0])) * -10,
            np.ones(len(x[0])) * -30,
            np.ones(len(x[0])) * -0.7,
            np.ones(len(x[0])) * -0.4,
            np.ones(len(x[0])) * -2.3,
            np.ones(len(x[0])) * -0.7
            ]),
        phase_number=2,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * 10,
            np.ones(len(x[0])) * 10,
            np.ones(len(x[0])) * 30,
            np.ones(len(x[0])) * 3.1,
            np.ones(len(x[0])) * 2.6,
            np.ones(len(x[0])) * 0.02,
            np.ones(len(x[0])) * 0.7
            ]),
        phase_number=2,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q_dot",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * -20,
            np.ones(len(x[0])) * -20,
            np.ones(len(x[0])) * -15,
            np.ones(len(x[0])) * -17,
            np.ones(len(x[0])) * -10,
            np.ones(len(x[0])) * -13,
            np.ones(len(x[0])) * -17
            ]),
        phase_number=2,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q_dot",
        lambda x, u, p: np.array([
            np.ones(len(x[0])) * 20,
            np.ones(len(x[0])) * 20,
            np.ones(len(x[0])) * 15,
            np.ones(len(x[0])) * 17,
            np.ones(len(x[0])) * 8,
            np.ones(len(x[0])) * 20,
            np.ones(len(x[0])) * 17
            ]),
        phase_number=2,
        plot_type=PlotType.PLOT,
    )

    sol = ocp.solve(show_online_optim=True, solver_options={"hessian_approximation": "exact", "max_iter": 1000})

    #--- Show results --- #
    # param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    # print(
    #     f"The optimized phases times are: {param['time'][0, 0]}s and {param['time'][1, 0]}s."
    # )

    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    result = ShowResult(ocp, sol)
    result.graphs(adapt_graph_size_to_bounds=True)
    result.animate(nb_frames=61)
