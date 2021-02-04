from time import time

import numpy as np
import biorbd
import math
from casadi import if_else, lt, vertcat

from bioptim import (
    Data,
    PlotType,
    InitialGuess,
    InterpolationType,
    PenaltyNodes,
    Node,
    OptimalControlProgram,
    ConstraintList,
    ConstraintFcn,
    ObjectiveFcn,
    ObjectiveList,
    DynamicsList,
    DynamicsFcn,
    BidirectionalMapping,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    PhaseTransitionList,
    PhaseTransitionFcn,
    ShowResult
)


def no_force_on_heel(nodes: PenaltyNodes):
    return nodes.ocp.nlp[0].contact_forces_func(nodes.x[0], nodes.u[0], nodes.p)[[2, 5], -1]


def no_force_on_toe(nodes: PenaltyNodes):
    return nodes.ocp.nlp[1].contact_forces_func(nodes.x[0], nodes.u[0], nodes.p)[:, -1]


def toe_on_floor(nodes: PenaltyNodes):
    # floor = -0.77865438
    nlp = nodes.nlp

    nb_q = nlp.shape["q"]
    q_reduced = nlp.X[0][:nb_q]
    q = nlp.mapping["q"].to_second.map(q_reduced)
    marker_func = biorbd.to_casadi_func("toe_on_floor", nlp.model.marker, nlp.q, 2)
    toe_marker_z = marker_func(q)[2]
    return toe_marker_z + 0.779


def heel_on_floor(nodes: PenaltyNodes):
    # floor = -0.77865829
    nlp = nodes.nlp

    nb_q = nlp.shape["q"]
    q_reduced = nlp.X[0][:nb_q]
    q = nlp.mapping["q"].to_second.map(q_reduced)
    marker_func = biorbd.to_casadi_func("heel_on_floor", nlp.model.marker, nlp.q, 3)
    tal_marker_z = marker_func(q)[2]
    return tal_marker_z + 0.779


def com_dot_z(nodes: PenaltyNodes):
    nlp = nodes.nlp
    x = nodes.x

    q = nlp.mapping["q"].to_second.map(x[0][: nlp.shape["q"]])
    qdot = nlp.mapping["q"].to_second.map(x[0][nlp.shape["q"]:])
    com_dot_func = biorbd.to_casadi_func("Compute_CoM_dot", nlp.model.CoMdot, nlp.q, nlp.qdot)

    com_dot = com_dot_func(q, qdot)
    return com_dot[2]


def tau_actuator_constraints(nodes: PenaltyNodes, minimal_tau=None):
    nlp = nodes.nlp

    nq = nlp.mapping["q"].to_first.len
    q = [nlp.mapping["q"].to_second.map(mx[:nq]) for mx in nodes.x]
    qdot = [nlp.mapping["qdot"].to_second.map(mx[nq:]) for mx in nodes.x]

    min_bound = []
    max_bound = []

    func = biorbd.to_casadi_func("torqueMax", nlp.model.torqueMax, nlp.q, nlp.qdot)
    for i in range(len(nodes.u)):
        bound = func(q[i], qdot[i])
        if minimal_tau:
            min_bound.append(nlp.mapping["tau"].to_first.map(if_else(lt(bound[:, 1], minimal_tau), minimal_tau, bound[:, 1])))
            max_bound.append(nlp.mapping["tau"].to_first.map(if_else(lt(bound[:, 0], minimal_tau), minimal_tau, bound[:, 0])))
        else:
            min_bound.append(nlp.mapping["tau"].to_first.map(bound[:, 1]))
            max_bound.append(nlp.mapping["tau"].to_first.map(bound[:, 0]))

    obj = vertcat(*nodes.u)
    min_bound = vertcat(*min_bound)
    max_bound = vertcat(*max_bound)

    return (
        vertcat(np.zeros(min_bound.shape), np.ones(max_bound.shape) * -np.inf),
        vertcat(obj + min_bound, obj - max_bound),
        vertcat(np.ones(min_bound.shape) * np.inf, np.zeros(max_bound.shape)),
    )


def plot_com(x, nlp):
    q = nlp.mapping["q"].to_second.map(x[:7, :])
    com_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoM, nlp.q)
    return np.array(com_func(q))[2]


def plot_com_dot(x, nlp):
    q = nlp.mapping["q"].to_second.map(x[:7, :])
    qdot = nlp.mapping["q"].to_second.map(x[7:, :])
    com_dot_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoMdot, nlp.q, nlp.qdot)
    return np.array(com_dot_func(q, qdot))[2]


def plot_torque_bounds(x, min_or_max, nlp, minimal_tau=None):
    q = nlp.mapping["q"].to_second.map(x[:7, :])
    qdot = nlp.mapping["q"].to_second.map(x[7:, :])
    func = biorbd.to_casadi_func("TorqueMax", nlp.model.torqueMax, nlp.q, nlp.qdot)

    res = []
    for dof in [6, 7, 8, 9]:
        bound = []

        for i in range(len(x[0])):
            tmp = func(q[:, i], qdot[:, i])
            if minimal_tau and tmp[dof, min_or_max] < minimal_tau:
                bound.append(minimal_tau)
            else:
                bound.append(tmp[dof, min_or_max])
        res.append(np.array(bound))

    return np.array(res)


def plot_sum_contact_forces(x, u, p, nlp):
    if nlp.contact_forces_func:
        return nlp.contact_forces_func(x, u, p)
    else:
        return np.zeros((20, 1))


def add_custom_plots(ocp, nb_phases, x_bounds, nq, minimal_tau=None):
    for i in range(nb_phases):
        nlp = ocp.nlp[i]
        # Plot Torque Bounds
        if not minimal_tau:
            ocp.add_plot("tau", lambda x, u, p: plot_torque_bounds(x, 0, nlp), phase=i, plot_type=PlotType.STEP, color="g")
            ocp.add_plot("tau", lambda x, u, p: -plot_torque_bounds(x, 1, nlp), phase=i, plot_type=PlotType.STEP, color="g")
        else:
            ocp.add_plot("tau", lambda x, u, p: plot_torque_bounds(x, 0, nlp), phase=i, plot_type=PlotType.STEP, color="g", linestyle="-.")
            ocp.add_plot("tau", lambda x, u, p: -plot_torque_bounds(x, 1, nlp), phase=i, plot_type=PlotType.STEP, color="g", linestyle="-.")
            ocp.add_plot("tau", lambda x, u, p: plot_torque_bounds(x, 0, nlp, minimal_tau=minimal_tau), phase=i, plot_type=PlotType.STEP, color="g")
            ocp.add_plot("tau", lambda x, u, p: -plot_torque_bounds(x, 1, nlp, minimal_tau=minimal_tau), phase=i, plot_type=PlotType.STEP, color="g")
        ocp.add_plot("q_degree", lambda x, u, p: x[:7, :] * 180 / math.pi, phase=i, plot_type=PlotType.INTEGRATED, legend=("q_Pelvis_Trans_Y", "q_Pelvis_Trans_Z", "q_Pelvis_Rot_X", "q_BrasD_Rot_X", "q_CuisseD_Rot_X", "q_JambeD_Rot_X", "q_Pied_Rot_X"))
        ocp.add_plot("qdot_degree", lambda x, u, p: x[7:, :] * 180 / math.pi, phase=i, plot_type=PlotType.INTEGRATED, legend=("qdot_Pelvis_Trans_Y", "qdot_Pelvis_Trans_Z", "qdot_Pelvis_Rot_X", "qdot_BrasD_Rot_X", "qdot_CuisseD_Rot_X", "qdot_JambeD_Rot_X", "qdot_Pied_Rot_X"))
        ocp.add_plot("tau", lambda x, u, p: np.zeros((4, len(x[0]))), phase=i, plot_type=PlotType.STEP, color="b")
        # Plot CoM pos and speed
        ocp.add_plot("CoM", lambda x, u, p: plot_com(x, nlp), phase=i, plot_type=PlotType.PLOT)
        ocp.add_plot("CoM_dot", lambda x, u, p: plot_com_dot(x, nlp), phase=i, plot_type=PlotType.PLOT)
        # Plot q and nb_qdot ranges
        ocp.add_plot(
            "q",
            lambda x, u, p: np.repeat(x_bounds[i].min[:nq, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "q",
            lambda x, u, p: np.repeat(x_bounds[i].max[:nq, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "qdot",
            lambda x, u, p: np.repeat(x_bounds[i].min[nq:, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "qdot",
            lambda x, u, p: np.repeat(x_bounds[i].max[nq:, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "q_degree",
            lambda x, u, p: np.repeat(x_bounds[i].min[:nq, 1][:, np.newaxis], x.shape[1], axis=1) * 180 / math.pi,
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "q_degree",
            lambda x, u, p: np.repeat(x_bounds[i].max[:nq, 1][:, np.newaxis], x.shape[1], axis=1) * 180 / math.pi,
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "qdot_degree",
            lambda x, u, p: np.repeat(x_bounds[i].min[nq:, 1][:, np.newaxis], x.shape[1], axis=1) * 180 / math.pi,
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "qdot_degree",
            lambda x, u, p: np.repeat(x_bounds[i].max[nq:, 1][:, np.newaxis], x.shape[1], axis=1) * 180 / math.pi,
            phase=i,
            plot_type=PlotType.PLOT,
        )


def warm_start_nmpc(sol, ocp):
    state, ctrl, param = Data.get_data(ocp, sol, concatenate=False, get_parameters=True)
    u_init = InitialGuessList()
    x_init = InitialGuessList()
    for i in range(ocp.n_phases):
        if ocp.n_phases == 1:
            u_init.add(np.concatenate([ctrl[d][:, :-1] for d in ctrl]), interpolation=InterpolationType.EACH_FRAME)
            x_init.add(np.concatenate([state[d] for d in state]), interpolation=InterpolationType.EACH_FRAME)
        else:
            u_init.add(np.concatenate([ctrl[d][i][:, :-1] for d in ctrl]), interpolation=InterpolationType.EACH_FRAME)
            x_init.add(np.concatenate([state[d][i] for d in state]), interpolation=InterpolationType.EACH_FRAME)

    time = InitialGuess(param["time"], name="time")
    ocp.update_initial_guess(x_init=x_init, u_init=u_init, param_init=time)
    ocp.solver.set_lagrange_multiplier(sol)


def optimize_jumping_ocp(model_path, phase_time, ns, time_min, time_max, init=None):
    biorbd_model = [biorbd.Model(elt) for elt in model_path]
    n_phases = len(biorbd_model)
    takeoff_phase = 0 if n_phases == 1 else 1  # Flat foot jump or tip toe jump
    pose_at_first_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]
    flat_foot_phases = []
    toe_only_phases = []

    q_mapping = BidirectionalMapping([0, 1, 2, None, 3, None, 3, 4, 5, 6, 4, 5, 6], [0, 1, 2, 4, 7, 8, 9])
    q_mapping = [q_mapping for _ in range(n_phases)]
    tau_mapping = BidirectionalMapping([None, None, None, None, 0, None, 0, 1, 2, 3, 1, 2, 3], [4, 7, 8, 9])
    tau_mapping = [tau_mapping for _ in range(n_phases)]
    n_q = q_mapping[0].to_first.len
    n_qdot = n_q
    n_tau = tau_mapping[0].to_first.len

    dynamics = DynamicsList()
    flat_foot_phases.append(0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Flat foot
    final_phase = -1
    if n_phases >= 2:
        toe_only_phases.append(1)
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Toe only
    if n_phases >= 3:
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)  # Aerial phase
    if n_phases >= 4:
        toe_only_phases.append(3)
        final_phase = 3
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Toe only
    if n_phases >= 5:
        flat_foot_phases.append(4)
        final_phase = 4
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Flat foot

    # Declare some lists to fill
    objective_functions = ObjectiveList()
    x_bounds = BoundsList()
    constraints = ConstraintList()
    x_init = InitialGuessList()
    u_bounds = BoundsList()
    u_init = InitialGuessList()

    # Maximize the jump height
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-100, phase=takeoff_phase)
    if n_phases >= 3:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE_DERIVATIVE, weight=0.1, phase=2, index=range(7, 14))

    # Positivity of CoM_dot on z axis prior the take-off
    constraints.add(com_dot_z, phase=takeoff_phase, node=Node.END, min_bound=0, max_bound=np.inf)

    # Constraint arm positivity (prevent from local minimum with arms in the back)
    constraints.add(ConstraintFcn.TRACK_STATE, phase=takeoff_phase, node=Node.END, index=3, min_bound=1.0, max_bound=np.inf)

    # Floor constraints
    for p in flat_foot_phases:
        # Do not pull on floor
        for i in (1, 2, 4, 5):  # flat foot
            constraints.add(ConstraintFcn.CONTACT_FORCE, phase=p, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)

        # Non-slipping constraints
        constraints.add(  # flat foot, on only one of the feet
            ConstraintFcn.NON_SLIPPING,
            phase=p,
            node=Node.ALL,
            normal_component_idx=(1, 2),
            tangential_component_idx=0,
            static_friction_coefficient=0.5,
        )

    for p in toe_only_phases:
        # Do not pull on floor
        for i in (1, 3):  # toe only
            constraints.add(ConstraintFcn.CONTACT_FORCE, phase=p, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)

        # Non-slipping constraints
        constraints.add(  # toe only, on only one of the feet
            ConstraintFcn.NON_SLIPPING,
            phase=p,
            node=Node.ALL,
            normal_component_idx=1,
            tangential_component_idx=0,
            static_friction_coefficient=0.5,
        )

        # Heel over the floor
        # constraints.add(heel_on_floor, phase=p, node=Node.ALL, min_bound=-0.0001, max_bound=np.inf)

    for i in range(n_phases):
        # Minimize time of the phase
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.1, phase=i, min_bound=time_min[i], max_bound=time_max[i]
        )

        # Torque constrained to torqueMax
        constraints.add(tau_actuator_constraints, phase=i, node=Node.ALL, minimal_tau=30)

        # Path constraints
        x_bounds.add(bounds=QAndQDotBounds(biorbd_model[i], q_mapping=q_mapping[i], qdot_mapping=q_mapping[i]))
        u_bounds.add([-500] * n_tau, [500] * n_tau)

        # Initial guesses
        x_init.add(pose_at_first_node + [0] * n_qdot)
        if init is not None:
            u_init.add(init)
        else:
            u_init.add([0] * n_tau)

    # Enforce the initial and final pose (except for translation at final)
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot
    if final_phase >= 3:
        # pass
        # constraints.add(ConstraintFcn.TRACK_STATE, phase=4, node=Node.END, min_bound=-0.1, max_bound=0.1, target=pose_at_first_node[2:] + [0] * n_qdot, index=range(2, 14))
        x_bounds[final_phase].min[2:, -1] = np.concatenate((np.array(pose_at_first_node[2:]) - 0.01, [0] * n_qdot))
        x_bounds[final_phase].max[2:, -1] = np.concatenate((np.array(pose_at_first_node[2:]) + 0.01, [0] * n_qdot))

    # Phase transition
    phase_transitions = PhaseTransitionList()
    if n_phases >= 2:
        phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)
    if n_phases >= 3:
        phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)
    if n_phases >= 4:  # Phase transition at toe strike
        phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=2)
        constraints.add(toe_on_floor, phase=3, node=Node.START, min_bound=-0.001, max_bound=0.001)
        x_bounds[3].min[n_q:, 0] = 2 * x_bounds[3].min[n_q:, 0]  # Allow for passive velocity at reception
        x_bounds[3].max[n_q:, 0] = 2 * x_bounds[3].max[n_q:, 0]
    if n_phases >= 5:  # Phase transition at heel strike
        phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)
        constraints.add(heel_on_floor, phase=4, node=Node.START, min_bound=-0.001, max_bound=0.001)
        x_bounds[4].min[n_q:, 0] = 2 * x_bounds[4].min[n_q:, 0]  # Allow for passive velocity at reception
        x_bounds[4].max[n_q:, 0] = 2 * x_bounds[4].max[n_q:, 0]

    ocp = OptimalControlProgram(
        biorbd_model,
        dynamics,
        ns,
        phase_time,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        q_mapping=q_mapping,
        qdot_mapping=q_mapping,
        tau_mapping=tau_mapping,
        phase_transitions=phase_transitions,
        n_threads=4,
    )
    add_custom_plots(ocp, n_phases, x_bounds, n_q, minimal_tau=20)

    # Run optimizations
    tic = time()
    sol = ocp.solve(
        show_online_optim=False,
        solver_options={"hessian_approximation": "limited-memory", "max_iter": 200}
    )
    warm_start_nmpc(sol, ocp)
    sol = ocp.solve(
        show_online_optim=True,
        solver_options={"hessian_approximation": "exact",
                        "max_iter": 1000,
                        "warm_start_init_point": "yes",
                        }
    )

    ShowResult(ocp, sol).objective_functions()
    ShowResult(ocp, sol).constraints()
    print(f"Time to solve : {time() - tic}sec")

    return ocp, sol
