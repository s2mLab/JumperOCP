import numpy as np
import biorbd
from casadi import if_else, lt, vertcat

from bioptim import (
    Data,
    PlotType,
    InitialGuessOption,
    InitialGuessList,
    InterpolationType,
)


def no_force_on_heel(ocp, nlp, t, x, u, p):
    return ocp.nlp[0].contact_forces_func(x[0], u[0], p)[[2, 5], -1]


def no_force_on_toe(ocp, nlp, t, x, u, p):
    return ocp.nlp[1].contact_forces_func(x[0], u[0], p)[:, -1]


def toe_on_floor(ocp, nlp, t, x, u, p):
    # floor = -0.77865438
    nb_q = nlp.shape["q"]
    q_reduced = nlp.X[0][:nb_q]
    q = nlp.mapping["q"].expand.map(q_reduced)
    marker_func = biorbd.to_casadi_func("toe_on_floor", nlp.model.marker, nlp.q, 2)
    toe_marker_z = marker_func(q)[2]
    return toe_marker_z + 0.779


def heel_on_floor(ocp, nlp, t, x, u, p):
    # floor = -0.77865829
    nb_q = nlp.shape["q"]
    q_reduced = nlp.X[0][:nb_q]
    q = nlp.mapping["q"].expand.map(q_reduced)
    marker_func = biorbd.to_casadi_func("heel_on_floor", nlp.model.marker, nlp.q, 3)
    tal_marker_z = marker_func(q)[2]
    return tal_marker_z + 0.779


def com_dot_z(ocp, nlp, t, x, u, p):
    q = nlp.mapping["q"].expand.map(x[0][: nlp.shape["q"]])
    q_dot = nlp.mapping["q"].expand.map(x[0][nlp.shape["q"]:])
    com_dot_func = biorbd.to_casadi_func("Compute_CoM_dot", nlp.model.CoMdot, nlp.q, nlp.q_dot)

    com_dot = com_dot_func(q, q_dot)
    return com_dot[2]


def tau_actuator_constraints(ocp, nlp, t, x, u, p, minimal_tau=None):
    nq = nlp.mapping["q"].reduce.len
    q = [nlp.mapping["q"].expand.map(mx[:nq]) for mx in x]
    q_dot = [nlp.mapping["q_dot"].expand.map(mx[nq:]) for mx in x]

    min_bound = []
    max_bound = []

    func = biorbd.to_casadi_func("torqueMax", nlp.model.torqueMax, nlp.q, nlp.q_dot)
    for i in range(len(u)):
        bound = func(q[i], q_dot[i])
        if minimal_tau:
            min_bound.append(nlp.mapping["tau"].reduce.map(if_else(lt(bound[:, 1], minimal_tau), minimal_tau, bound[:, 1])))
            max_bound.append(nlp.mapping["tau"].reduce.map(if_else(lt(bound[:, 0], minimal_tau), minimal_tau, bound[:, 0])))
        else:
            min_bound.append(nlp.mapping["tau"].reduce.map(bound[:, 1]))
            max_bound.append(nlp.mapping["tau"].reduce.map(bound[:, 0]))

    obj = vertcat(*u)
    min_bound = vertcat(*min_bound)
    max_bound = vertcat(*max_bound)

    return (
        vertcat(np.zeros(min_bound.shape), np.ones(max_bound.shape) * -np.inf),
        vertcat(obj + min_bound, obj - max_bound),
        vertcat(np.ones(min_bound.shape) * np.inf, np.zeros(max_bound.shape)),
    )


def plot_com(x, nlp):
    q = nlp.mapping["q"].expand.map(x[:7, :])
    com_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoM, nlp.q)
    return np.array(com_func(q))[2]


def plot_com_dot(x, nlp):
    q = nlp.mapping["q"].expand.map(x[:7, :])
    q_dot = nlp.mapping["q"].expand.map(x[7:, :])
    com_dot_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoMdot, nlp.q, nlp.q_dot)
    return np.array(com_dot_func(q, q_dot))[2]


def plot_torque_bounds(x, min_or_max, nlp, minimal_tau=None):
    q = nlp.mapping["q"].expand.map(x[:7, :])
    q_dot = nlp.mapping["q"].expand.map(x[7:, :])
    func = biorbd.to_casadi_func("TorqueMax", nlp.model.torqueMax, nlp.q, nlp.q_dot)

    res = []
    for dof in [6, 7, 8, 9]:
        bound = []

        for i in range(len(x[0])):
            tmp = func(q[:, i], q_dot[:, i])
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
        # Plot CoM pos and speed
        ocp.add_plot("CoM", lambda x, u, p: plot_com(x, nlp), phase=i, plot_type=PlotType.PLOT)
        ocp.add_plot("CoM_dot", lambda x, u, p: plot_com_dot(x, nlp), phase=i, plot_type=PlotType.PLOT)
        # Plot q and nb_q_dot ranges
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
            "q_dot",
            lambda x, u, p: np.repeat(x_bounds[i].min[nq:, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "q_dot",
            lambda x, u, p: np.repeat(x_bounds[i].max[nq:, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
    return ocp


def warm_start_nmpc(sol, ocp):
    state, ctrl, param = Data.get_data(ocp, sol, concatenate=False, get_parameters=True)
    u_init = InitialGuessList()
    x_init = InitialGuessList()
    for i in range(ocp.nb_phases):
        u_init.add(np.concatenate([ctrl[d][i][:, :-1] for d in ctrl]), interpolation=InterpolationType.EACH_FRAME)
        x_init.add(np.concatenate([state[d][i] for d in state]), interpolation=InterpolationType.EACH_FRAME)

    time = InitialGuessOption(param["time"], name="time")
    ocp.update_initial_guess(x_init=x_init, u_init=u_init, param_init=time)

