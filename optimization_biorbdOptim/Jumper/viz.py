import numpy as np
import biorbd

from bioptim import PlotType

from .jumper import Jumper


def add_custom_plots(ocp, minimal_tau=None):
    jumper = Jumper()
    nq = jumper.q_mapping.to_first.len

    def plot_torque_bounds(x, u, p, min_or_max, nlp, minimal_tau=None):
        # min = 1, max = 0
        q = nlp.mapping["q"].to_second.map(x[:nq, :])
        qdot = nlp.mapping["q"].to_second.map(x[nq:, :])
        func = biorbd.to_casadi_func("TorqueMax", nlp.model.torqueMax, nlp.q, nlp.qdot)

        res = []
        for dof in nlp.mapping["tau"].to_first.map_idx:
            bound = []

            for i in range(q.shape[1]):
                tmp = func(q[:, i], qdot[:, i])
                if minimal_tau and tmp[dof, min_or_max] < minimal_tau:
                    bound.append(minimal_tau)
                else:
                    bound.append(float(tmp[dof, min_or_max]))
            res.append(np.array(bound))

        return -np.array(res) if min_or_max else np.array(res)

    def plot_com(x, u, p, nlp):
        q = nlp.mapping["q"].to_second.map(x[:nq, :])
        qdot = nlp.mapping["q"].to_second.map(x[nq:, :])

        com_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoM, nlp.q)
        com_dot_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoMdot, nlp.q, nlp.qdot)

        return np.concatenate((com_func(q)[2, :], com_dot_func(q, qdot)[2, :]))

    def plot_sum_contact_forces(x, u, p, nlp, is_flatfoot):
        if nlp.contact_forces_func:
            forces = nlp.contact_forces_func(x, u, p)
            x_idx = jumper.flatfoot_contact_x_idx if is_flatfoot else jumper.toe_contact_x_idx
            y_idx = jumper.flatfoot_contact_y_idx if is_flatfoot else jumper.toe_contact_y_idx
            z_idx = jumper.flatfoot_contact_z_idx if is_flatfoot else jumper.toe_contact_z_idx
            x = np.sum(forces[x_idx, :], axis=0)
            y = np.sum(forces[y_idx, :], axis=0)
            z = np.sum(forces[z_idx, :], axis=0)
            return np.array((x, y, z))
        else:
            return np.zeros((3, 1))

    for i, nlp in enumerate(ocp.nlp):
        nq = nlp.var_states["q"]
        x_bounds = nlp.x_bounds

        # Plot q and nb_qdot ranges
        ocp.add_plot(
            "q",
            lambda x, u, p: np.repeat(x_bounds.min[:nq, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "q",
            lambda x, u, p: np.repeat(x_bounds.max[:nq, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "qdot",
            lambda x, u, p: np.repeat(x_bounds.min[nq:, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "qdot",
            lambda x, u, p: np.repeat(x_bounds.max[nq:, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )

        # Plot Torque Bounds
        line_style = "-." if minimal_tau else "-"
        ocp.add_plot(
            "tau",
            plot_torque_bounds, min_or_max=0, nlp=nlp,
            phase=i,
            plot_type=PlotType.STEP,
            color="g",
            linestyle=line_style,
        )
        ocp.add_plot(
            "tau",
            plot_torque_bounds, min_or_max=1, nlp=nlp,
            phase=i,
            plot_type=PlotType.STEP,
            color="g",
            linestyle=line_style,
        )
        if minimal_tau:
            ocp.add_plot(
                "tau",
                plot_torque_bounds, min_or_max=0, nlp=nlp, minimal_tau=minimal_tau,
                phase=i,
                plot_type=PlotType.STEP,
                color="g",
            )
            ocp.add_plot(
                "tau",
                plot_torque_bounds, min_or_max=1, nlp=nlp, minimal_tau=minimal_tau,
                phase=i,
                plot_type=PlotType.STEP,
                color="g",
            )

        ocp.add_plot("tau", lambda x, u, p: np.zeros(u.shape), phase=i, plot_type=PlotType.STEP, color=[0.1, 0.1, 0.1])

        # Plot CoM pos and speed
        ocp.add_plot("CoM", plot_com, nlp=nlp, phase=i, legend=["CoM", "CoM_dot"])

        # Plot contact sum
        ocp.add_plot("Sum of contacts", plot_sum_contact_forces, nlp=nlp, is_flatfoot=i in jumper.flat_foot_phases, phase=i, legend=["X", "Y", "Z"])

