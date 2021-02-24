import numpy as np
import biorbd
import math

from bioptim import PlotType


def add_custom_plots(ocp, minimal_tau=None):
    for i in range(len(ocp.nlp)):
        nlp = ocp.nlp[i]
        nq = nlp.var_states["q"]
        x_bounds = nlp.x_bounds

        # Plot Torque Bounds
        if not minimal_tau:
            ocp.add_plot(
                "tau", lambda x, u, p: plot_torque_bounds(x, 0, nlp), phase=i, plot_type=PlotType.STEP, color="g"
            )
            ocp.add_plot(
                "tau", lambda x, u, p: -plot_torque_bounds(x, 1, nlp), phase=i, plot_type=PlotType.STEP, color="g"
            )
        else:
            ocp.add_plot(
                "tau",
                lambda x, u, p: plot_torque_bounds(x, 0, nlp),
                phase=i,
                plot_type=PlotType.STEP,
                color="g",
                linestyle="-.",
            )
            ocp.add_plot(
                "tau",
                lambda x, u, p: -plot_torque_bounds(x, 1, nlp),
                phase=i,
                plot_type=PlotType.STEP,
                color="g",
                linestyle="-.",
            )
            ocp.add_plot(
                "tau",
                lambda x, u, p: plot_torque_bounds(x, 0, nlp, minimal_tau=minimal_tau),
                phase=i,
                plot_type=PlotType.STEP,
                color="g",
            )
            ocp.add_plot(
                "tau",
                lambda x, u, p: -plot_torque_bounds(x, 1, nlp, minimal_tau=minimal_tau),
                phase=i,
                plot_type=PlotType.STEP,
                color="g",
            )
        ocp.add_plot(
            "q_degree",
            lambda x, u, p: x[:7, :] * 180 / math.pi,
            phase=i,
            plot_type=PlotType.INTEGRATED,
            legend=(
                "q_Pelvis_Trans_Y",
                "q_Pelvis_Trans_Z",
                "q_Pelvis_Rot_X",
                "q_BrasD_Rot_X",
                "q_CuisseD_Rot_X",
                "q_JambeD_Rot_X",
                "q_Pied_Rot_X",
            ),
        )
        ocp.add_plot(
            "qdot_degree",
            lambda x, u, p: x[7:, :] * 180 / math.pi,
            phase=i,
            plot_type=PlotType.INTEGRATED,
            legend=(
                "qdot_Pelvis_Trans_Y",
                "qdot_Pelvis_Trans_Z",
                "qdot_Pelvis_Rot_X",
                "qdot_BrasD_Rot_X",
                "qdot_CuisseD_Rot_X",
                "qdot_JambeD_Rot_X",
                "qdot_Pied_Rot_X",
            ),
        )
        ocp.add_plot("tau", lambda x, u, p: np.zeros((4, len(x[0]))), phase=i, plot_type=PlotType.STEP, color="b")
        # Plot CoM pos and speed
        ocp.add_plot("CoM", lambda x, u, p: plot_com(x, nlp), phase=i, plot_type=PlotType.PLOT)
        ocp.add_plot("CoM_dot", lambda x, u, p: plot_com_dot(x, nlp), phase=i, plot_type=PlotType.PLOT)
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
        ocp.add_plot(
            "q_degree",
            lambda x, u, p: np.repeat(x_bounds.min[:nq, 1][:, np.newaxis], x.shape[1], axis=1) * 180 / math.pi,
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "q_degree",
            lambda x, u, p: np.repeat(x_bounds.max[:nq, 1][:, np.newaxis], x.shape[1], axis=1) * 180 / math.pi,
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "qdot_degree",
            lambda x, u, p: np.repeat(x_bounds.min[nq:, 1][:, np.newaxis], x.shape[1], axis=1) * 180 / math.pi,
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "qdot_degree",
            lambda x, u, p: np.repeat(x_bounds.max[nq:, 1][:, np.newaxis], x.shape[1], axis=1) * 180 / math.pi,
            phase=i,
            plot_type=PlotType.PLOT,
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
