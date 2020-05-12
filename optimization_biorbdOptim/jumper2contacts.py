from matplotlib import pyplot as plt
import numpy as np
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
    Dynamics,
    Data,
    ShowResult,
)


def custom_func_2c_to_1c_transition(ocp, nlp, t, x, u,):
    val = ocp.nlp[0]["contact_forces_func"](x[0], u[0])[[2, 5], 0]
    return val

def custom_func_1c_to_0c_transition(ocp, nlp, t, x, u,):
    val = nlp["contact_forces_func"](x[-1], u[-1])[:, 0]
    return val

def custom_func_anatomical_constraint(ocp, nlp, t, x, u,):
    val = x[0][7:14]
    return val

def prepare_ocp(
    model_path, phase_time, number_shooting_points, show_online_optim=False, use_symmetry=True,
):
    # --- Options --- #
    # Model path
    biorbd_model = [biorbd.Model(elt) for elt in model_path]

    nb_phases = len(biorbd_model)
    torque_min, torque_max, torque_init = -1000, 1000, 0

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
            Mapping([i for i in range(biorbd_model[0].nbQ())]), Mapping([i for i in range(biorbd_model[0].nbQ())]),
        )
        q_mapping = q_mapping, q_mapping
        tau_mapping = q_mapping

    # Add objective functions
    objective_functions = (
        (),
        (
            {"type": Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, "weight": -1},
            {"type": Objective.Lagrange.MINIMIZE_ALL_CONTROLS, "weight": 1 / 100},
        ),
    )

    # Dynamics
    problem_type = (
        ProblemType.torque_driven_with_contact,
        ProblemType.torque_driven_with_contact,
    )

    constraints_first_phase = []
    constraints_second_phase = []

    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        constraints_first_phase.append(
            {
                "type": Constraint.CONTACT_FORCE_INEQUALITY,
                "direction": "GREATER_THAN",
                "instant": Instant.ALL,
                "contact_force_idx": i,
                "boundary": 0,
            }
        )
    contact_axes = (1, 3)
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
    constraints_first_phase.append(
        {
            "type": Constraint.NON_SLIPPING,
            "instant": Instant.ALL,
            "normal_component_idx": (1, 2),
            "tangential_component_idx": 0,
            "static_friction_coefficient": 0.5,
        }
    )
    constraints_second_phase.append(
        {
            "type": Constraint.NON_SLIPPING,
            "instant": Instant.ALL,
            "normal_component_idx": 1,
            "tangential_component_idx": 0,
            "static_friction_coefficient": 0.5,
        }
    )
    constraints_second_phase.append(
        {
            "type": Constraint.CUSTOM,
            "function": custom_func_2c_to_1c_transition,
            "instant": Instant.START,
        }
    )
    # constraints_second_phase.append(
    #     {
    #         "type": Constraint.CUSTOM,
    #         "function": custom_func_anatomical_constraint,
    #         "instant": Instant.END,
    #     }
    # )
    if not use_symmetry:
        first_dof = (3, 4, 7, 8, 9)
        second_dof = (5, 6, 10, 11, 12)
        coeff = (-1, 1, 1, 1, 1)
        for i in range(len(first_dof)):
            constraints_first_phase.append(
                {
                    "type": Constraint.PROPORTIONAL_STATE,
                    "instant": Instant.ALL,
                    "first_dof": first_dof[i],
                    "second_dof": second_dof[i],
                    "coef": coeff[i],
                }
            )

        for i in range(len(first_dof)):
            constraints_second_phase.append(
                {
                    "type": Constraint.PROPORTIONAL_STATE,
                    "instant": Instant.ALL,
                    "first_dof": first_dof[i],
                    "second_dof": second_dof[i],
                    "coef": coeff[i],
                }
            )
    constraints = (constraints_first_phase, constraints_second_phase)

    # Path constraint
    if use_symmetry:
        nb_q = q_mapping[0].reduce.len
        nb_qdot = nb_q
        pose_at_first_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]
    else:
        nb_q = q_mapping[0].reduce.len
        nb_qdot = nb_q
        pose_at_first_node = [
            0,
            0,
            -0.5336,
            0,
            1.4,
            0,
            1.4,
            0.8,
            -0.9,
            0.47,
            0.8,
            -0.9,
            0.47,
        ]

    # Initialize X_bounds
    X_bounds = [QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping[i]) for i in range(nb_phases)]
    X_bounds[0].first_node_min = pose_at_first_node + [0] * nb_qdot
    X_bounds[0].first_node_max = pose_at_first_node + [0] * nb_qdot

    # Initial guess
    X_init = [
        InitialConditions(pose_at_first_node + [0] * nb_qdot),
        InitialConditions(pose_at_first_node + [0] * nb_qdot),
    ]

    # Define control path constraint
    U_bounds = [
        Bounds(min_bound=[torque_min] * tau_m.reduce.len, max_bound=[torque_max] * tau_m.reduce.len)
        for tau_m in tau_mapping
    ]

    U_init = [InitialConditions([torque_init] * tau_m.reduce.len) for tau_m in tau_mapping]
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        phase_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        q_mapping=q_mapping,
        q_dot_mapping=q_mapping,
        tau_mapping=tau_mapping,
        show_online_optim=show_online_optim,
    )


def run_and_save_ocp(model_path, phase_time, number_shooting_points):
    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=phase_time,
        number_shooting_points=number_shooting_points,
        show_online_optim=False,
        use_symmetry=True,
    )
    sol = ocp.solve()
    OptimalControlProgram.save(ocp, sol, "../Results/jumper2contacts_sol")


if __name__ == "__main__":
    model_path = ("../models/jumper2contacts.bioMod", "../models/jumper1contacts.bioMod")
    phase_time = [0.4, 0.2]
    number_shooting_points = [8, 8]
    run_and_save_ocp(model_path, phase_time=phase_time, number_shooting_points=number_shooting_points)
    ocp, sol = OptimalControlProgram.load(biorbd_model_path=model_path, name="../Results/jumper2contacts_sol.bo")

    contact_forces = np.zeros((6, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    cs_map = (range(6), (0, 1, 3, 4))

    for i, nlp in enumerate(ocp.nlp):
        states, controls = Data.get_data(ocp, sol["x"], phase_idx=i)
        q, q_dot, u = states["q"], states["q_dot"], controls["tau"]
        x = np.concatenate((q, q_dot))
        if i == 0:
            contact_forces[cs_map[i], : nlp["ns"] + 1] = nlp["contact_forces_func"](x, u)
        else:
            contact_forces[cs_map[i], ocp.nlp[i - 1]["ns"] : ocp.nlp[i - 1]["ns"] + nlp["ns"] + 1] = nlp[
                "contact_forces_func"
            ](x, u)

    names_contact_forces = ocp.nlp[0]["model"].contactNames()
    for i, elt in enumerate(contact_forces):
        plt.plot(elt.T, ".-", label=f"{names_contact_forces[i].to_string()}")
    plt.legend()
    plt.grid()
    plt.title("Contact forces")
    plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate(nb_frames=150)
