from time import time

import numpy as np
import biorbd
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
    StateTransition,
    StateTransitionList,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ShowResult,
)

import utils


def prepare_ocp(model_path, phase_time, ns, time_min, time_max):
    # --- Options --- #
    # Model path
    biorbd_model = [biorbd.Model(elt) for elt in model_path]

    nb_phases = len(biorbd_model)

    q_mapping = BidirectionalMapping(
        Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
    )
    q_mapping = q_mapping, q_mapping, q_mapping
    tau_mapping = BidirectionalMapping(
        Mapping([-1, -1, -1, -1, 0, -1, 0, 1, 2, 3, 1, 2, 3], [5]), Mapping([4, 7, 8, 9])
    )
    tau_mapping = tau_mapping, tau_mapping, tau_mapping
    nq = len(q_mapping[0].reduce.map_idx)

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)
    dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)
    dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)

    # --- Constraints --- #
    constraints = ConstraintList()

    # Positivity constraints of the normal component of the reaction forces
    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        constraints.add(Constraint.CONTACT_FORCE, phase=2, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)
    contact_axes = (1, 3)
    for i in contact_axes:
        constraints.add(Constraint.CONTACT_FORCE, phase=1, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)

    # Non-slipping constraints
    # N.B.: Application on only one of the two feet is sufficient, as the slippage cannot occurs on only one foot.
    constraints.add(
        Constraint.NON_SLIPPING,
        phase=1,
        node=Node.ALL,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
    )
    constraints.add(
        Constraint.NON_SLIPPING,
        phase=2,
        node=Node.ALL,
        normal_component_idx=(1, 2),
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
    )

    # Custom constraints for contact forces at transitions
    constraints.add(utils.toe_on_floor, phase=1, node=Node.START)
    constraints.add(utils.heel_on_floor, phase=2, node=Node.START)

    # Torque constraint + minimize_state
    for i in range(nb_phases):
        constraints.add(utils.tau_actuator_constraints, phase=i, node=Node.ALL, minimal_tau=20)
        objective_functions.add(
            Objective.Mayer.MINIMIZE_TIME, weight=0.0001, phase=i, min_bound=time_min[i], max_bound=time_max[i]
        )
        # objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=0.0001, phase=i)

    # State transitions
    state_transitions = StateTransitionList()
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=0)
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=1)

    # Path constraint
    nb_q = q_mapping[0].reduce.len
    nb_q_dot = nb_q
    first_node_state = [0, 0, -0.8, 2.21, 1.05, -1, -0.15, 1.7, 5, 7.5, 0, -20, 20, -17.5]
    end_pose = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]

    # Initialize x_bounds (Interpolation type is CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    x_bounds = BoundsList()
    for i in range(nb_phases):
        x_bounds.add(QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping[i]))
    x_bounds[0][:, 0] = first_node_state

    x_bounds[1].min[13, 0] = -1000
    x_bounds[1].max[13, 0] = 1000

    x_bounds[2].min[13, 0] = -1000
    x_bounds[2].max[13, 0] = 1000

    x_bounds[2][2:, -1] = end_pose[2:] + [0] * nb_q_dot

    # Initial guess for states (Interpolation type is CONSTANT)
    x_init = InitialGuessList()
    for i in range(nb_phases):
        x_init.add(end_pose + [0] * nb_q_dot)

    # Define control path constraint
    u_bounds = BoundsList()
    for i in range(nb_phases):
        u_bounds.add([[-500] * tau_mapping[i].reduce.len, [500] * tau_mapping[i].reduce.len])

    # Define initial guess for controls
    u_init = InitialGuessList()
    for i in range(nb_phases):
        u_init.add([0] * tau_mapping[i].reduce.len)

    # ------------- #

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
        q_dot_mapping=q_mapping,
        tau_mapping=tau_mapping,
        state_transitions=state_transitions,
        nb_threads=1,
        use_SX=False,
    )
    return utils.add_custom_plots(ocp, nb_phases, x_bounds, nq, minimal_tau=20)


if __name__ == "__main__":
    model_path = (
        "../models/jumper1contacts.bioMod",
        "../models/jumper1contacts.bioMod",
        "../models/jumper2contacts.bioMod",
    )
    time_min = [0.05, 0.01, 0.05]
    time_max = [2, 0.3, 1]
    phase_time = [1, 0.2, 0.6]
    number_shooting_points = [30, 30, 30]

    tic = time()

    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=phase_time,
        ns=number_shooting_points,
        time_min=time_min,
        time_max=time_max,
    )

    sol = ocp.solve(
        show_online_optim=True,
        solver_options={"hessian_approximation": "limited-memory", "max_iter": 500}
    )
    quit()

    ocp = utils.warm_start_nmpc(sol, ocp)
    ocp.solver.set_lagrange_multiplier(sol)

    sol = ocp.solve(
        show_online_optim=True,
        solver_options={"hessian_approximation": "exact",
                        "max_iter": 1000,
                        "warm_start_init_point": "yes",
                        }
    )

    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    result = ShowResult(ocp, sol)
    result.animate(nb_frames=241)
