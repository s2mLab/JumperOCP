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
    objective_functions.add(Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1, phase=1)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)
    dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # --- Constraints --- #
    constraints = ConstraintList()

    # Positivity constraints of the normal component of the reaction forces
    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        constraints.add(Constraint.CONTACT_FORCE, phase=0, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)
    contact_axes = (1, 3)
    for i in contact_axes:
        constraints.add(Constraint.CONTACT_FORCE, phase=1, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)

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
        static_friction_coefficient=0.5,
    )

    # Custom constraints for positivity of CoM_dot on z axis just before the take-off
    constraints.add(utils.com_dot_z, phase=1, node=Node.END, min_bound=0, max_bound=np.inf)

    # Constraint arm positivity
    constraints.add(Constraint.TRACK_STATE, phase=1, node=Node.END, index=3, min_bound=1.0, max_bound=np.inf)

    # Torque constraint + minimize_state
    for i in range(nb_phases):
        constraints.add(utils.tau_actuator_constraints, phase=i, node=Node.ALL)
        objective_functions.add(
            Objective.Mayer.MINIMIZE_TIME, weight=0.0001, phase=i, min_bound=time_min[i], max_bound=time_max[i]
        )
        # objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=0.0001, phase=i)

    # --- Path constraints --- #
    nb_q = q_mapping[0].reduce.len
    nb_q_dot = nb_q
    pose_at_first_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]

    # Initialize x_bounds (Interpolation type is CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    x_bounds = BoundsList()
    for i in range(nb_phases):
        x_bounds.add(QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping[i]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * nb_q_dot

    # Initial guess for states (Interpolation type is CONSTANT)
    x_init = InitialGuessList()
    for i in range(nb_phases):
        x_init.add(pose_at_first_node + [0] * nb_q_dot)

    # Define control path constraint
    u_bounds = BoundsList()
    for tau_m in tau_mapping:
        u_bounds.add([[-500] * tau_m.reduce.len, [500] * tau_m.reduce.len])

    # Define initial guess for controls
    u_init = InitialGuessList()
    for tau_m in tau_mapping:
        u_init.add([0] * tau_m.reduce.len)  # Interpolation type is CONSTANT (default value)

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
        nb_threads=4,
        use_SX=False,
    )

    ocp = utils.add_custom_plots(ocp, nb_phases, x_bounds, nq)

    return ocp


if __name__ == "__main__":
    model_path = (
        "../models/jumper2contacts.bioMod",
        "../models/jumper1contacts.bioMod",
        "../models/jumper1contacts.bioMod",
    )
    time_min = [0.2, 0.05, 0.6]
    time_max = [1, 1, 0.6]
    phase_time = [0.6, 0.2, 1]
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
        show_online_optim=True, solver_options={"hessian_approximation": "exact", "max_iter": 1000}
    )

    # --- Show results --- #
    # param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    # print(
    #     f"The optimized phases times are: {param['time'][0, 0]}s and {param['time'][1, 0]}s."
    # )

    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    result = ShowResult(ocp, sol)
    result.animate(nb_frames=121)
