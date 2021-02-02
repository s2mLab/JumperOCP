from time import time
import os.path
from scp import SCPClient
from paramiko import SSHClient

import numpy as np
import biorbd
from bioptim import (
    Node,
    OptimalControlProgram,
    ConstraintList,
    ConstraintFcn,
    ObjectiveFcn,
    ObjectiveList,
    DynamicsList,
    DynamicsFcn,
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


def prepare_ocp(model_path, phase_time, ns, time_min, time_max, init):
    # --- Options --- #
    # Model path
    biorbd_model = [biorbd.Model(elt) for elt in model_path]

    nb_phases = len(biorbd_model)

    q_mapping = BidirectionalMapping(
        Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
    )
    q_mapping = q_mapping, q_mapping, q_mapping, q_mapping, q_mapping
    tau_mapping = BidirectionalMapping(
        Mapping([-1, -1, -1, -1, 0, -1, 0, 1, 2, 3, 1, 2, 3], [5]), Mapping([4, 7, 8, 9])
    )
    tau_mapping = tau_mapping, tau_mapping, tau_mapping, tau_mapping, tau_mapping
    nq = len(q_mapping[0].reduce.map_idx)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-100, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)

    # --- Constraints --- #
    constraints = ConstraintList()

    # Positivity constraints of the normal component of the reaction forces
    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        constraints.add(ConstraintFcn.CONTACT_FORCE, phase=0, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)
        constraints.add(ConstraintFcn.CONTACT_FORCE, phase=4, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)
    contact_axes = (1, 3)
    for i in contact_axes:
        constraints.add(ConstraintFcn.CONTACT_FORCE, phase=1, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)
        constraints.add(ConstraintFcn.CONTACT_FORCE, phase=3, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)

    # Non-slipping constraints
    # N.B.: Application on only one of the two feet is sufficient, as the slippage cannot occurs on only one foot.
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        phase=0,
        node=Node.ALL,
        normal_component_idx=(1, 2),
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
    )
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        phase=1,
        node=Node.ALL,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
    )
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        phase=3,
        node=Node.ALL,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
    )
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        phase=4,
        node=Node.ALL,
        normal_component_idx=(1, 2),
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
    )

    # Custom constraints for contact forces at transitions
    constraints.add(utils.toe_on_floor, phase=3, node=Node.START, min_bound=-0.0001, max_bound=0.0001)
    constraints.add(utils.heel_on_floor, phase=4, node=Node.START, min_bound=-0.0001, max_bound=0.0001)

    # Custom constraints for positivity of CoM_dot on z axis just before the take-off
    constraints.add(utils.com_dot_z, phase=1, node=Node.END, min_bound=0, max_bound=np.inf)

    # Constraint arm positivity
    constraints.add(ConstraintFcn.TRACK_STATE, phase=1, node=Node.END, index=3, min_bound=1.0, max_bound=np.inf)

    # Constraint foot positivity
    constraints.add(utils.heel_on_floor, phase=1, node=Node.ALL, min_bound=-0.0001, max_bound=np.inf)
    constraints.add(utils.heel_on_floor, phase=2, node=Node.ALL, min_bound=-0.0001, max_bound=np.inf)
    constraints.add(utils.toe_on_floor, phase=2, node=Node.ALL, min_bound=-0.0001, max_bound=np.inf)
    constraints.add(utils.heel_on_floor, phase=3, node=Node.ALL, min_bound=-0.0001, max_bound=np.inf)

    # Torque constraint + minimize_state
    for i in range(nb_phases):
        constraints.add(utils.tau_actuator_constraints, phase=i, node=Node.ALL, minimal_tau=20)
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.0001, phase=i, min_bound=time_min[i], max_bound=time_max[i]
        )

    # State transitions
    state_transitions = StateTransitionList()
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=2)
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=3)

    # Path constraint
    nb_q = q_mapping[0].reduce.len
    nb_q_dot = nb_q
    pose_at_first_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]

    # Initialize x_bounds (Interpolation type is CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    x_bounds = BoundsList()
    for i in range(nb_phases):
        x_bounds.add(bounds=QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping[i]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * nb_q_dot

    x_bounds[3].min[13, 0] = -1000
    x_bounds[3].max[13, 0] = 1000

    x_bounds[4].min[13, 0] = -1000
    x_bounds[4].max[13, 0] = 1000

    x_bounds[4][2:, -1] = pose_at_first_node[2:] + [0] * nb_q_dot

    # Initial guess for states (Interpolation type is CONSTANT)
    x_init = InitialGuessList()
    for i in range(nb_phases):
        x_init.add(pose_at_first_node + [0] * nb_q_dot)

    # Define control path constraint
    u_bounds = BoundsList()
    for i in range(nb_phases):
        u_bounds.add([-500] * tau_mapping[i].reduce.len, [500] * tau_mapping[i].reduce.len)

    # Define initial guess for controls
    u_init = InitialGuessList()
    for i in range(nb_phases):
        if init is not None:
            u_init.add(init)
        else:
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
        nb_threads=2,
        use_SX=False,
    )
    return utils.add_custom_plots(ocp, nb_phases, x_bounds, nq, minimal_tau=20)

def main(args=None):
    init = None
    if args:
        init = args[:-1]
        pwd = args[-1]
        save_path = "5p_init_"+str(init[0])+"_"+str(init[1])+"_"+str(init[2])+"_"+str(init[3])+"_sol.bo"
        if os.path.exists(save_path):
            return

    model_path = (
        "../models/jumper2contacts.bioMod",
        "../models/jumper1contacts.bioMod",
        "../models/jumper1contacts.bioMod",
        "../models/jumper1contacts.bioMod",
        "../models/jumper2contacts.bioMod",
    )
    time_min = [0.2, 0.05, 0.05, 0.01, 0.01]
    time_max = [1, 1, 2, 0.2, 1]
    phase_time = [0.6, 0.2, 1, 0.2, 0.6]
    number_shooting_points = [30, 15, 20, 15, 30]

    tic = time()

    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=phase_time,
        ns=number_shooting_points,
        time_min=time_min,
        time_max=time_max,
        init=init,
    )

    sol = ocp.solve(
        show_online_optim=False,
        solver_options={"hessian_approximation": "limited-memory", "max_iter": 200}
    )

    utils.warm_start_nmpc(sol, ocp)
    ocp.solver.set_lagrange_multiplier(sol)

    sol = ocp.solve(
        show_online_optim=False,
        solver_options={"hessian_approximation": "exact",
                        "max_iter": 1000,
                        "warm_start_init_point": "yes",
                        }
    )

    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    if init:
        ocp.save(sol, save_path)
        ocp.save_get_data(sol, save_path + 'b')
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect('pariterre.net', username='aws', password=pwd)
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(save_path, save_path)
            scp.get(save_path)
            scp.put(save_path + 'b', save_path + 'b')
            scp.get(save_path + 'b')


if __name__ == "__main__":
    main()
