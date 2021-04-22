import numpy as np
import biorbd
from bioptim import (
    InitialGuess,
    InterpolationType,
    Node,
    OptimalControlProgram,
    ConstraintFcn,
    ObjectiveFcn,
    DynamicsFcn,
    QAndQDotBounds,
    PhaseTransitionFcn,
    ConstraintList,
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InitialGuessList,
    PhaseTransitionList,
)

from .jumper import Jumper
from .ocp import maximal_tau, com_dot_z, marker_on_floor
from .viz import add_custom_plots
from .find_initial_pose import find_initial_root_pose


class JumperOcp:
    jumper = Jumper()

    models = []
    n_q, n_qdot, n_tau = -1, -1, -1
    q_mapping, qdot_mapping, tau_mapping = None, None, None

    dynamics = DynamicsList()
    constraints = ConstraintList()
    objective_functions = ObjectiveList()
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    phase_transitions = PhaseTransitionList()
    initial_states = []
    x_init = InitialGuessList()
    u_init = InitialGuessList()

    def __init__(self, path_to_models, n_phases, n_thread=8):
        if n_phases < 1 or n_phases > 5:
            raise ValueError("n_phases must be comprised between 1 and 5")
        self.n_phases = n_phases
        self.takeoff = 0 if self.n_phases == 1 else 1  # The index of takeoff phase

        self._load_models(path_to_models)
        self._set_dimensions_and_mapping()
        self._set_initial_states()

        self._set_dynamics()
        self._set_constraints()
        self._set_objective_functions()

        self._set_boundary_conditions()
        self._set_phase_transitions()

        self._set_initial_guesses()

        self.ocp = OptimalControlProgram(
            self.models[:self.n_phases],
            self.dynamics,
            self.jumper.n_shoot[:self.n_phases],
            self.jumper.phase_time[:self.n_phases],
            x_init=self.x_init,
            x_bounds=self.x_bounds,
            u_init=self.u_init,
            u_bounds=self.u_bounds,
            objective_functions=self.objective_functions,
            constraints=self.constraints,
            q_mapping=self.q_mapping,
            qdot_mapping=self.q_mapping,
            tau_mapping=self.tau_mapping,
            phase_transitions=self.phase_transitions,
            n_threads=n_thread,
        )

    def _load_models(self, path_to_models):
        self.models = [biorbd.Model(path_to_models + elt) for elt in self.jumper.model_files][:self.n_phases]

    def _set_initial_states(self):
        initial_pose = np.array([self.jumper.body_at_first_node]).T
        initial_velocity = np.array([self.jumper.initial_velocity]).T

        pos_body_no_root = self.q_mapping[0].to_second.map(initial_pose)[self.models[0].nbRoot():, :]
        initial_pose[:self.models[0].nbRoot(), 0] = find_initial_root_pose(self.models[0], pos_body_no_root[:, 0])

        self.initial_states = np.concatenate((initial_pose, initial_velocity))

    def _set_dimensions_and_mapping(self):
        self.q_mapping = [self.jumper.q_mapping for _ in range(self.n_phases)]
        self.qdot_mapping = [self.jumper.q_mapping for _ in range(self.n_phases)]
        self.tau_mapping = [self.jumper.tau_mapping for _ in range(self.n_phases)]
        self.n_q = self.jumper.q_mapping.to_first.len
        self.n_qdot = self.n_q
        self.n_tau = self.jumper.tau_mapping.to_first.len

    def _set_dynamics(self):
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Flat foot
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Toe only
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN)  # Aerial phase
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Toe only
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Flat foot

    def _set_constraints(self):
        # Torque constrained to torqueMax
        for i in range(self.n_phases):
            self.constraints.add(maximal_tau, phase=i, node=Node.ALL, minimal_tau=self.jumper.tau_min)

        # Positivity of CoM_dot on z axis prior the take-off
        self.constraints.add(com_dot_z, phase=self.takeoff, node=Node.END, min_bound=0, max_bound=np.inf)

        # Constraint arm positivity (prevent from local minimum with arms in the back)
        self.constraints.add(
            ConstraintFcn.TRACK_STATE, phase=self.takeoff, node=Node.END, index=3, min_bound=0, max_bound=np.inf
        )

        # Floor constraints for flat foot phases
        for p in self.jumper.flat_foot_phases:
            if p >= self.n_phases:
                break

            # Do not pull on floor
            for i in self.jumper.flatfoot_contact_z_idx:
                self.constraints.add(
                    ConstraintFcn.CONTACT_FORCE, phase=p, node=Node.ALL, contact_force_idx=i, max_bound=np.inf
                )

            # Non-slipping constraints
            self.constraints.add(  # On only one of the feet
                ConstraintFcn.NON_SLIPPING,
                phase=p,
                node=Node.ALL,
                tangential_component_idx=self.jumper.flatfoot_non_slipping[0],
                normal_component_idx=self.jumper.flatfoot_non_slipping[1],
                static_friction_coefficient=self.jumper.static_friction_coefficient,
            )

        # Floor constraints for toe only phases
        for p in self.jumper.toe_only_phases:
            if p >= self.n_phases:
                break

            # Do not pull on floor
            for i in self.jumper.toe_contact_z_idx:
                self.constraints.add(
                    ConstraintFcn.CONTACT_FORCE, phase=p, node=Node.ALL, contact_force_idx=i, max_bound=np.inf
                )

            # The heel must remain over floor
            self.constraints.add(
                marker_on_floor,
                phase=p,
                node=Node.ALL,
                min_bound=-0.0001,
                max_bound=np.inf,
                marker=self.jumper.heel_marker_idx,
                floor_z=self.jumper.floor_z,
            )

            # Non-slipping constraints
            self.constraints.add(  # On only one of the feet
                ConstraintFcn.NON_SLIPPING,
                phase=p,
                node=Node.ALL,
                tangential_component_idx=self.jumper.toe_non_slipping[0],
                normal_component_idx=self.jumper.toe_non_slipping[1],
                static_friction_coefficient=self.jumper.static_friction_coefficient,
            )

    def _set_objective_functions(self):
        # Maximize the jump height
        self.objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-100, phase=self.takeoff)

        # Minimize the tau on root if present
        for p in range(self.n_phases):
            root = [i for i in self.jumper.tau_mapping.to_second.map_idx[:self.models[p].nbRoot()] if i is not None]
            if root:
                self.objective_functions.add(
                    ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
                    weight=0.1,
                    phase=p,
                    index=root,
                )

        # Minimize unnecessary movement during for the aerial and reception phases
        for p in range(2, 5):
            if p >= self.n_phases:
                break
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE_DERIVATIVE,
                weight=0.1,
                phase=p,
                index=range(self.n_q, self.n_q + self.n_qdot),
            )

        # Minimize time of the phase
        for i in range(self.n_phases):
            if self.jumper.time_min[i] != self.jumper.time_max[i]:
                self.objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_TIME,
                    weight=0.1,
                    phase=i,
                    min_bound=self.jumper.time_min[i],
                    max_bound=self.jumper.time_max[i],
                )

    def _set_boundary_conditions(self):
        for i in range(self.n_phases):
            # Path constraints
            self.x_bounds.add(
                bounds=QAndQDotBounds(self.models[i], q_mapping=self.q_mapping[i], qdot_mapping=self.qdot_mapping[i])
            )
            self.u_bounds.add([-500] * self.n_tau, [500] * self.n_tau)

        # Enforce the initial pose and velocity
        self.x_bounds[0][:, 0] = self.initial_states[:, 0]

        # Target the final pose (except for translation) and velocity
        if self.n_phases >= 4:
            self.objective_functions.add(
                ObjectiveFcn.Mayer.TRACK_STATE,
                phase=self.n_phases - 1,
                index=range(2, self.n_q + self.n_qdot),
                target=self.initial_states[2:, :],
                weight=1,
            )

    def _set_initial_guesses(self):
        for i in range(self.n_phases):
            self.x_init.add(self.initial_states)
            self.u_init.add([0] * self.n_tau)

    def _set_phase_transitions(self):
        if self.n_phases >= 2:  # 2 contacts to 1 contact
            self.phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)
        if self.n_phases >= 3:  # 1 contact to aerial
            self.phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)
        if self.n_phases >= 4:  # aerial to 1 contact
            self.phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=2)
        if self.n_phases >= 5:  # 1 contact to 2 contacts
            self.phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)

        if self.n_phases >= 3:  # The end of the aerial
            self.constraints.add(
                marker_on_floor,
                phase=2,
                node=Node.END,
                min_bound=-0.001,
                max_bound=0.001,
                marker=self.jumper.toe_marker_idx,
                floor_z=self.jumper.floor_z,
            )
        if self.n_phases >= 4:  # 2 contacts on floor
            self.constraints.add(
                marker_on_floor,
                phase=3,
                node=Node.END,
                min_bound=-0.001,
                max_bound=0.001,
                marker=self.jumper.heel_marker_idx,
                floor_z=self.jumper.floor_z,
            )

        # Allow for passive velocity at reception
        if self.n_phases >= 4:
            self.x_bounds[3].min[self.n_q :, 0] = 2 * self.x_bounds[3].min[self.n_q :, 0]
            self.x_bounds[3].max[self.n_q :, 0] = 2 * self.x_bounds[3].max[self.n_q :, 0]
        if self.n_phases >= 5:
            self.x_bounds[4].min[self.n_q :, 0] = 2 * self.x_bounds[4].min[self.n_q :, 0]
            self.x_bounds[4].max[self.n_q :, 0] = 2 * self.x_bounds[4].max[self.n_q :, 0]

    def solve(self, limit_memory_max_iter, exact_max_iter, load_path=None, force_no_graph=False):
        def warm_start(ocp, sol):
            state, ctrl, param = sol.states, sol.controls, sol.parameters
            u_init_guess = InitialGuessList()
            x_init_guess = InitialGuessList()
            for i in range(ocp.n_phases):
                if ocp.n_phases == 1:
                    u_init_guess.add(ctrl["all"][:, :-1], interpolation=InterpolationType.EACH_FRAME)
                    x_init_guess.add(state["all"], interpolation=InterpolationType.EACH_FRAME)
                else:
                    u_init_guess.add(ctrl[i]["all"][:, :-1], interpolation=InterpolationType.EACH_FRAME)
                    x_init_guess.add(state[i]["all"], interpolation=InterpolationType.EACH_FRAME)

            time_init_guess = InitialGuess(param["time"], name="time")
            ocp.update_initial_guess(x_init=x_init_guess, u_init=u_init_guess, param_init=time_init_guess)
            ocp.solver.set_lagrange_multiplier(sol)

        # Run optimizations
        if not force_no_graph:
            add_custom_plots(self.ocp, self.jumper.tau_min)

        if load_path:
            _, sol = OptimalControlProgram.load(load_path)
            return sol
        else:
            sol = None
            if limit_memory_max_iter > 0:
                sol = self.ocp.solve(
                    show_online_optim=exact_max_iter == 0 and not force_no_graph,
                    solver_options={"hessian_approximation": "limited-memory", "max_iter": limit_memory_max_iter},
                )
            if limit_memory_max_iter > 0 and exact_max_iter > 0:
                warm_start(self.ocp, sol)
            if exact_max_iter > 0:
                sol = self.ocp.solve(
                    show_online_optim=True and not force_no_graph,
                    solver_options={
                        "hessian_approximation": "exact",
                        "max_iter": exact_max_iter,
                        "warm_start_init_point": "yes",
                    },
                )

            return sol
