import numpy as np
import biorbd
from bioptim import (
    InitialGuess,
    InterpolationType,
    Node,
    OptimalControlProgram,
    ConstraintList,
    ConstraintFcn,
    ObjectiveFcn,
    ObjectiveList,
    DynamicsList,
    DynamicsFcn,
    BiMapping,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    PhaseTransitionList,
    PhaseTransitionFcn,
)

from .ocp import maximal_tau, com_dot_z, marker_on_floor
from .viz import add_custom_plots


class Jumper:
    def __init__(self, model_paths, n_shoot, time_min, phase_time, time_max, initial_pose, n_thread=1):
        self.models = []
        self._load_models(model_paths)

        # Element for the optimization
        self.n_phases = len(self.models)
        self.n_shoot = n_shoot
        self.time_min = time_min
        self.phase_time = phase_time
        self.time_max = time_max
        self.takeoff = 0 if self.n_phases == 1 else 1  # The index of takeoff phase
        self.flat_foot_phases = ((0, 4) if self.n_phases >= 5 else 0,)  # The indices of flat foot phases
        self.toe_only_phases = ((1, 3) if self.n_phases >= 4 else 1,)  # The indices of toe only phases

        # Elements from the model
        self.initial_states = []
        self._set_initial_states(initial_pose, [0, 0, 0, 0, 0, 0, 0])
        self.heel_and_toe_contact_idx = (1, 2, 4, 5)  # Contacts indices of heel and toe in bioMod 2 contacts
        self.toe_contact_idx = (1, 3)  # Contacts indices of toe in bioMod 1 contact
        self.toe_marker = 3
        self.heel_marker = 3
        self.floor = 0.779  # floor = -0.77865829
        self.tau_min = 20

        self.n_q, self.n_qdot, self.n_tau = -1, -1, -1
        self.q_mapping, self.qdot_mapping, self.tau_mapping = None, None, None
        self._set_dimensions_and_mapping()

        # Prepare the optimal control program
        self.dynamics = DynamicsList()
        self._set_dynamics()

        self.constraints = ConstraintList()
        self._set_constraints()

        self.objective_functions = ObjectiveList()
        self._set_objective_functions()

        self.x_bounds = BoundsList()
        self.u_bounds = BoundsList()
        self._set_boundary_conditions()

        self.phase_transitions = PhaseTransitionList()
        self._set_phase_transitions()

        self.x_init = InitialGuessList()
        self.u_init = InitialGuessList()
        self._set_initial_guesses()

        self.ocp = OptimalControlProgram(
            self.models,
            self.dynamics,
            self.n_shoot,
            self.phase_time,
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

    def _load_models(self, model_paths):
        self.models = [biorbd.Model(elt) for elt in model_paths]

    def _set_initial_states(self, initial_pose, initial_velocity):
        self.initial_states = np.array([list(initial_pose) + initial_velocity]).T

    def _set_dimensions_and_mapping(self):
        q_mapping = BiMapping([0, 1, 2, None, 3, None, 3, 4, 5, 6, 4, 5, 6], [0, 1, 2, 4, 7, 8, 9])
        self.q_mapping = [q_mapping for _ in range(self.n_phases)]
        self.qdot_mapping = [q_mapping for _ in range(self.n_phases)]
        tau_mapping = BiMapping([None, None, None, None, 0, None, 0, 1, 2, 3, 1, 2, 3], [4, 7, 8, 9])
        self.tau_mapping = [tau_mapping for _ in range(self.n_phases)]
        self.n_q = q_mapping.to_first.len
        self.n_qdot = self.n_q
        self.n_tau = tau_mapping.to_first.len

    def _set_dynamics(self):
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Flat foot
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Toe only
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN)  # Aerial phase
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Toe only
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)  # Flat foot

    def _set_constraints(self):
        # Torque constrained to torqueMax
        for i in range(self.n_phases):
            self.constraints.add(maximal_tau, phase=i, node=Node.ALL, minimal_tau=self.tau_min)

        # Positivity of CoM_dot on z axis prior the take-off
        self.constraints.add(com_dot_z, phase=self.takeoff, node=Node.END, min_bound=0, max_bound=np.inf)

        # Constraint arm positivity (prevent from local minimum with arms in the back)
        self.constraints.add(
            ConstraintFcn.TRACK_STATE, phase=self.takeoff, node=Node.END, index=3, min_bound=1.0, max_bound=np.inf
        )

        # Floor constraints for flat foot phases
        for p in self.flat_foot_phases:
            # Do not pull on floor
            for i in self.heel_and_toe_contact_idx:
                self.constraints.add(
                    ConstraintFcn.CONTACT_FORCE, phase=p, node=Node.ALL, contact_force_idx=i, max_bound=np.inf
                )

            # Non-slipping constraints
            self.constraints.add(  # On only one of the feet
                ConstraintFcn.NON_SLIPPING,
                phase=p,
                node=Node.ALL,
                normal_component_idx=(1, 2),
                tangential_component_idx=0,
                static_friction_coefficient=0.5,
            )

        # Floor constraints for toe only phases
        for p in self.toe_only_phases:
            # Do not pull on floor
            for i in self.toe_contact_idx:
                self.constraints.add(
                    ConstraintFcn.CONTACT_FORCE, phase=p, node=Node.ALL, contact_force_idx=i, max_bound=np.inf
                )

            # The heel must remain over floor
            self.constraints.add(
                marker_on_floor,
                phase=p,
                node=Node.ALL,
                min_bound=-0.001,
                max_bound=np.inf,
                marker=self.heel_marker,
                floor=self.floor,
            )

            # Non-slipping constraints
            self.constraints.add(  # On only one of the feet
                ConstraintFcn.NON_SLIPPING,
                phase=p,
                node=Node.ALL,
                normal_component_idx=1,
                tangential_component_idx=0,
                static_friction_coefficient=0.5,
            )

    def _set_objective_functions(self):
        # Maximize the jump height
        self.objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-100, phase=self.takeoff)

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

        for i in range(self.n_phases):
            # Minimize time of the phase
            if self.time_min[i] != self.time_max[i]:
                self.objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_TIME,
                    weight=0.1,
                    phase=i,
                    min_bound=self.time_min[i],
                    max_bound=self.time_max[i],
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
                marker=self.toe_marker,
                floor=self.floor,
            )
        if self.n_phases >= 4:  # 2 contacts on floor
            self.constraints.add(
                marker_on_floor,
                phase=3,
                node=Node.END,
                min_bound=-0.001,
                max_bound=0.001,
                marker=self.heel_marker,
                floor=self.floor,
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
                u_init_guess.add(ctrl[i]["all"][:, :-1], interpolation=InterpolationType.EACH_FRAME)
                x_init_guess.add(state[i]["all"], interpolation=InterpolationType.EACH_FRAME)

            time_init_guess = InitialGuess(param["time"], name="time")
            ocp.update_initial_guess(x_init=x_init_guess, u_init=u_init_guess, param_init=time_init_guess)
            ocp.solver.set_lagrange_multiplier(sol)

        # Run optimizations
        if not force_no_graph:
            add_custom_plots(self.ocp, self.tau_min)

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
