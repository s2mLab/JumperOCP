from bioptim import BiMapping


class Jumper:
    model_files = (
        "jumper2contacts.bioMod",
        "jumper1contacts.bioMod",
        "jumper1contacts.bioMod",
        "jumper1contacts.bioMod",
        "jumper2contacts.bioMod",
    )
    time_min = 0.2, 0.05, 0.6, 0.05, 0.1
    time_max = 0.5, 0.5, 2.0, 0.5, 0.5
    phase_time = 0.3, 0.2, 0.6, 0.2, 0.2
    n_shoot = 30, 15, 20, 30, 30

    q_mapping = BiMapping([0, 1, 2, 3, None, 4, None, 4, 5, 6, 7, 5, 6, 7], [0, 1, 2, 3, 5, 8, 9, 10])
    tau_mapping = BiMapping([0, None, None, None, None, 1, None, 1, 2, 3, 4, 2, 3, 4], [0, 5, 8, 9, 10])
    initial_states = []
    body_at_first_node = [0, 0, 0, 0, 2.10, 1.15, 0.80, 0.20]
    initial_velocity = [0, 0, 0, 0, 0, 0, 0, 0]
    tau_min = 20  # Tau minimal bound despite the torque activation
    heel_marker_idx = 85
    toe_marker_idx = 86

    floor_z = 0.0
    flat_foot_phases = 0, 4  # The indices of flat foot phases
    toe_only_phases = 1, 3  # The indices of toe only phases

    flatfoot_contact_x_idx = (1, 5)  # Contacts indices of heel and toe in bioMod 2 contacts
    flatfoot_contact_y_idx = (2, 6)  # Contacts indices of heel and toe in bioMod 2 contacts
    flatfoot_contact_z_idx = (0, 3, 4, 7)  # Contacts indices of heel and toe in bioMod 2 contacts
    flatfoot_non_slipping = ((1, 2), (0, 3))  # (X-Y components), Z components

    toe_contact_x_idx = (0, 3)  # Contacts indices of toe in bioMod 1 contact
    toe_contact_y_idx = (1, 4)  # Contacts indices of toe in bioMod 1 contact
    toe_contact_z_idx = (2, 5)  # Contacts indices of toe in bioMod 1 contact
    toe_non_slipping = ((0, 1), 2)  # (X-Y components), Z components
    static_friction_coefficient = 0.5
