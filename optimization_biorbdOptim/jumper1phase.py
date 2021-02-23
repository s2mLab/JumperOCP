from Jumper import Jumper


def jumping_1phase_parameters():
    model_path = "../models/jumper2contacts.bioMod",
    time_min = 0.2,
    time_max = 1.0,
    phase_time = 0.6,
    number_shooting_points = 30,
    pose_at_first_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]

    return Jumper(
        model_paths=model_path,
        n_shoot=number_shooting_points,
        time_min=time_min,
        phase_time=phase_time,
        time_max=time_max,
        initial_pose=pose_at_first_node
    )


if __name__ == "__main__":
    jumper = jumping_1phase_parameters()
    sol = jumper.solve(limit_memory_max_iter=200, exact_max_iter=1000)
    sol.animate()
