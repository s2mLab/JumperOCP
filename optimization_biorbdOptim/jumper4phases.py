from Jumper import Jumper


def jumping_4phases_parameters():
    model_path = (
        "../models/jumper2contacts.bioMod",
        "../models/jumper1contacts.bioMod",
        "../models/jumper1contacts.bioMod",
        "../models/jumper1contacts.bioMod",
    )
    time_min = 0.2, 0.05, 0.6, 0.05
    time_max = 0.5, 0.5, 2.0, 0.5
    phase_time = 0.3, 0.2, 0.6, 0.2
    number_shooting_points = 30, 15, 20, 30
    pose_at_first_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]

    return Jumper(
        model_paths=model_path,
        n_shoot=number_shooting_points,
        time_min=time_min,
        phase_time=phase_time,
        time_max=time_max,
        initial_pose=pose_at_first_node,
    )


if __name__ == "__main__":
    jumper = jumping_4phases_parameters()
    sol = jumper.solve(limit_memory_max_iter=200, exact_max_iter=1000)
    sol.print()
    sol.animate()
