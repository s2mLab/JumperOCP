import utils


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

    return utils.optimize_jumping_ocp(
        model_path=model_path,
        phase_time=phase_time,
        ns=number_shooting_points,
        time_min=time_min,
        time_max=time_max,
    )


if __name__ == "__main__":
    ocp, sol = jumping_4phases_parameters()
    sol.animate(n_frames=241)
