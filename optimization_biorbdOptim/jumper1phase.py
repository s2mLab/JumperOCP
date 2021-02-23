import utils


def jumping_1phase_parameters():
    model_path = "../models/jumper2contacts.bioMod",
    time_min = 0.2,
    time_max = 1.0,
    phase_time = 0.6,
    number_shooting_points = 30,

    return utils.optimize_jumping_ocp(
        model_path=model_path,
        phase_time=phase_time,
        ns=number_shooting_points,
        time_min=time_min,
        time_max=time_max,
    )


if __name__ == "__main__":
    ocp, sol = jumping_1phase_parameters()
    sol.animate(n_frames=241)
