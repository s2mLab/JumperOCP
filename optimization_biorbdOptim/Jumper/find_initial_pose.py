import biorbd
from scipy import optimize
import numpy as np
from casadi import MX
from bioptim import BiMapping


def find_initial_root_pose(model, body_pose_no_root):
    body_pose_no_root = np.array(body_pose_no_root)
    bimap = BiMapping([None] + [0, 1, 2] + [None] * body_pose_no_root.shape[0], [1, 2, 3])

    bound_min = []
    bound_max = []
    for i in range(model.nbSegment()):
        seg = model.segment(i)
        for r in seg.QRanges():
            bound_min.append(r.min())
            bound_max.append(r.max())
    bound_min = bimap.to_first.map(np.array(bound_min)[:, np.newaxis])
    bound_max = bimap.to_first.map(np.array(bound_max)[:, np.newaxis])
    root_bounds = (list(bound_min[:, 0]), list(bound_max[:, 0]))

    q_sym = MX.sym("Q", model.nbQ(), 1)
    com_func = biorbd.to_casadi_func("com", model.CoM, q_sym)
    contacts_func = biorbd.to_casadi_func("contacts", model.constraintsInGlobal, q_sym, True)
    shoulder_jcs_func = biorbd.to_casadi_func("shoulder_jcs", model.globalJCS, q_sym, 3)
    hand_marker_func = biorbd.to_casadi_func("hand_marker", model.marker, q_sym, 32)

    def objective_function(q_root, *args, **kwargs):
        # Center of mass
        q = bimap.to_second.map(q_root[:, np.newaxis])[:, 0]
        q[model.nbRoot():] = body_pose_no_root
        com = np.array(com_func(q))
        contacts = np.array(contacts_func(q))[:, [0, 1, 3, 4]]
        mean_contacts = np.mean(contacts, axis=1)
        shoulder_jcs = np.array(shoulder_jcs_func(q))
        hand = np.array(hand_marker_func(q))

        # Prepare output
        out = np.ndarray((10, ))

        # The center of contact points should be at 0
        out[0] = mean_contacts[0]
        out[1] = mean_contacts[1]
        out[2] = contacts[2, 0]
        out[3] = contacts[2, 1]
        out[4] = contacts[2, 2]
        out[5] = contacts[2, 3]

        # The projection of the center of mass should be at 0 and at 0.95 meter high
        out[6] = com[0]
        out[7] = com[1]
        out[8] = com[2] - 0.95

        # Keep the arms horizontal
        out[9] = shoulder_jcs[2, 3] - hand[2]

        return out

    q_root0 = np.mean(root_bounds, axis=0)
    pos = optimize.least_squares(objective_function, x0=q_root0, bounds=root_bounds)
    root = np.zeros(model.nbRoot())
    root[bimap.to_first.map_idx] = pos.x
    return root


if __name__ == "__main__":
    import bioviz

    model_path = "../../models/jumper2contacts.bioMod"
    model = biorbd.Model(model_path)
    pos_body_no_root = [0, 2.10, 0, 2.10, 1.15, 0.80, 0.20, 1.15, 0.80, 0.20]

    initial_root_pose = find_initial_root_pose(model, pos_body_no_root)
    body_pose = np.array(list(initial_root_pose) + pos_body_no_root)
    print(f"The initial pose is {body_pose}")

    b = bioviz.Viz(loaded_model=model, markers_size=0.003, show_markers=True)
    b.set_q(body_pose)
    b.exec()
