from copy import copy

import biorbd_casadi as biorbd
import numpy as np

m = biorbd.Model("../models/jumper2contacts.bioMod")

q0 = np.array([0, 0, -0.5336, 0, 1.4, 0, 1.4, 0.8, -0.9, 0.47, 0.8, -0.9, 0.47])
q1 = np.array([-0.12, -0.23, -1.10, 0, 1.85, 0, 1.85, 2.06, -1.67, 0.55, 2.06, -1.67, 0.55])
q2 = np.array(
    [
        -0.02323732,
        0.00370982,
        -1.13177391,
        0.0,
        0.98645467,
        -0.0,
        0.98645467,
        1.4230514,
        -0.75729551,
        0.15237644,
        1.4230514,
        -0.75729551,
        0.15237644,
    ]
)

qdot_null = np.zeros((13,))
qdot0 = np.array(
    [
        -0.69722964,
        -1.33958085,
        -1.8479633,
        0.0,
        -0.64892519,
        -0.0,
        -0.64892519,
        6.15902598,
        -5.95116095,
        1.63984244,
        6.15902598,
        -5.95116095,
        1.63984244,
    ]
)
qdot1 = np.array(
    [
        -0.49307798,
        -0.73470639,
        -1.5045398,
        0.0,
        3.29799032,
        -0.0,
        3.29799032,
        4.48009106,
        -4.34959904,
        1.37437795,
        4.48009106,
        -4.34959904,
        1.37437795,
    ]
)
qdot2 = np.array(
    [
        0.03517037,
        -0.53697087,
        -1.70861396,
        0.0,
        0.84842762,
        -0.0,
        0.84842762,
        0.6213254,
        1.95644332,
        3.43022919,
        0.6213254,
        1.95644332,
        3.43022919,
    ]
)

torque_act = np.array([0, 0, 0, 0, 1, 0, 1, -1, 1, -1, -1, 1, -1])

torque0 = [m.torque(torque_act, q0, qdot_null).to_array(), m.torque(torque_act, q0, qdot0).to_array()]
torque1 = [m.torque(torque_act, q1, qdot_null).to_array(), m.torque(torque_act, q1, qdot1).to_array()]
torque2 = [m.torque(torque_act, q2, qdot_null).to_array(), m.torque(torque_act, q2, qdot2).to_array()]

name = [m.nameDof()[i].to_string() for i in range(m.nbDof())]

"""
torque_evolution0 = {"Pelvis_TransY": torque0[1][0]/torque0[0][0]*100, "Pelvis_TransZ": torque0[1][1]/torque0[0][1]*100, "Pelvis_RotX": torque0[1][1]/torque0[0][1]*100, "BrasD_RotZ": torque0[1][2]/torque0[0][2]*100, "BrasD_RotX": torque0[1][3]/torque0[0][3]*100, "BrasG_RotZ": torque0[1][4]/torque0[0][4]*100, "BrasG_RotX": torque0[1][5]/torque0[0][5]*100, "CuisseD_RotX": torque0[1][6]/torque0[0][6]*100, "JambeD_RotX": torque0[1][7]/torque0[0][7]*100, 
"""

torque_evolution0 = {f"{name[i]}": torque0[1][i] / torque0[0][i] * 100 for i in range(len(name))}
torque_evolution1 = {f"{name[i]}": torque1[1][i] / torque1[0][i] * 100 for i in range(len(name))}
torque_evolution2 = {f"{name[i]}": torque2[1][i] / torque2[0][i] * 100 for i in range(len(name))}

tau_from_id_qdot_null_0 = m.InverseDynamics(q0, qdot_null, np.zeros((13,)), None).to_array()
tau_no_root_0 = copy(tau_from_id_qdot_null_0)
tau_no_root_0[
    :3,
] = 0

tau_from_id_qdot_null_1 = m.InverseDynamics(q1, qdot_null, np.zeros((13,)), None).to_array()
tau_no_root_1 = copy(tau_from_id_qdot_null_1)
tau_no_root_1[
    :3,
] = 0

tau_from_id_qdot_null_2 = m.InverseDynamics(q2, qdot_null, np.zeros((13,)), None).to_array()
tau_no_root_2 = copy(tau_from_id_qdot_null_2)
tau_no_root_2[
    :3,
] = 0


cs_0 = m.getConstraints()
qddot0 = m.ForwardDynamicsConstraintsDirect(q0, qdot_null, tau_no_root_0, cs_0).to_array()
cs_1 = m.getConstraints()
qddot1 = m.ForwardDynamicsConstraintsDirect(q1, qdot_null, tau_no_root_1, cs_1).to_array()
cs_2 = m.getConstraints()
qddot2 = m.ForwardDynamicsConstraintsDirect(q2, qdot_null, tau_no_root_2, cs_2).to_array()

print("\n")
print(f"Mass of the model : {m.mass()}")
print("3 key positions: node 0 (with qdot0 = qdot at node 1), node 45, and node 54\n")
print(
    f"For position 0: \n\n-> With qdot null, torqueMax and forces are: \n{torque0[0]} \n{cs_0.getForce().to_array()} \n\n-> With non-zero qdot: \n{torque0[1]}"
)
print("\n")
keys = list(torque_evolution0.keys())
for i in range(len(keys)):
    print(
        f"{keys[i]}: \n	from {torque0[0][i]} without velocity to {torque0[1][i]} with non-zero velocity i.e. {round(torque_evolution0[keys[i]],2)}%"
    )
print("\n\n")
print(
    f"For position 1: \n\n-> With qdot null, torqueMax and forces are: \n{torque1[0]} \n{cs_1.getForce().to_array()} \n\n-> With non-zero qdot: \n{torque1[1]}"
)
print("\n")
for i in range(len(keys)):
    print(
        f"{keys[i]}: \n	from {torque1[0][i]} without velocity to {torque1[1][i]} with non-zero velocity i.e. {round(torque_evolution1[keys[i]],2)}%"
    )
print("\n")


"""
for (i, q) in enumerate(q_mat):
	print(f"\n\n\nFor position {i}: \n\n-> With qdot null : \n{m.torque(torque_act, q, qdot_null).to_array()}  \n->With non-zero qdot: \n{m.torque(torque_act, q, qdot[i]).to_array()}")
"""
