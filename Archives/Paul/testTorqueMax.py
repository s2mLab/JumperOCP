import biorbd
import numpy as np

m = biorbd.Model("../models/jumper2contacts.bioMod")

q0 = np.array([0, 0, -0.5336, 0, 1.4, 0, 1.4, 0.8, -0.9, 0.47, 0.8, -0.9, 0.47])

q1 = np.array([-0.12, -0.23, -1.10, 0, 1.85, 0, 1.85, 2.06, -1.67, 0.55, 2.06, -1.67, 0.55])

q_mat = [q0, q1]

torque_act = np.array([0, 0, 0, 0, 1, 0, 1, -1, 1, -1, -1, 1, -1])

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
qdot = [qdot0, qdot1]

torque0 = [m.torque(torque_act, q0, qdot_null).to_array(), m.torque(torque_act, q0, qdot0).to_array()]
torque1 = [m.torque(torque_act, q1, qdot_null).to_array(), m.torque(torque_act, q1, qdot1).to_array()]
name = [m.nameDof()[i].to_string() for i in range(m.nbDof())]

"""
torque_evolution0 = {"Pelvis_TransY": torque0[1][0]/torque0[0][0]*100, "Pelvis_TransZ": torque0[1][1]/torque0[0][1]*100, "Pelvis_RotX": torque0[1][1]/torque0[0][1]*100, "BrasD_RotZ": torque0[1][2]/torque0[0][2]*100, "BrasD_RotX": torque0[1][3]/torque0[0][3]*100, "BrasG_RotZ": torque0[1][4]/torque0[0][4]*100, "BrasG_RotX": torque0[1][5]/torque0[0][5]*100, "CuisseD_RotX": torque0[1][6]/torque0[0][6]*100, "JambeD_RotX": torque0[1][7]/torque0[0][7]*100, 
"""

torque_evolution0 = {f"{name[i]}": torque0[1][i] / torque0[0][i] * 100 for i in range(len(name))}
torque_evolution1 = {f"{name[i]}": torque1[1][i] / torque1[0][i] * 100 for i in range(len(name))}

print("\n")
print(f"For position 0: \n-> With qdot null: \n{torque0[0]} \n-> With non-zero qdot: \n{torque0[1]}")
print("\n")
keys = list(torque_evolution0.keys())
for i in range(len(keys)):
    print(
        f"{keys[i]}: \n	from {torque0[0][i]} without velocity to {torque0[1][i]} with non-zero velocity i.e. {round(torque_evolution0[keys[i]],2)}%"
    )
print("\n\n")
print(f"For position 1: \n-> With qdot null: \n{torque1[0]} \n-> With non-zero qdot: \n{torque1[1]}")
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
