from copy import copy

import biorbd_casadi as biorbd
import numpy as np


def computeTorqueMaxAndForces(q, qdot):
    torqueMax = m.torque(torque_act, q, qdot).to_array()

    tau_no_root = copy(torqueMax)
    tau_no_root[
        :3,
    ] = 0
    cs = m.getConstraints()

    qddot_with_v = m.ForwardDynamicsConstraintsDirect(q, qdot, tau_no_root, cs).to_array()
    forces = cs.getForce().to_array()

    return torqueMax, forces


"""
	print(f"\n\nFor position {i}: \n\n-> With qdot null, torqueMax and forces are: \n{torque[0]} \n{cs_without_v.getForce().to_array()} \n\n-> With non-zero qdot torqueMax are: \n{torque[1]} \nand forces are : \n{cs_with_v.getForce().to_array()}")
	print("\n")
	keys = list(torque_evolution.keys())
	for i in range(len(keys)):
		print(f"{keys[i]}: \n	from {torque[0][i]} without velocity to {torque[1][i]} with non-zero velocity i.e. {round(torque_evolution[keys[i]],2)}%")
	print("\n\n")
"""

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

contact_names = [m.contactName(i).to_string() for i in range(m.nbContacts())]

print(f"\n\nMass of the model : {m.mass()}")
print("3 key positions: node 0 (with qdot0 = qdot at node 1), node 45, and node 54\n")

torqueMax_0_qdot_null, forces_0_qdot_null = computeTorqueMaxAndForces(q0, qdot_null)
torqueMax_0, forces_0 = computeTorqueMaxAndForces(q0, qdot0)
# torqueMax_evolution_0 = {f"{name[i]}": torqueMax_0_qdot_null[i]/torqueMax_0[i]*100 for i in range(len(name))}
contact_forces_evolution_0 = {
    f"{contact_names[i]}": forces_0_qdot_null[i] / forces_0[i] * 100 for i in range(len(contact_names))
}

print(
    f"\nFor position 0: \n\n -> With qdot null, torqueMax are: \n{torqueMax_0_qdot_null} \nand reaction forces are: \n {forces_0_qdot_null} \n\n"
)
print(f"-> With non-zero qdot, torqueMax are: \n{torqueMax_0} \nand reaction forces are: \n {forces_0} \n\n")
print("Regarding the effect of a non-zero qdot on contact forces:")
keys = list(contact_forces_evolution_0.keys())
for i in range(len(keys)):
    print(
        f"{keys[i]}: \n	from {forces_0_qdot_null[i]} with qdot null to {forces_0[i]} with non-zero velocity i.e. {round(contact_forces_evolution_0[keys[i]],2)}%"
    )

torqueMax_1_qdot_null, forces_1_qdot_null = computeTorqueMaxAndForces(q1, qdot_null)
torqueMax_1, forces_1 = computeTorqueMaxAndForces(q1, qdot1)
# torqueMax_evolution_1 = {f"{name[i]}": torqueMax_1_qdot_null[i]/torqueMax_1[i]*100 for i in range(len(name))}
contact_forces_evolution_1 = {
    f"{contact_names[i]}": forces_1_qdot_null[i] / forces_1[i] * 100 for i in range(len(contact_names))
}

print(
    f"\n\nFor position 1: \n\n -> With qdot null, torqueMax are: \n{torqueMax_1_qdot_null} \nand reaction forces are: \n {forces_1_qdot_null} \n\n"
)
print(f"-> With non-zero qdot, torqueMax are: \n{torqueMax_1} \nand reaction forces are: \n {forces_1} \n\n")
print("Regarding the effect of a non-zero qdot on contact forces:")
for i in range(len(keys)):
    print(
        f"{keys[i]}: \n	from {forces_1_qdot_null[i]} with qdot null to {forces_1[i]} with non-zero velocity i.e. {round(contact_forces_evolution_1[keys[i]],2)}%"
    )

torqueMax_2_qdot_null, forces_2_qdot_null = computeTorqueMaxAndForces(q2, qdot_null)
torqueMax_2, forces_2 = computeTorqueMaxAndForces(q2, qdot2)
# torqueMax_evolution_2 = {f"{name[i]}": torqueMax_2_qdot_null[i]/torqueMax_2[i]*100 for i in range(len(name))}
contact_forces_evolution_2 = {
    f"{contact_names[i]}": forces_2_qdot_null[i] / forces_2[i] * 100 for i in range(len(contact_names))
}

print(
    f"For position 2: \n\n -> With qdot null, torqueMax are: \n{torqueMax_2_qdot_null} \nand reaction forces are: \n {forces_2_qdot_null} \n\n"
)
print(f"-> With non-zero qdot, torqueMax are: \n{torqueMax_0} \nand reaction forces are: \n {forces_0} \n\n")
print("Regarding the effect of a non-zero qdot on contact forces:")
for i in range(len(keys)):
    print(
        f"{keys[i]}: \n	from {forces_2_qdot_null[i]} with qdot null to {forces_2[i]} with non-zero velocity i.e. {round(contact_forces_evolution_2[keys[i]],2)}%"
    )

print("\n\n")
