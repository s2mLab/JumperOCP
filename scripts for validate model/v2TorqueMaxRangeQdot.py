from copy import copy
import biorbd
import numpy as np

m = biorbd.Model("/home/iornaith/Documents/GitKraken/JumperOCP/models/jumper2contacts.bioMod")

q0 = np.array([0, 0, -0.5336, 0, 1.4, 0, 1.4, 0.8, -0.9, 0.47, 0.8, -0.9, 0.47])
qdot = np.zeros((13,))

tau_from_id = m.InverseDynamics(q0, np.zeros((13,)), np.zeros((13,)), None).to_array()

tau_no_root = copy(tau_from_id)
tau_no_root[:3,] = 0

cs = m.getConstraints()
a = m.ForwardDynamicsConstraintsDirect(q0, qdot, tau_from_id, cs).to_array()
print(f"\nQddot: {a} \nForces: {cs.getForce().to_array()}")
print("")
print("removing root")
print(f"tau: {tau_no_root}")
cs = m.getConstraints()
b = m.ForwardDynamicsConstraintsDirect(q0, qdot, tau_no_root, cs).to_array()
print(f"\nQddot: {b} \nForces: {cs.getForce().to_array()}")
