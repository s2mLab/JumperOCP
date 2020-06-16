from copy import copy

import biorbd
import numpy as np 

m = biorbd.Model("/home/iornaith/Documents/GitKraken/JumperOCP/models/jumper2contacts.bioMod")

q0 = np.array([0, 0, -0.5336, 0, 1.4, 0, 1.4, 0.8, -0.9, 0.47, 0.8, -0.9, 0.47])
q1 = np.array([-0.12, -0.23, -1.10, 0, 1.85, 0, 1.85, 2.06, -1.67, 0.55, 2.06, -1.67, 0.55])
q2 = np.array([-0.02323732,  0.00370982, -1.13177391, 0., 0.98645467, -0.,  0.98645467, 1.4230514 , -0.75729551,  0.15237644, 1.4230514, -0.75729551, 0.15237644])

qdot_null = np.zeros((13,))
qdot0 = np.array([-0.69722964, -1.33958085, -1.8479633 ,  0.        , -0.64892519,
       -0.        , -0.64892519,  6.15902598, -5.95116095,  1.63984244,
        6.15902598, -5.95116095,  1.63984244])
qdot1 = np.array([-0.49307798, -0.73470639, -1.5045398 ,  0.        ,  3.29799032,
       -0.        ,  3.29799032,  4.48009106, -4.34959904,  1.37437795,
        4.48009106, -4.34959904,  1.37437795])
qdot2 = np.array([ 0.03517037, -0.53697087, -1.70861396,  0.        ,  0.84842762,
       -0.        ,  0.84842762,  0.6213254 ,  1.95644332,  3.43022919,
        0.6213254 ,  1.95644332,  3.43022919])

torque_act = np.array([0, 0, 0, 0, 1, 0, 1, -1, 1, -1, -1, 1, -1])

name = [m.nameDof()[i].to_string() for i in range(m.nbDof())] 

print(f"\n\nMass of the model : {m.mass()}")
print("3 key positions: node 0 (with qdot0 = qdot at node 1), node 45, and node 54\n")

def computeTorque(i, q, qdot):
	torque = [m.torque(torque_act, q, qdot_null).to_array(), m.torque(torque_act, q, qdot).to_array()]
	torque_evolution = {f"{name[i]}": torque[1][i]/torque[0][i]*100 for i in range(len(name))}
	tau_from_id_qdot_null = m.InverseDynamics(q, qdot_null, np.zeros((13,)), None).to_array()
	tau_no_root = copy(tau_from_id_qdot_null)
	tau_no_root[:3,] = 0
	cs = m.getConstraints()
	qddot = m.ForwardDynamicsConstraintsDirect(q, qdot_null, tau_no_root, cs).to_array()

	print(f"\n\nFor position {i}: \n\n-> With qdot null, torqueMax and forces are: \n{torque[0]} \n{cs.getForce().to_array()} \n\n-> With non-zero qdot: \n{torque[1]}")
	print("\n")
	keys = list(torque_evolution.keys())
	for i in range(len(keys)):
		print(f"{keys[i]}: \n	from {torque[0][i]} without velocity to {torque[1][i]} with non-zero velocity i.e. {round(torque_evolution[keys[i]],2)}%")
	print("\n\n")
	
computeTorque(0, q0, qdot0)
computeTorque(1, q1, qdot1)
computeTorque(2, q2, qdot2)

