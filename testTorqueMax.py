import biorbd
import numpy as np 

m = biorbd.Model("/home/iornaith/Documents/GitKraken/JumperOCP/models/jumper2contacts.bioMod")

q0 = np.array([0, 0, -0.5336, 0, 1.4, 0, 1.4, 0.8, -0.9, 0.47, 0.8, -0.9, 0.47])

torque_act_pos = np.array([1]*13)
torque_act_neg = np.array([-1]*13)

rangeQdot = np.linspace(-10*3.14, 10*3.14, 20)

for i in range(len(rangeQdot)):
	qdot = np.array([rangeQdot[i]]*13)
	print(f"\nAt iteration {i}: \nPositive -> \n{m.torque(torque_act_pos, q0,qdot).to_array()} \nand negative: \n{m.torque(torque_act_neg, q0,qdot).to_array()} \n")

