from JumperOcp import Jumper

jumper = Jumper("../models/")
pos_body = jumper.q_mapping.to_second.map(jumper.body_at_first_node)[:, 0]

initial_root_pose = jumper.find_initial_root_pose()
pos_body[:jumper.models[0].nbRoot()] = initial_root_pose
print(f"The initial pose is {pos_body}")
jumper.show(pos_body)
