import numpy as np
import biorbd
from casadi import if_else, lt, vertcat

from bioptim import PenaltyNode


def maximal_tau(penalty_node: PenaltyNode, minimal_tau):
    if penalty_node.u is None:
        return None

    nlp = penalty_node.nlp
    nq = nlp.mapping["q"].to_first.len
    q = nlp.mapping["q"].to_second.map(penalty_node.x[:nq])
    qdot = nlp.mapping["qdot"].to_second.map(penalty_node.x[nq:])
    func = nlp.add_casadi_func("torqueMax", nlp.model.torqueMax, nlp.q, nlp.qdot)

    bound = func(q, qdot)
    min_bound = nlp.mapping["tau"].to_first.map(if_else(lt(bound[:, 1], minimal_tau), minimal_tau, bound[:, 1]))
    max_bound = nlp.mapping["tau"].to_first.map(if_else(lt(bound[:, 0], minimal_tau), minimal_tau, bound[:, 0]))
    obj = penalty_node.u

    return (
        vertcat(np.zeros(max_bound.shape), np.ones(min_bound.shape) * -np.inf),
        vertcat(obj + min_bound, obj - max_bound),
        vertcat(np.ones(min_bound.shape) * np.inf, np.zeros(max_bound.shape)),
    )


def com_dot_z(nodes: PenaltyNode):
    nlp = nodes.nlp
    x = nodes.x
    q = nlp.mapping["q"].to_second.map(x[: nlp.shape["q"]])
    qdot = nlp.mapping["q"].to_second.map(x[nlp.shape["q"] :])
    com_dot_func = biorbd.to_casadi_func("com_dot", nlp.model.CoMdot, nlp.q, nlp.qdot)
    com_dot = com_dot_func(q, qdot)
    return com_dot[2]


def marker_on_floor(nodes: PenaltyNode, marker, floor_z):
    nlp = nodes.nlp
    nb_q = nlp.shape["q"]
    q_reduced = nodes.x[:nb_q]
    q = nlp.mapping["q"].to_second.map(q_reduced)
    marker_func = biorbd.to_casadi_func("toe_on_floor", nlp.model.marker, nlp.q, marker)
    marker_z = marker_func(q)[2]
    return marker_z - floor_z
