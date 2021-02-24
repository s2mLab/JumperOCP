import numpy as np
import biorbd
from casadi import if_else, lt, vertcat

from bioptim import PenaltyNodes


def maximal_tau(nodes: PenaltyNodes, minimal_tau):
    nlp = nodes.nlp
    nq = nlp.mapping["q"].to_first.len
    q = [nlp.mapping["q"].to_second.map(mx[:nq]) for mx in nodes.x]
    qdot = [nlp.mapping["qdot"].to_second.map(mx[nq:]) for mx in nodes.x]

    min_bound = []
    max_bound = []
    func = biorbd.to_casadi_func("torqueMax", nlp.model.torqueMax, nlp.q, nlp.qdot)
    for n in range(len(nodes.u)):
        bound = func(q[n], qdot[n])
        min_bound.append(
            nlp.mapping["tau"].to_first.map(if_else(lt(bound[:, 1], minimal_tau), minimal_tau, bound[:, 1]))
        )
        max_bound.append(
            nlp.mapping["tau"].to_first.map(if_else(lt(bound[:, 0], minimal_tau), minimal_tau, bound[:, 0]))
        )

    obj = vertcat(*nodes.u)
    min_bound = vertcat(*min_bound)
    max_bound = vertcat(*max_bound)

    return (
        vertcat(np.zeros(min_bound.shape), np.ones(max_bound.shape) * -np.inf),
        vertcat(obj + min_bound, obj - max_bound),
        vertcat(np.ones(min_bound.shape) * np.inf, np.zeros(max_bound.shape)),
    )


def com_dot_z(nodes: PenaltyNodes):
    nlp = nodes.nlp
    x = nodes.x
    q = nlp.mapping["q"].to_second.map(x[0][: nlp.shape["q"]])
    qdot = nlp.mapping["q"].to_second.map(x[0][nlp.shape["q"] :])
    com_dot_func = biorbd.to_casadi_func("com_dot", nlp.model.CoMdot, nlp.q, nlp.qdot)
    com_dot = com_dot_func(q, qdot)
    return com_dot[2]


def marker_on_floor(nodes: PenaltyNodes, marker, floor):
    nlp = nodes.nlp
    nb_q = nlp.shape["q"]
    q_reduced = nodes.x[0][:nb_q]
    q = nlp.mapping["q"].to_second.map(q_reduced)
    marker_func = biorbd.to_casadi_func("toe_on_floor", nlp.model.marker, nlp.q, marker)
    marker_z = marker_func(q)[2]
    return marker_z + floor
