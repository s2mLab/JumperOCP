import jumper2phases, jumper3phases, jumper5phases

from multiprocessing import Pool
import os
import getpass

calls2p = []
calls3p = []
calls5p = []
pwd = getpass.getpass()
for a in range(-5, 6, 2):
    for b in range(-5, 6, 2):
        calls2p.append([a / 10, b / 10, pwd])
        for c in range(-5, 6, 2):
            calls3p.append([a / 10, b / 10, c / 10, pwd])
            for d in range(-5, 6, 2):
                for e in range(-5, 6, 2):
                    calls5p.append([a / 10, b / 10, c / 10, d / 10, e / 10, pwd])

with Pool(2) as p:
    p.map(jumper2phases.main, calls2p)
    p.map(jumper5phases.main, calls5p)
    p.map(jumper3phases.main, calls3p)
