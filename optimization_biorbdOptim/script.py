import jumper2phases, jumper3phases, jumper5phases

from multiprocessing import Pool
import getpass

calls = []
pwd = getpass.getpass()
for a in range(-5, 6, 2):
    for b in range(-5, 6, 2):
        for c in range(-5, 6, 2):
            for d in range(-5, 6, 2):
                calls.append([a / 10, b / 10, c / 10, d / 10, pwd])

with Pool(2) as p:
    p.map(jumper2phases.main, calls)
    p.map(jumper5phases.main, calls)
    p.map(jumper3phases.main, calls)
