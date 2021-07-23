from JumperOcp import JumperOcp
from bioptim import ControlType

if __name__ == "__main__":
    jumper = JumperOcp(path_to_models="../models/", control_type=ControlType.CONSTANT, n_phases=3)
    sol = jumper.solve(limit_memory_max_iter=100, exact_max_iter=1000)
    sol.print()
    sol.animate()
