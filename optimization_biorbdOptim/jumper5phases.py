from JumperOcp import JumperOcp

if __name__ == "__main__":
    jumper = JumperOcp(path_to_models="../models/", n_phases=5)
    sol = jumper.solve(limit_memory_max_iter=50, exact_max_iter=1000)
    sol.print()
    sol.animate()
