import os
from scp import SCPClient
from paramiko import SSHClient


def main(args=None):
    # Collect the number of phases to optimize here
    n_phase = None

    init = None
    if args:
        init = args[:-1]
        pwd = args[-1]
        save_path = f"{n_phase}p_init_{init[0]}_{init[1]}_{init[2]}_{init[3]}_sol.bo"
        if os.path.exists(save_path):
            return

    # Call the proper jumperXphases.py here
    ocp, sol = None, None

    if init:
        ocp.save(sol, save_path)
        ocp.save_get_data(sol, save_path + 'b')
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect('pariterre.net', username='aws', password=pwd)
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(save_path, save_path)
            scp.get(save_path)
            scp.put(save_path + 'b', save_path + 'b')
            scp.get(save_path + 'b')


if __name__ == "__main__":
    main()
