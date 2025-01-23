import os
from smartsim import Experiment
from smartsim.settings import PalsMpiexecSettings


if __name__ == '__main__':
    # Initialize the SmartSim Experiment
    PORT = 6780
    exp = Experiment('test', launcher='pbs')

    # Set sim settings and launch
    nprocs = 4
    pwd = os.getenv('PWD')
    sim_settings = PalsMpiexecSettings(
                                        'python',
                                        exe_args=pwd+'/sim.py',
                                        env_vars={'MPICH_OFI_CXI_PID_BASE':str(0)}
    )
    sim_settings.set_tasks(nprocs)
    sim_settings.set_tasks_per_node(nprocs)
    colo_model = exp.create_model("sim", sim_settings)
    colo_model.colocate_db_uds(db_cpus=1, debug=False)
    exp.generate(colo_model, overwrite=True)
    exp.start(colo_model, block=False, summary=False)

    # Set vis settings and launch
    SSDB = colo_model.run_settings.env_vars['SSDB']
    vis_settings = PalsMpiexecSettings('python', 
                                        exe_args=pwd+'/vis.py',
                                        env_vars={'SSDB':SSDB, 'MPICH_OFI_CXI_PID_BASE':str(1)}
    )
    vis_settings.set_tasks(nprocs)
    vis_settings.set_tasks_per_node(nprocs)
    vis_model = exp.create_model("vis", vis_settings)
    exp.generate(vis_model, overwrite=True)
    exp.start(vis_model, block=True, summary=False)


