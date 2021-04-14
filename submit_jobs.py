import os
import tempfile
from time import sleep

# fmt: off
JOBS = [
#     ('n256_g0_r30', 'owners,normal', '00:10:00', '10', 'python utils/cli_phate.py --n_knn 256 --gamma 0 --resolution 30'),
#     ('n512_g0_r15', 'owners,normal', '00:10:00', '10', 'python utils/cli_phate.py --n_knn 512 --gamma 0 --resolution 15'),
#     ('n768_g0_r10', 'owners,normal', '00:10:00', '10', 'python utils/cli_phate.py --n_knn 768 --gamma 0 --resolution 10'),
#     ('n1536_g0_r05', 'owners,normal', '00:30:00', '20', 'python utils/cli_phate.py --n_knn 1536 --gamma 0 --resolution 5'),
#     ('n2560_g0_r03', 'owners,normal', '00:30:00', '20', 'python utils/cli_phate.py --n_knn 2560 --gamma 0 --resolution 3'),
#     ('n7680_g0_r01', 'owners,normal', '01:00:00', '20', 'python utils/cli_phate.py --n_knn 7680 --gamma 0 --resolution 1'),
#     ('n8192_g0_r01', 'owners,normal', '05:00:00', '64', 'python utils/cli_phate.py --n_knn 8192 --gamma 0 --resolution 1'),
    # ('n256_g0.5_r30', 'owners,normal', '00:10:00', '10', 'python utils/cli_phate.py --n_knn 256 --gamma 0.5 --resolution 30'),
    # ('n512_g0.5_r15', 'owners,normal', '00:10:00', '10', 'python utils/cli_phate.py --n_knn 512 --gamma 0.5 --resolution 15'),
    # ('n768_g0.5_r10', 'owners,normal', '00:10:00', '10', 'python utils/cli_phate.py --n_knn 768 --gamma 0.5 --resolution 10'),
    # ('n1536_g0.5_r05', 'owners,normal', '00:30:00', '20', 'python utils/cli_phate.py --n_knn 1536 --gamma 0.5 --resolution 5'),
    # ('n2560_g0.5_r03', 'owners,normal', '00:30:00', '20', 'python utils/cli_phate.py --n_knn 2560 --gamma 0.5 --resolution 3'),
    # ('n256_g-0.5_r30', 'owners,normal', '00:10:00', '10', 'python utils/cli_phate.py --n_knn 256 --gamma -0.5 --resolution 30'),
    # ('n512_g-0.5_r15', 'owners,normal', '00:10:00', '10', 'python utils/cli_phate.py --n_knn 512 --gamma -0.5 --resolution 15'),
    # ('n768_g-0.5_r10', 'owners,normal', '00:10:00', '10', 'python utils/cli_phate.py --n_knn 768 --gamma -0.5 --resolution 10'),
    # ('n1536_g-0.5_r05', 'owners,normal', '00:30:00', '20', 'python utils/cli_phate.py --n_knn 1536 --gamma -0.5 --resolution 5'),
    # ('n2560_g-0.5_r03', 'owners,normal', '00:30:00', '20', 'python utils/cli_phate.py --n_knn 2560 --gamma -0.5 --resolution 3'),
    # ('anim-phate_n256_r30', 'owners,normal', '01:00:00', '10', 'python scripts/cli_animate_phate.py --n_knn 256 --gamma 0 --resolution 30 --topk 10'),
    # ('anim-phate_n512_r15', 'owners,normal', '10:00:00', '10', 'python scripts/cli_animate_phate.py --n_knn 512 --gamma 0 --resolution 15 --topk 10'),
    # ('anim-phate_n768_r10', 'owners,normal', '10:00:00', '10', 'python scripts/cli_animate_phate.py --n_knn 768 --gamma 0 --resolution 10 --topk 10'),
    # ('anim-phate_n1536_r05', 'owners,normal', '20:00:00', '20', 'python scripts/cli_animate_phate.py --n_knn 1536 --gamma 0 --resolution 05 --topk 10'),
    # ('anim-phate_n2560_r03', 'owners,normal', '20:00:00', '20', 'python scripts/cli_animate_phate.py --n_knn 2560 --gamma 0 --resolution 03 --topk 10'),
    # ('anim-phate_n1536_r05', 'mignot', '7-00:00', '10', 'python scripts/cli_animate_phate.py --n_knn 1536 --gamma 0 --resolution 05 --topk 10'),
    # ('anim-phate_n2560_r03', 'mignot', '7-00:00', '10', 'python scripts/cli_animate_phate.py --n_knn 2560 --gamma 0 --resolution 03 --topk 10'),
]
# fmt: on


def submit_job(jobname, partition, time, ncpus, command, *args):

    content = f"""#!/bin/bash
#
#SBATCH --job-name={jobname}
#SBATCH -p {partition}
#SBATCH --time={time}
#SBATCH --cpus-per-task={ncpus}
#SBATCH --mem=100GB
#SBATCH --output=/home/users/alexno/sleep-staging/logs/phate_{jobname}.out
#SBATCH --error=/home/users/alexno/sleep-staging/logs/phate_{jobname}.err
##################################################

source $PI_HOME/miniconda3/bin/activate
conda activate pt1.7
cd $HOME/sleep-staging

{command}
"""

    #     print('')
    #     print('#######################################################################################')
    #     print(content)
    #     print('#######################################################################################')
    #     print('')
    with tempfile.NamedTemporaryFile(delete=False) as j:
        j.write(content.encode())
    os.system(f"sbatch {j.name}")


if __name__ == "__main__":

    for jobinfo in JOBS:
        submit_job(*jobinfo)

    print("All jobs have been submitted!")
