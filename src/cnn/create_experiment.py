import os
import time
import sys


def create_slurm_script(name, queue, time):
    script = f'''#!/bin/bash
#SBATCH --job-name="{name}"
#SBATCH --qos={queue}
#SBATCH --workdir=.
#SBATCH --output={name}_%j.out
#SBATCH --error={name}_%j.err
#SBATCH --cpus-per-task=40
#SBATCH --gres gpu:1
#SBATCH --time={time}

module load gcc/8.3.0 cuda/10.1 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2
module load fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1
module load python/3.7.4_ML

bash train.sh
    '''
    return script


def create_train_script(parameters):
    relative_train_path = '/'.join(['..', '..', 'src', 'cnn', 'train.py'])
    script = f'''#!/bin/bash
python {relative_train_path} {parameters}'''
    return script


def create_experiment(name, parameters):
    train_script = create_train_script(parameters)
    slurm_debug_script = create_slurm_script(name, 'debug', '00:15:00')
    slurm_main_script = create_slurm_script(name, 'training', '12:00:00')
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    exp_dir = os.path.join('..', '..', 'experiments', f'{name}-{timestamp}')
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, 'train.sh'), 'w', newline='\n') as f:
        f.write(train_script)
    with open(os.path.join(exp_dir, 'debug_launcher.sh'), 'w', newline='\n') as f:
        f.write(slurm_debug_script)
    with open(os.path.join(exp_dir, 'main_launcher.sh'), 'w', newline='\n') as f:
        f.write(slurm_main_script)
    print(f'Created {exp_dir} experiment')


def main():
    name = sys.argv[1]
    create_experiment(name, ' '.join(sys.argv[2:]))


if __name__ == '__main__':
    main()
