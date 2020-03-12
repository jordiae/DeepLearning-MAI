 #!/bin/bash
    #SBATCH --job-name="prova1"
    #SBATCH --qos=debug
    #SBATCH --workdir=.
    #SBATCH --output=prova1_%j.out
    #SBATCH --error=prova1_%j.err
    #SBATCH --cpus-per-task=40
    #SBATCH --gres gpu:1
    #SBATCH --time=00:30:00

    module load gcc/8.3.0 cuda/10.1 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2
    module load fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1
    module load python/3.7.4_ML
    
    bash train.sh
    