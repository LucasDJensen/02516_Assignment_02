#!/bin/sh
### General options
### -- specify queue --
#BSUB -q c02516
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### –- specify queue --
### -- set the job Name --
#BSUB -J 02516_eval_runs
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need xGB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
###BSUB -R "select[gpu80gb]"
###BSUB -R "select[gpu32gb]"
#### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
###BSUB -M 128GB
### -- set walltime limit: hh:mm --
#BSUB -W 00:20
### -- set the email address --
#BSUB -u s210025@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- set the job output file --
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o 02516_eval_runs_%J.out
#BSUB -e 02516_eval_runs_%J.err
# all  BSUB option comments should be above this line!

# execute our command
source ~/02516_venv/bin/activate
module load cuda/12.8.1

python ~/02516_assignment_02/eval.py --runs_dir /zhome/65/f/160006/runs/no_leakage --data_dir /dtu/datasets1/02516/ucf101_noleakage --img_size 112 112 --batch_size 8 --num_workers 0
