#!/bin/bash
#PBS -l nodes=1:ppn=8,walltime=30:00:00
#PBS -N my_job
cd $PBS_O_WORKDIR
module load gcc intel python NumPy
python run_multiprocessing.py -n 3 -s "test1"
module purge
