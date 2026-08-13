#!/bin/bash

###############################################
###     SLURM script for planet_sims        ###
###      ARRAY JOB runs all schedules       ###
###                                         ###
###     Tested on Uni Koeln Ramses Cluster  ###
###             v2.0 , 13.08.2026           ###
###             Author: Ankur Dev           ###
###############################################

### This script runs through all the schedule files
### launching SLURM ARRAY jobs for each schedule independently
### NOTE: ndets parameter must be set in Section 4
### The MODULE_ARRAYS parameter is set in Section 4

### section 1 - SLURM params

#SBATCH --partition=smp
#SBATCH --account=ag-riechers # For Ramses/ CCAT

#SBATCH --ntasks-per-node 24 # 24 # tasks per node ; num of processes per node
#SBATCH --cpus-per-task 8 #8 # number of cores per task ; threads per process

#SBATCH --nodes 1  # number of nodes
#SBATCH --array=0 #0-23  # Define the array size, 24 schs

#SBATCH --job-name=det100_n24c8_sim 
### Job name, det#ndets_#ntaskstotal_#corespertask_sim

#SBATCH --mem=200G                # max mem requested, per node [min 400G for 1000 dets]
### Maximum requested time (days-hrs:min:sec)
#SBATCH --time 0-00:30:00 #estimated runtime max 0-04:00:00

#SBATCH --output ./logs_sims/%A_%a_%x.out 
#SBATCH --error ./logs_sims/%A_%a_%x.err 
### To handle dir ownership issues
umask 0002
mkdir -p logs_sims
chmod g+rws logs_sims

### section 2 - Runtime env set-up
echo ""
echo "*************"
echo "Loading Modules and Env..."
module purge
### load all modules needed for Job
module load lib/mpi4py/3.1.5-gompi-2023b
module load lang/Miniconda3/23.9.0-0
### initiate conda
conda deactivate
conda activate toast3
echo "*************"

### section 3 - Job Logging
echo ""
echo "*************"
echo "Running Job..."
echo "Starting at `date`"
echo "Hostname $HOSTNAME"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Slurm Ntasks: $SLURM_NTASKS"
echo "Number of Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Number of CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Cores per Node: $SLURM_CPUS_ON_NODE"
echo "Total Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Current working directory is `pwd`"
echo "Python path: $(which python)"
echo "Using MPI lib: $(which mpirun)"
echo "mpi4py import version: $(python -c 'import mpi4py; print(mpi4py.__version__)')"
echo "Using GCC lib: $(which gcc)"
echo ""

### section 4 - Set Job parameters
### Set number of detectors. Use FULL for no trimming.
ndets=100
### ndets=FULL # FULL chooses the maximum ndets available

### Set module arrays: 1 selects w2, 3 selects w1/w2/w3 detectors
MODULE_ARRAYS=1 # choose from 1 or 3 arrays

### section 5 - Job Run
echo "***** EXEC SCRIPT *****"
echo `date '+%F %H:%M:%S'`
echo "***********************"
echo ""

### Collect all schedule files from input dir
schedule_files_list=($(basename -a input_files/schedules/*.txt))
### Select schedule based on array index
schedule_file=${schedule_files_list[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running schedule file: $schedule_file"
echo "Number of dets: $ndets"
echo "Module arrays: $MODULE_ARRAYS"
echo ""

### Making detector files
## Check if the current array task ID is not 1
if [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
    sleep 5
fi
## Run the fp_trim script with the specified number of detectors (ndets)
python -m scripts.fp_scripts.fp_trim $ndets --module-arrays $MODULE_ARRAYS
## Check if the fp_trim script executed successfully
if [ $? -ne 0 ]; then
    # If there was an error, print an error message and exit the script
    echo "Error occurred in FP file generation script. Exiting."
    exit 1
fi

### Simulating timestream data from schedules
mpirun -np $SLURM_NTASKS python3 sim_data_primecam_mpi.py -s $schedule_file -d $ndets --module-arrays $MODULE_ARRAYS

echo ""
echo "******** DONE *********"
echo `date '+%F %H:%M:%S'`
echo "***********************"


