# My notes regarding running on Epic/Idun
### Partitions
```
#SBATCH --partition=WORKQ
#SBATCH --partition=TRAINING
#SBATCH --partition=EPT
#SBATCH --partition=TEST
#SBATCH --partition=EPIC     #GPU
#SBATCH --partition=EPIC2    #GPU
#SBATCH --partition=EPICALL  #GPU
```
Find partitions by running:
```
scontrol show partition|grep ^Par
```
### Modules
Need to load modules.

## job.slurm file
If you do not need a lot of CPUs then use a partition with GPUs like EPICALL.

Use the share-ie-idi account not the ie-idi account.

```
#!/bin/sh
#SBATCH --partition=EPICALL
#SBATCH --account=share-ie-idi
#SBATCH --time=hh:mm:ss
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntask-per-node=1
#SBATCH --job-name="name-of-job"
#SBATCH --output=test-srun.out
#SBATCH --mail-user=iturkerud@gmail.com
#SBATCH --mail-type=ALL

cd ${HOME}/TDT4900-Master-Thesis
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo "the name of the job is $SLURM_JOB_NAME"
echo "the job ID is $SLURM_JOB_ID"
echo "the job was run on these nodes: $SLURM_JOB_NODELIST"
echo "number of nodes: $SLURM_JOB_NUM_NODES"
echo "we are using $SLURM_CPUS_ON_NODE cores"
echo "we are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module load foss/2019b
module load Python/3.7.4
source ${HOME}/env1/bin/activate

python3 -m src.namespace.module --args
```
Add gpus
```
#SBATCH --gres=gpu:1
```
Example from hpc documentation:
```
#!/bin/sh
#SBATCH --partition=EPIC2
#SBATCH --account=<account>
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1  
#SBATCH --job-name="LBM_CUDA"
#SBATCH --output=lbm_cuda.out

cd ${SLURM_SUBMIT_DIR}

module purge
module load fosscuda/2018b
```

## Running a job
1. Edit job.slurm file.
2. run:
   ```
   sbatch job.slurm
   ```
3. Find job in queue
   ```
   squeue -u <username>
   ```
4. View log.
   ```
   vim test-srun.out
   ```