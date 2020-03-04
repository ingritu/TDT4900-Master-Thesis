# My notes regarding running on Epic/Idun

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