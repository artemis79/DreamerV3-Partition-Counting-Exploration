#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=1 
#SBATCH --gpus-per-node=1 
#SBATCH --mem=16G 
#SBATCH --time=0-2:59
#SBATCH --array=1-30
#SBATCH --output=model_free/vanilla_count_model_free_%j.out



echo "Starting task $SLURM_ARRAY_TASK_ID"
# SOCKS5 Proxy
if [ "$SLURM_TMPDIR" != "" ]; then
    echo "Setting up SOCKS5 proxy..."
    ssh -q -N -T -f -D 8888 `echo $SSH_CONNECTION | cut -d " " -f 3`
    export ALL_PROXY=socks5h://localhost:8888
fi
 
module load python/3.11.5 cuda/12.2 StdEnv/2023

cd $SLURM_TMPDIR

export ALL_PROXY=socks5h://localhost:8888

# Clone project
git config --global http.proxy 'socks5://127.0.0.1:8888'
git clone https://github.com/artemis79/DreamerV3-Partition-Counting-Exploration.git


python -m venv .venv
source .venv/bin/activate

pip install requests[socks] --no-index

cd DreamerV3-Partition-Counting-Exploration/

cp $HOME/scratch/dreamerv3/requirements.txt .
pip install -U -r requirements.txt 

AutoROM --accept-license
pip install "autorom[accept-rom-license]"

export ALE_ROM_PATH="/tmp/.venv/lib/python3.11/site-packages/AutoROM/roms"



