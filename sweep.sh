#!/bin/bash

SHOTS=(2)
SEEDS=$(seq 1 128)

BASE_SAVE_DIR="ckpt"
MAX_JOBS=3   # maximum number of simultaneous jobs

mkdir -p ${BASE_SAVE_DIR}

function wait_for_slot() {
  while true; do
    NJOBS=$(squeue -u $USER -h -o "%j" | grep -c "^aaclip")
    if [ "$NJOBS" -lt "$MAX_JOBS" ]; then
      break
    fi
    echo "[$(date)] ${NJOBS} jobs running... waiting for a free slot"
    sleep 30
  done
}

for shot in "${SHOTS[@]}"; do
  for seed in ${SEEDS}; do

    wait_for_slot

    SAVE_PATH="${BASE_SAVE_DIR}/shot_${shot}/seed_${seed}"
    mkdir -p ${SAVE_PATH}

    echo "Submitting job: shot=${shot}, seed=${seed}"

    sbatch <<EOT
#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --partition=ea
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40-48:1
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=aaclip_s${shot}_seed${seed}
#SBATCH --output=${SAVE_PATH}/job_%j.out
#SBATCH --error=${SAVE_PATH}/job_%j.err

module purge
module load pytorch/2.6

python train.py \
  --shot ${shot} \
  --seed ${seed} \
  --save_path ${SAVE_PATH} \

EOT

  done
done