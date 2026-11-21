#!/usr/bin/env bash
#SBATCH --job-name=5fqj11-boltz
#SBATCH --array=1-5%1
#SBATCH --partition=gpu
#SBATCH --constraint=g6.xlarge
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH --output=logs/boltz11-%A_%a.log

set -eo pipefail
cd /mnt/work/projects/hg317_pocket_seed
source ./env.sh

seed="$SLURM_ARRAY_TASK_ID"
out="$PWD/results/boltz-5fqj-job11-$SLURM_ARRAY_JOB_ID/seed-$seed"
cache="$out/cache"
mkdir -p "$cache"
for asset in boltz2_conf.ckpt boltz2_aff.ckpt mols.tar mols; do
    test -e "$BOLTZ_ASSETS/$asset"
    ln -sfn "$BOLTZ_ASSETS/$asset" "$cache/$asset"
done

"$ENV_ROOT/env-boltz2/bin/boltz" predict "$PWD/inputs/boltz-5fqj-job11" \
    --model boltz2 --cache "$cache" --out_dir "$out/predictions" \
    --accelerator gpu --devices 1 --no_kernels \
    --sampling_steps 200 --recycling_steps 3 --diffusion_samples 1 \
    --num_workers 0 --preprocessing-threads 1 --output_format mmcif \
    --seed "$seed"
