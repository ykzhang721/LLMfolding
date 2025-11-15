export WANDB_API_KEY=bc2e2b14aacbadfd88a86ceab37243b8944b0eaf
export WANDB_BASE_URL=https://api.bandw.top
export HF_DATASETS_CACHE=.cache/hf_datasets
torchrun --master_port=29505 --nnodes=1 --nproc_per_node=2 pipe.py --config-name='folding'
