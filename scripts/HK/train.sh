export HTTPS_PROXY="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export HTTP_PROXY="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export https_proxy="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export http_proxy="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export WANDB_API_KEY=bc2e2b14aacbadfd88a86ceab37243b8944b0eaf
export WANDB_IGNORE_GIT=True
export WANDB_INSECURE_DISABLE_SSL=True
torchrun --nproc_per_node=2 pipe.py --config-name='folding'
