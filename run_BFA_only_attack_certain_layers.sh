#!/usr/bin/env bash

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"alpha")
    PYTHON="python3" #"/usr/bin/python3" # python environment path
    TENSORBOARD='/home/elliot/anaconda3/envs/bindsnet/bin/tensorboard' # tensorboard environment path
    data_path="/data1/cifar-10-batches-py" # dataset path
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard displaymodel=deit_base_patch16_224_test 

model=deit_tiny_cifar100
layer='head_details'

optimizer='AdamW'
dataset='cifar100'
test_batch_size=256 # number of training examples used in every iteration # ZX: batch size for testing set

#seeds=(8411)
seeds=(1 38 43 60 99 476 6611 5897 8411)

attack_sample_size=128 # number of data used for BFA  # ZX: batch size for training set
n_iter=100  # ZX: # of iterations for cross-layer search = # of total bits flipped
k_top=10 # only check k_top weights with top gradient ranking in each layer
# k_top=147456 for tiny

epochs=0
# vit tiny -> 25

save_path=/data1/Xuan_vit_ckp/${DATE}/${dataset}_${model}_${epochs}_${optimizer}
tb_path=/data1/Xuan_vit_ckp/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize}/tb_log  #tensorboard log path

############### Neural network ############################
{
for seed in "${seeds[@]}"
do
	python3 retrain.py --dataset ${dataset} --data_path /data1/   \
    		--arch ${model} --save_path ${save_path}  \
    		--test_batch_size ${test_batch_size} --workers 8 --ngpu 2 \
    		--evaluate \
    		--print_freq 1 --epochs ${epochs} \
    		--reset_weight --bfa --n_iter ${n_iter} --k_top ${k_top} --attack_sample_size ${attack_sample_size} \
    		--manualSeed ${seed} \
    		--optimizer ${optimizer}  | tee log_Adam/${model}_${layer}_seed${seed}.log
done
} &
# n_iter: number of iteration to perform BFA
# k_top: only check k_top weights with top gradient ranking in each layer (nb in paper)
# attack_sample_size: number of data used for BFA (batch_size in main.py)
# model: the ML model, related files can be found in models/vanilla_models. All models in this folder is pre-trained ResNet.

############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 30 
    wait
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 45
    wait
    case $HOST in
    "Hydrogen")
        firefox http://0.0.0.0:6006/
        ;;
    "alpha")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait
