#!/usr/bin/env sh

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
enable_tb_display=false 

model=deit_base_cifar100

optimizer='AdamW'
dataset='tiny_imagenet'
test_batch_size=256 # number of training examples used in every iteration # ZX: batch size for testing set
seed=17

attack_sample_size=128 # batch size for training set
n_iter=1  # of iterations for cross-layer search = # of total bits flipped
k_top=10 # only check k_top weights with top gradient ranking in each layer

lr=0.0001
epochs=0

save_path=/data1/Xuan_vit_ckp/${DATE}/${dataset}_${model}_${epochs}_${optimizer}
tb_path=/data1/Xuan_vit_ckp/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize}/tb_log  #tensorboard log path

############### Neural network ############################
{
python3 retrain.py --dataset ${dataset} --data_path /data1/tiny-imagenet-200/   \
    --arch ${model} --save_path ${save_path}  \
    --test_batch_size ${test_batch_size} --workers 8 --ngpu 2 \
    --evaluate --learning_rate ${lr} \
    --print_freq 1 --epochs ${epochs} \
    --reset_weight --bfa --n_iter ${n_iter} --k_top ${k_top} --attack_sample_size ${attack_sample_size} \
    --manualSeed ${seed} \
    --optimizer ${optimizer}  | tee log_Adam/train_attack_${model}_epoch${epochs}_lr1e-4_${dataset}.log
} &
# n_iter: number of iteration to perform BFA
# k_top: only check k_top weights with top gradient ranking in each layer 
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
