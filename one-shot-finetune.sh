#!/bin/bash
nvidia-smi

latencyratio=-1
wd=1e-4
optim=sgd
lr=0.001
lrdecayeach=10
ratio=1.0 #Ratio of used dataset
dataset=CIFAR10

resnet56(){
model=resnet
depth=56
loadmodel=$dataset'-baseline/'$model'50/model_best.pth.tar'
pruningconfig='./configs/cifar_resnet50.json'
}

resnet56

export CUDA_VISIBLE_DEVICES=0

for nr in 1 2 3 4
do
    echo 'Block pruning ... '$nr
    root='cifar10_resnet56'
    crit=$root'/crit0.pkl'
    dir=$root'/finetune-one-shot-'$nr

    echo "Checkpoint director: " $dir
    python finetune.py --dataset $dataset --arch $model --depth $depth --save $dir --remove $nr --criterion $crit \
        --lr=$lr --lr-decay-every=$lrdecayeach --momentum=0.9 --epochs=30 --batch-size=128 \
        --load-model $loadmodel
done
