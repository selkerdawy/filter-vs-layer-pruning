#!/bin/bash
nvidia-smi

latencyratio=-1
wd=1e-4
optim=sgd
lr=0.001
lrdecayeach=10
ratio=1.0 #Ratio of used dataset
dataset=CIFAR100

resnet56(){
model=resnet
depth=56
loadmodel=$dataset'-baseline/'$model'50/model_best.pth.tar'
}

resnet56

nr=1

#for method in 0 2 6 22 30 31
for method in 0
do
    root=$dataset'/'$model'50/one_shot_criterion'$method
    crit=$root'/criteria_'$method'_importance.pickle'
    dir=$root'/finetune'-$nr

    echo "Checkpoint director: " $dir
    python finetune.py --dataset $dataset --arch $model --depth $depth --save $dir --remove $nr --criterion $crit --load-model $loadmodel --lr 1e-3 --epochs 30
done

