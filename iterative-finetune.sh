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
pruningconfig='./configs/cifar_resnet50.json'
}

resnet56

export CUDA_VISIBLE_DEVICES=0

echo 'Filter pruning ...'

for nr in 1 2 3 4
do
    echo 'Block pruning ... '$nr
    for method in imprinting
    do
        root=$dataset'_resnet56'
        prev=$(( $nr - 1 ))
        crit=$root'/crit'$prev'.pkl'
        dir=$root'/finetune'-$nr
        if [ "$nr" -gt 1 ] 
        then
            loadmodel=$root'/finetune-'$prev'/best_model.pth.tar'
        fi

        echo "Checkpoint directory: " $dir
        echo $loadmodel

        python finetune.py --dataset $dataset --arch $model --depth $depth --save $dir --remove $nr --criterion $crit \
            --lr=$lr --lr-decay-every=$lrdecayeach --momentum=0.9 --epochs=30 --batch-size=128 \
            --load-model $loadmodel --iterative

        curmodel=$dir'/best_model.pth.tar'

        python iterative_imprint_cifar.py -d $dataset --arch resnet56 --pretrained $curmodel \
            -c $root --remove-layers $nr --criterion $crit

    done
done
