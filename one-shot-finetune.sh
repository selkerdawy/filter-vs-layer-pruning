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

for method in 0 2 6 22 30
do
    root=$dataset'/'$model'50/one_shot_criterion'$method
    dir=$root'/filterpruning'

    echo "Checkpoint director: " $dir
    python main.py --name=$dir --dataset=$dataset \
        --lr=$lr --lr-decay-every=$lrdecayeach --momentum=0.9 --epochs=30 --batch-size=128 \
        --pruning=True --seed=0 --model=$model'50' \
        --mgpu=False --group_wd_coeff=1e-8 --wd=$wd --tensorboard=True --pruning-method=$method \
        --data=${datasetdir} --no_grad_clip=True --pruning_config=$pruningconfig \
        --only-estimate-latency=True \
        --data=${datasetdir} --ratio $ratio --prune-latency-ratio $latencyratio \
        --load_model $loadmodel
done

for nr in 1 2
do
    echo 'Block pruning ... '$nr
    for method in 0 2 6 22 30
    do
        root=$dataset'/'$model'50/one_shot_criterion'$method
        crit=$root'/criteria_'$method'_importance.pickle'
        dir=$root'/finetune'-$nr

        echo "Checkpoint director: " $dir
        python finetune.py --dataset $dataset --arch $model --depth $depth --save $dir --remove $nr --criterion $crit \
            --lr=$lr --lr-decay-every=$lrdecayeach --momentum=0.9 --epochs=30 --batch-size=128 \
            --load-model $loadmodel
    done
done

echo 'Generate results table ...'
python generate-table.py $dataset'/'$model'50/'
