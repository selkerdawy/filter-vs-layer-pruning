#!/bin/bash
nvidia-smi

latencyratio=-1
wd=1e-4
optim=sgd
lr=0.001
lrdecayeach=10
ratio=1.0 #Ratio of used dataset
dataset=CIFAR100

vgg(){
model=vgg19_bn
}

resnet56(){
model=resnet50
}

#vgg
resnet56

pruningconfig=./configs/cifar_one_shot.json
loadmodel=$dataset'-baseline/'$model'/model_best.pth.tar'
for method in 0 2 6 22 30
do
    dir=$dataset'/'$model'/one_shot_criterion'$method

    echo "Checkpoint director: " $dir
    python main.py --name=$dir --dataset=$dataset \
        --lr=$lr --lr-decay-every=$lrdecayeach --momentum=0.9 --epochs=1 --batch-size=128 \
        --pruning=True --seed=0 --model=$model \
        --mgpu=True --group_wd_coeff=1e-8 --wd=$wd --tensorboard=True --pruning-method=$method \
        --data=${datasetdir} --no_grad_clip=True --pruning_config=$pruningconfig \
        --only-estimate-latency=True --optimizer $optim \
        --data=${datasetdir} --optimizer $optim --ratio $ratio --prune-latency-ratio $latencyratio \
        --load_model $loadmodel --mgpu=False

done

echo "Plot per block importance ..."
python plot_layer_importance.py $model $dataset

