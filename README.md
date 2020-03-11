# To filter prune, or to layer prune, that is the question 

Code is based on Taylor pruning
https://github.com/NVlabs/Taylor_pruning


## Setup requirements
```bash
# Virtual environment creation
virtualenv .envpy36 -p python3.6
source .envpy36/bin/activate
#Install libraries
pip install -r req.txt
#Download pretrained
mkdir -p cifar100-baseline/cifar100-vgg19-best
```

## Run ResNet56 on CIFAR100
```bash
# One-shot pass for criteria collection
sh one_shot.sh
# Train filter and block pruned models, this will generate CIFAR100 folder with ResultsTable.html inisde with accuracy compariosn (table 2 in the paper)
sh one-shot-finetune.sh
```
