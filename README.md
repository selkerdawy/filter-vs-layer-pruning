# To filter prune, or to layer prune, that is the question 

## Setup requirements
```bash
# Virtual environment creation
virtualenv .envpy36 -p python3.6
source .envpy36/bin/activate
#Install libraries
pip install -r req.txt
```

## Run ResNet56 on CIFAR100
```bash
# One-shot pass for criteria collection
sh one_shot.sh
# Train filter and block pruned models, this will generate CIFAR100 folder with ResultsTable.html inisde with accuracy compariosn (table 2 in the paper)
sh one-shot-finetune.sh
```

## Cite
If you find this code useful in your research, please consider citing:
```
@InProceedings{Elkerdawy_2020_ACCV,
    author    = {Elkerdawy, Sara and Elhoushi, Mostafa and Singh, Abhineet and Zhang, Hong and Ray, Nilanjan},
    title     = {To Filter Prune, or to Layer Prune, That Is The Question},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {November},
    year      = {2020}
}
```

## Resources

Code is based on Taylor pruning
https://github.com/NVlabs/Taylor_pruning

Layer pruning by imprinting 
https://github.com/selkerdawy/one-shot-layer-pruning


