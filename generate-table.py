import sys
import os
import pickle
import torch
import pdb


def get_css():
    css = '<style>  table {   border-spacing: 0; } tr.border_top td { border-top:1pt solid black;} tr.border_bottom td { border-bottom:1pt solid black;} td {padding:20px;} </style>'
    return css



if __name__ == "__main__":
    root = sys.argv[1]
    dirs = list(os.walk(root))[0][1]
    mapping = {0: 'Weights Taylor', 2: 'Weights norm', 6:'Feature maps', 22:'Gate Taylor', 30: 'BN scale'}
    strTable = "<html>"+get_css()+"<table><tr class='border_bottom'><th>Method</th><th>Accuracy</th></tr>"

    for d in dirs:
        subdir = os.path.join(*[root, d])
        method = int(subdir.split('criterion')[-1])
        if method not in mapping:
            continue
        name = mapping[method]

        #Filter pruning
        filterprune = [os.path.join(*[subdir,f,'models']) for f in list(os.walk(subdir))[0][1] if 'filterpruning' in f][-1]
        modelpath = os.path.join(filterprune,'best_model.weights')
        best_prec = torch.load(modelpath)['best_prec1']
        strRW = "<tr><td>"+name+"_filter pruning </td><td>"+('%.2f %%'%best_prec)+"</td></tr>"
        strTable = strTable+strRW

        #Block pruning
        finetune = sorted([os.path.join(subdir,f) for f in list(os.walk(subdir))[0][1] if 'finetune' in f])
        for i,f in enumerate(finetune):
            modelpath = os.path.join(f,'best_model.pth.tar')
            best_prec = torch.load(modelpath)['best_prec1'].item()*100
            nlayer = f.split('finetune-')[-1]
            if i == len(finetune)-1:
                strRW = "<tr class='border_bottom'><td>"+name+'_Block pruning<sub>'+nlayer+ "</sub></td><td>"+('%.2f %%'%best_prec)+"</td></tr>"
            else:
                strRW = "<tr><td>"+name+'_Block pruning<sub>'+nlayer+ "</sub></td><td>"+('%.2f %%'%best_prec)+"</td></tr>"
            strTable = strTable+strRW

    strTable = strTable+"</table></html>"

    hs = open(os.path.join(root,"ResultsTable.html"), 'w')
    hs.write(strTable)

