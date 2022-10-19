import torch 
from torch import nn 
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def F1score(output, target, num_labels):
    
    with torch.no_grad():
        output = output.to('cpu')
        output = torch.argmax(output, dim=1).numpy()

        target = target.to('cpu').numpy()

    return f1_score(target, output, average = 'macro')

def Accuray(output, target, num_labels):
    
    with torch.no_grad():
        output = output.to('cpu')
        output = torch.argmax(output, dim=1).numpy()

        target = target.to('cpu').numpy()

    return sum(output == target)/len(target)

def Roc_auc_ovo(output, target, num_labels):
    SFfunc = nn.Softmax(dim=1)
    with torch.no_grad():
        output = SFfunc(output).to('cpu').numpy()
        target = target.to('cpu').numpy()
    
    if num_labels == 2:
        output = output[:,1]
        return roc_auc_score(target, output)
    
    return roc_auc_score(target, output, multi_class = 'ovo')        

    

def Roc_auc_ovr(output, target, num_labels):
    SFfunc = nn.Softmax(dim=1)
    with torch.no_grad():
        output = SFfunc(output).to('cpu').numpy()
        target = target.to('cpu').numpy()
    
    if num_labels == 2:
        output = output[:,1]
        return roc_auc_score(target, output)
    
    return roc_auc_score(target, output, multi_class = 'ovr')

    
if __name__ == '__main__':
    import torch
    output = torch.tensor([[23,0.4,0.4],[0.7,0.2,0.1],[0.2,0.5,0.3],
                           [0.4,0.1,0.5],[0.0,0.9,0.1],[0.1,0.1,0.8]])
    # output = torch.argmax(output, dim=1)
    target = torch.tensor([1,0,2,0,1,2])
    F1score(output, target)
    Accuray(output, target)

    Roc_auc_ovo(output, target)
    Roc_auc_ovr(output, target)

