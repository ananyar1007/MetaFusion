from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import torch
import torch.nn.functional as F
import numpy as np

def calc_metric(model_predictions, all_labels):
    balanced_acc = metrics.balanced_accuracy_score(all_labels, model_predictions)
    print("Balanced ACC", balanced_acc)
    conf = metrics.confusion_matrix(all_labels, model_predictions)
    print("Confusion matrix", conf)
    acc = metrics.accuracy_score(all_labels, model_predictions)
    print("Acc", acc)
    return balanced_acc, acc


def write_results(exp_dir,loss_list, ba_list, acc_list, loss_val_list, ba_val_list, acc_val_list, loss_test_list, ba_test_list, acc_test_list):
    d={}
    d['trn_loss'] = loss_list
    d['trn_ba'] = ba_list
    d['trn_acc'] = acc_list
    d['val_loss'] = loss_val_list
    d['val_ba'] = ba_val_list
    d['val_acc'] = acc_val_list
    d['test_loss'] = loss_test_list
    d['test_ba'] = ba_test_list
    d['test_acc'] = acc_test_list


    print(loss_list)
    keys = list(sorted(d.keys()))
    print(keys)
    with open(exp_dir+"metrics_logs.csv", "w") as outfile:
       writer = csv.writer(outfile, delimiter = ",")
       writer.writerow(keys)
       writer.writerows(zip(*[d[key] for key in keys]))
        
    plt.plot(loss_list)
    plt.title('Loss:Trn')
    plt.grid()
    
    plt.savefig(exp_dir+ "Trn_loss.png")
    plt.show()
    plt.plot(ba_list)
    plt.plot(acc_list)
    plt.title('BA and ACC')
    plt.grid()
    plt.legend(['BA','ACC'])
    plt.savefig(exp_dir + "Trn_ba_acc.png")
    plt.show()
    
    plt.plot(loss_test_list)
    plt.title('Loss:test')
    plt.grid()
    plt.savefig(exp_dir+ "Test_loss.png")
    
    plt.show()
    
    plt.plot(ba_test_list)
    plt.plot(acc_test_list)
    plt.title('BA and ACC:Test')
    plt.legend(['BA','ACC'])
    
    plt.grid()
    plt.savefig(exp_dir + "Test_ba_acc.png")
    plt.show()
    
    plt.plot(ba_val_list)
    plt.plot(acc_val_list)
    plt.title('BA and ACC:Val')
    plt.legend(['BA','ACC'])
    
    plt.grid()
    plt.savefig(exp_dir + "Val_ba_acc.png")
    
    plt.show()
    plt.plot(loss_val_list)
    plt.title('Loss:Val')
    plt.grid()
    plt.savefig(exp_dir+ "Val_loss.png")
    
    plt.show()

def list_to_cuda(x):
    for i in range(len(x)):
        x[i] = x[i].cuda()
    return x


def eval_model(model, test_loader):
        all_scores=[]
        all_labels=[]
  

        all_loss=[]
   
        for idx, v  in enumerate(tqdm(test_loader)):
            (data, (metadata),labels) = v
            with torch.no_grad():
                if idx < 40000:
                    metadata = list_to_cuda(metadata)
                    logits = model(data.cuda(),metadata)
                    loss =  F.cross_entropy(logits, labels.cuda())
                    scores = torch.argmax(logits,dim=1).cpu().numpy()             
                    all_scores.extend(scores)
                    all_labels.extend(labels.detach().cpu())
                    all_loss.append(loss.cpu().detach().numpy())   
   
                else:
                    break

        
        print('Loss', np.mean(all_loss))
        ba, acc = calc_metric(all_scores, all_labels)
        return ba, acc,np.mean(all_loss)    


def train_one_epoch(model, optimizer, scheduler, train_loader):
    trn_loss_all =[]
    all_scores = []
    all_labels =[]
    for idx, v  in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
            # Runs the forward pass with autocasting.     
        (data, metadata, labels) = v
        metadata = list_to_cuda(metadata)
        logits = model(data.cuda(),metadata)
        loss =  F.cross_entropy(logits, labels.cuda())
        trn_loss_all.append(loss.cpu().detach().numpy())
        scores = torch.argmax(logits,dim=1).cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend(labels.detach().cpu())

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        loss.backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
        optimizer.step()
    mean_loss =  np.mean(trn_loss_all)
    scheduler.step() #torch.tensor(mean_loss))
    print("LR", scheduler.get_last_lr())
    print("Trn Loss",mean_loss)
    ba, acc = calc_metric(all_scores, all_labels)
    return mean_loss, ba, acc
                   