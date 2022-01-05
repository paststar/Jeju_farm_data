import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from itertools import cycle
#from utils import save_model
#from utils import visualize
#from utils import set_model_mode,get_pseudo_label
import time

def train_transformer(args,transformer, train_loader1, train_loader2, optimizer, loss ,epoch, _print, writer):
    start = time.time()

    classifier_criterion = loss[0]

    transformer.train()
    len_dataloader = min(len(train_loader1), len(train_loader2))

    correct1 = 0
    correct2 = 0

    total_loss1 = 0
    total_loss2 = 0
    total_num1 = 0
    total_num2 = 0
    
    for batch_idx, (data1, data2) in enumerate(zip(train_loader1, cycle(train_loader2))):

        image1, label1 = data1
        image2, label2 = data2

        image1, label1 = image1.cuda(), label1.cuda()
        image2, label2 = image2.cuda(), label2.cuda()

        pred1 = transformer(image1)
        pred2 = transformer(image2)

        loss1 = classifier_criterion(pred1, label1)
        loss2 = classifier_criterion(pred2, label2)
       
        total_loss = loss1 + loss2

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss1 += loss1
        total_loss2 += loss2
        #total_domain_loss += domain_loss

        ### train accuracy ###
        pred = pred1.argmax(dim=1, keepdim=True)
        correct1 += pred.eq(label1.long().view_as(pred)).sum().item()
        total_num1 += label1.size(0)

        pred = pred2.argmax(dim=1, keepdim=True)
        correct2 += pred.eq(label2.long().view_as(pred)).sum().item()
        total_num2 += label2.size(0)

    _print("(train) loss1 : {:.4f} loss2 : {:.4f}".format(
    total_loss1 / len_dataloader, total_loss2 / len_dataloader))
    _print("(train) accuracy1 : {:.2f}% accuracy2 : {:.2f}% train time: {:.2f}s".format(
    (correct1 / total_num1) * 100,(correct2 / total_num2) * 100,time.time() - start))

    writer.add_scalar('train/acc/acc1', (correct1 / total_num1) * 100,epoch)
    writer.add_scalar('train/acc/acc2', (correct2 / total_num2) * 100,epoch)

    writer.add_scalar('train/loss/loss1', total_loss1 / len_dataloader, epoch)
    writer.add_scalar('train/loss/loss2', total_loss2 / len_dataloader, epoch)
