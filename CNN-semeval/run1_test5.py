#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
from loss.pskd_loss import Custom_CrossEntropy_PSKD

from config import Config
from utils import WordEmbeddingLoader, RelationLoader, SemEvalDataLoader
from model import CNN
from evaluate import Eval
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from scipy import interp
from itertools import cycle


def print_result(predict_label, id2rel, start_idx=8001):
    with open('predicted_result.txt', 'w', encoding='utf-8') as fw:
        for i in range(0, predict_label.shape[0]):
            fw.write('{}\t{}\n'.format(
                start_idx+i, id2rel[int(predict_label[i])]))


def train(model, criterion, loader, config):
    train_loader, dev_loader, _ = loader
    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           weight_decay=config.L2_decay)

    print(model)
    print('traning model parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    # 获得模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    print('--------------------------------------')
    print('start to train the model ...')
    print('开始时间/s', time.time())  

    eval_tool = Eval(config)
    max_f1 = -float('inf')
    logits_save = dict()
    for epoch in range(0, config.epoch):
        for step, (data, label, bert_logits) in enumerate(train_loader):
            model.train()
            data = data.to(config.device)
            label = label.to(config.device)
            targets_numpy = label.cpu().detach().numpy()
            identity_matrix = torch.eye(19)
            targets_one_hot = identity_matrix[targets_numpy]

            optimizer.zero_grad()
            logits = model(data)
            T = 4
            stu_logits = logits/T
            if epoch == 0:
                tea_softmax = targets_one_hot
            else:
                tea_logits = bert_logits/T
                tea_softmax = nn.functional.softmax(tea_logits, dim=1)
            similarity = torch.cosine_similarity(targets_one_hot, tea_softmax).to(config.device)
            lamda_weight = 0.5
            similarity = torch.mean(similarity)
            lamda_weight = similarity * lamda_weight
            new_softmax = tea_softmax.cuda()
            #交叉熵损失函数
            criterion_CE_pskd = Custom_CrossEntropy_PSKD().cuda()
            soft_loss = criterion_CE_pskd(stu_logits, new_softmax)
            #硬损失
            hard_loss = criterion(logits, label)
            loss = (1-lamda_weight) * hard_loss + lamda_weight * soft_loss * T * T
            loss.backward()
            optimizer.step()
            logits_save[step] = logits.cpu().detach()  
        _, train_loss, _, _, _ = eval_tool.evaluate(model, criterion, train_loader)
        f1, dev_loss, _, p, r = eval_tool.evaluate(model, criterion, dev_loader)

        print('[%03d] train_loss: %.3f | dev_loss: %.3f | micro f1 on dev: %.4f | pre on dev: %.4f | recall on dev: %.4f'
              % (epoch, train_loss, dev_loss, f1, p, r), end=' ')
        if f1 > max_f1:
            max_f1 = f1
            torch.save(model.state_dict(), os.path.join(
                config.model_dir, 'model.pkl'))
            print('>>> save models!')
        else:
            print()


def test(model, criterion, loader, config):
    print('--------------------------------------')
    print('start test ...')
    print('测试开始时间/s', time.time()) 

    _, _, test_loader = loader
    model.load_state_dict(torch.load(
        os.path.join(config.model_dir, 'model.pkl')))
    eval_tool = Eval(config)
    f1, test_loss, predict_label, p, r = eval_tool.evaluate(
        model, criterion, test_loader)
    # roc(test_loader)
    print('test_loss: %.3f | macro f1 on test:  %.4f | p: %.4f | r: %.4f' % (test_loss, f1, p, r))
    return predict_label

def roc(test_loader):
    num_class = 19
    score_list = []     # 存储预测得分
    label_list = []     # 存储真实标签
    for i, (inputs, labels, bert_logits) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
 
        outputs = model(inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
        score_tmp = outputs  # (batchsize, nclass)
 
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())
 
    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
 
    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])
 
    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
 
    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
    print(roc_auc_dict)
 
    # 绘制所有类别平均的roc曲线
    plt.figure()
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=2)
 
    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=2)
 
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                       ''.format(i, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('wl.jpg')
    plt.show()


if __name__ == '__main__':
    config = Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    loader = SemEvalDataLoader(rel2id, word2id, config)

    train_loader, dev_loader = None, None
    if config.mode == 1:  # train mode
        train_loader = loader.get_train()
        dev_loader = loader.get_dev()
    test_loader = loader.get_test()
    loader = [train_loader, dev_loader, test_loader]
    print('finish!')

    print('--------------------------------------')
    model = CNN(word_vec=word_vec, class_num=class_num, config=config)
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()

    if config.mode == 1:  # train mode
        train(model, criterion, loader, config)
    predict_label = test(model, criterion, loader, config)
    print_result(predict_label, id2rel)
    print('结束时间/s', time.time()) 
