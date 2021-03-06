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
            identity_matrix = torch.eye(80)
            targets_one_hot = identity_matrix[targets_numpy]

            optimizer.zero_grad()
            logits = model(data)
            T = 1
            stu_logits = logits/T
            if epoch == 0:
                tea_softmax = targets_one_hot
            else:
                tea_logits = bert_logits/T
                tea_softmax = nn.functional.softmax(tea_logits, dim=1)
            similarity = torch.cosine_similarity(targets_one_hot, tea_softmax).to(config.device)
            lamda_weight = 0.8
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
