'''
该脚本是用于训练 Transformer 模型的主要训练流程脚本。
包括模型训练、验证、损失计算、数据加载和参数设置等。
'''

import argparse  # 解析命令行参数
import math  # 数学运算库
import time  # 时间处理库
import dill as pickle  # dill 是 pickle 的扩展，可用于保存复杂对象
from tqdm import tqdm  # 进度条库
import numpy as np  # 数值计算库
import random  # 随机数库
import os  # 文件操作

# PyTorch 相关库
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator  # torchtext 的数据处理
from torchtext.datasets import TranslationDataset  # 用于翻译任务的数据集

# 导入自定义模块
import transformer.Constants as Constants
from transformer.Models import Transformer  # Transformer 模型定义
from transformer.Optim import ScheduledOptim  # 自定义优化器带 warmup 调度

__author__ = "Yu-Hsiang Huang"

# ------------------------- 性能评估函数 -------------------------
def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' 计算预测结果的损失和准确率 '''
    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]  # 取每个位置最大值对应的类别
    gold = gold.contiguous().view(-1)  # 展平目标序列
    non_pad_mask = gold.ne(trg_pad_idx)  # 非 padding 的 mask
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()  # 正确预测数
    n_word = non_pad_mask.sum().item()  # 有效词数

    return loss, n_correct, n_word

# ------------------------- 损失函数 -------------------------
# ------------------------- 损失函数 -------------------------
def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' 计算交叉熵损失，可选 label smoothing '''
    
    # 将目标序列 `gold` 展平为一维
    gold = gold.contiguous().view(-1)

    if smoothing:
        # 设置 label smoothing 的平滑系数
        eps = 0.1
        # 获取类别数（预测值的维度）
        n_class = pred.size(1)
        
        # 创建 one-hot 编码矩阵
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        
        # 在 one-hot 编码中应用 label smoothing
        # 将真实标签的值设为 (1 - eps)，其他类别的值设为 eps / (n_class - 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        
        # 计算预测值的对数概率
        log_prb = F.log_softmax(pred, dim=1)

        # 创建一个 mask 用于过滤 padding 部分
        non_pad_mask = gold.ne(trg_pad_idx)
        
        # 计算损失：真实标签与预测标签的对数概率的加权和
        loss = -(one_hot * log_prb).sum(dim=1)
        
        # 应用 mask 过滤掉 padding 部分
        loss = loss.masked_select(non_pad_mask).sum()
    else:
        # 如果不使用 label smoothing，直接计算标准的交叉熵损失
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    
    return loss

# ------------------------- 数据预处理（训练阶段） -------------------------
def patch_src(src, pad_idx):
    return src.transpose(0, 1)  # 转换维度：[batch, len] → [len, batch]

def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    return trg[:, :-1], trg[:, 1:].contiguous().view(-1)  # 输入序列和目标序列

# ------------------------- 训练一个 epoch -------------------------
def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        loss, n_correct, n_word = cal_performance(pred, gold, opt.trg_pad_idx, smoothing=smoothing)
        loss.backward()
        optimizer.step_and_update_lr()

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    return total_loss/n_word_total, n_word_correct/n_word_total

# ------------------------- 验证阶段 -------------------------
def eval_epoch(model, validation_data, device, opt):
    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(pred, gold, opt.trg_pad_idx, smoothing=False)

            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    return total_loss/n_word_total, n_word_correct/n_word_total

# ------------------------- 总训练函数 -------------------------
def train(model, training_data, validation_data, optimizer, device, opt):
    if opt.use_tb:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    def print_performances(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=ppl,
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        train_ppl = math.exp(min(train_loss, 100))
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, start, lr)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        if opt.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)

# ------------------------- 主函数入口 -------------------------
def main():
    parser = argparse.ArgumentParser()

    # 常规参数
    parser.add_argument('-data_pkl', default=None)
    parser.add_argument('-train_path', default=None)
    parser.add_argument('-val_path', default=None)
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    # Transformer 模型结构参数
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)

    # 优化器参数
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)
    parser.add_argument('-dropout', type=float, default=0.1)

    # 嵌入共享参数
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    # 保存设置
    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    device = torch.device('cuda' if opt.cuda else 'cpu')

    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)
    else:
        raise

    transformer = Transformer(...).to(device)  # 实例化模型
    optimizer = ScheduledOptim(...)  # 设置优化器

    train(transformer, training_data, validation_data, optimizer, device, opt)

# ------------------------- 数据加载器 -------------------------
def prepare_dataloaders_from_bpe_files(opt, device):
    # 加载 BPE 文件 + vocab
    # 返回训练和验证的 BucketIterator
    ...

def prepare_dataloaders(opt, device):
    # 加载 pickled 数据文件
    ...

if __name__ == '__main__':
    main()
