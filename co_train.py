"""Training script"""

import os
import time
import copy
import shutil
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from data import get_loader, get_dataset
from model import SGRAF
from vse import VSE, infonce
from vocab import Vocabulary, deserialize_vocab
from evaluation import i2t, t2i, encode_data, shard_attn_scores
import logging
from torch.nn.utils.clip_grad import clip_grad_norm_

# 创建 logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 设置全局日志级别

from utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    adjust_learning_rate,
)


class GraphRefiner(nn.Module):
    def __init__(self, dim=1024, k=5, n_heads=4, dropout=0.1):
        super(GraphRefiner, self).__init__()
        self.k = k
        self.n_heads = n_heads
        self.d_k = dim // n_heads

        # 线性变换，映射到一个适合计算关系的映射空间
        # self.embedding_layer = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        # 注意力机制：用于计算节点间的相互权重
        # self.attn_fc = nn.Linear(2 * dim, 1)
        # 2. 边权重推理：不仅计算节点相似度，还学习节点间的相对关系
        # 我们构建一个边特征生成器
        self.edge_nn = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 3. 输出层与残差连接
        self.output_layer = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, neighbors):
        """
        neighbors: shape [batch_size, k, dim] - 检索到的K个邻居
        """
        bs, k, dim = neighbors.size()
        
        # --- 多头图注意力机制 (Multi-head Graph Attention) ---
        # 计算 Q, K, V
        q = self.q_proj(neighbors).view(bs, k, self.n_heads, self.d_k).transpose(1, 2)
        k_val = self.k_proj(neighbors).view(bs, k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(neighbors).view(bs, k, self.n_heads, self.d_k).transpose(1, 2)
        
        # 缩放点积注意力 [bs, heads, k+1, k+1]
        attn_scores = torch.matmul(q, k_val.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(bs, k, dim)
        neighbors = self.layer_norm(neighbors + self.dropout(self.output_layer(context)))
        refined_feat = neighbors.mean(dim=1)
        return F.normalize(refined_feat, p=2, dim=-1)


class MemoryQueue:
    """
    维护一个干净样本特征的队列，用于 Unlabeled 数据的矫正。
    风格参考 simple_l2rm.txt 的 imgs_queue
    """
    def __init__(self, queue_size=2048, dim=1024):
        self.queue_size = queue_size
        self.dim = dim
        self.ptr = 0
        self.is_full = False
        
        # 存储 [特征, 及其对应的文本特征]
        # 注意：这里存的是 Tensor (在 GPU 上)
        self.img_queue = torch.zeros(queue_size, dim).cuda()
        self.txt_queue = torch.zeros(queue_size, dim).cuda()

    def enqueue(self, img_feats, txt_feats):
        """将 Labeled (Clean) 数据的特征存入队列"""
        batch_size = img_feats.shape[0]
        
        # 指针循环更新
        if self.ptr + batch_size <= self.queue_size:
            self.img_queue[self.ptr : self.ptr + batch_size] = img_feats.detach()
            self.txt_queue[self.ptr : self.ptr + batch_size] = txt_feats.detach()
            self.ptr += batch_size
        else:
            # 处理溢出，覆盖开头
            tail = self.queue_size - self.ptr
            self.img_queue[self.ptr :] = img_feats[:tail].detach()
            self.txt_queue[self.ptr :] = txt_feats[:tail].detach()
            
            rem = batch_size - tail
            self.img_queue[:rem] = img_feats[tail:].detach()
            self.txt_queue[:rem] = txt_feats[tail:].detach()
            self.ptr = rem
            self.is_full = True

    def retrieve_nearest_text(self, query_img_feats, k=1, refiner_node=None, img_refiner_node=None):
        """
        根据 query_img (Unlabeled) 找队列中最相似的 Clean Image，
        并返回该 Clean Image 对应的 Text Feature。
        """
        current_size = self.queue_size if self.is_full else self.ptr
        if current_size < k:
            return None

        valid_img_bank = self.img_queue[:current_size]
        valid_txt_bank = self.txt_queue[:current_size]

        # [Batch, Dim] x [Queue, Dim]^T -> [Batch, Queue]
        sims = torch.mm(query_img_feats, valid_img_bank.t())

        # values, indices: [Batch, k]
        _, indices = torch.topk(sims, k, dim=1)

        # [Batch, k] -> [Batch, k, Dim]
        neighbor_txt_feats = valid_txt_bank[indices] 
        
        if refiner_node is not None:
            # 使用 GNN 增强特征
            target_txt_feats = refiner_node(neighbor_txt_feats)
            # target_img_feats = img_refiner_node(valid_img_bank[indices]) 
        else:
            # [Batch, Dim]
            target_txt_feats = neighbor_txt_feats.mean(dim=1)
            target_txt_feats = torch.nn.functional.normalize(target_txt_feats, dim=-1)
        
        return target_txt_feats

    def retrieve_nearest_img(self, query_text_feats, k=1, refiner_node=None, txt_refiner_node=None):
        """
        根据 query_text (Unlabeled) 找队列中最相似的 Clean Text，
        并返回该 Clean Text 对应的 Image Feature。
        """
        current_size = self.queue_size if self.is_full else self.ptr
        if current_size < k:
            return None

        valid_img_bank = self.img_queue[:current_size]
        valid_txt_bank = self.txt_queue[:current_size]

        # [Batch, Dim] x [Queue, Dim]^T -> [Batch, Queue]
        sims = torch.mm(query_text_feats, valid_txt_bank.t())

        # values, indices: [Batch, k]
        _, indices = torch.topk(sims, k, dim=1)

        # [Batch, k] -> [Batch, k, Dim]
        neighbor_img_feats = valid_img_bank[indices] 
        

        if refiner_node is not None:
            # 使用 GNN 增强特征
            target_img_feats = refiner_node(neighbor_img_feats) 
            # target_txt_feats = txt_refiner_node(valid_txt_bank[indices])
        else:
            # [Batch, Dim]
            target_img_feats = neighbor_img_feats.mean(dim=1)
            
            target_img_feats = torch.nn.functional.normalize(target_img_feats, dim=-1)
        
        return target_img_feats


def main(opt):

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建文件处理器
    file_handler = logging.FileHandler(f'{opt.output_dir}/log.txt', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # load Vocabulary Wrapper
    logger.info("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    vocab.add_word('<mask>')  # add the mask, for testing cloze

    word2idx = vocab.word2idx

    logger.info('Add <mask> token into the vocab')
    opt.vocab_size = len(vocab)

    # load dataset
    captions_train, images_train = get_dataset(
        opt.data_path, opt.data_name, "train"
    )
    captions_dev, images_dev = get_dataset(opt.data_path, opt.data_name, "dev")

    # data loader
    noisy_trainloader, data_size, clean_labels = get_loader(
        captions_train,
        images_train,
        vocab,
        "warmup",
        opt.batch_size,
        opt.workers,
        opt.noise_ratio,
        opt.noise_file,
    )
    val_loader = get_loader(
        captions_dev, images_dev, vocab, "dev", opt.batch_size, opt.workers
    )

    # create models
    model_A = VSE(opt, word2idx)
    model_B = VSE(opt, word2idx)

    state_dict = model_A.state_dict()
    model_B.load_state_dict(state_dict)

    
    best_rsum = 0
    start_epoch = 0

    # save the history of losses from two networks
    all_loss = [[], []]

    # Warmup
    logger.info("\n* Warmup")
    if opt.warmup_model_path:
        if os.path.isfile(opt.warmup_model_path):
            checkpoint = torch.load(opt.warmup_model_path)
            model_A.load_state_dict(checkpoint["model_A"])
            model_B.load_state_dict(checkpoint["model_B"])
            logger.info(
                "=> load warmup checkpoint '{}' (epoch {})".format(
                    opt.warmup_model_path, checkpoint["epoch"]
                )
            )
            logger.info("\nValidattion ...")
            validate(opt, val_loader, [model_A, model_B])
        else:
            raise Exception(
                "=> no checkpoint found at '{}'".format(opt.warmup_model_path)
            )
    else:
        epoch = 0
        for epoch in range(0, opt.warmup_epoch):
            logger.info("[{}/{}] Warmup model_A".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, noisy_trainloader, model_A, epoch)
            logger.info("[{}/{}] Warmup model_B".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, noisy_trainloader, model_B, epoch)

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "opt": opt,
                },
                is_best=False,
                filename="warmup_model_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )

        # evaluate on validation set
            logger.info("\nValidattion ...")
            validate(opt, val_loader, [model_A, model_B])

    # save the history of losses from two networks
    all_loss = [[], []]
    logger.info("\n* Co-training")

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info("\nEpoch [{}/{}]".format(epoch, opt.num_epochs))
        adjust_learning_rate(opt, model_A.optimizer, epoch)
        adjust_learning_rate(opt, model_B.optimizer, epoch)

        # # Dataset split (labeled, unlabeled)
        logger.info("Split dataset ...")
        prob_A, prob_B, all_loss = eval_train(
            opt,
            model_A,
            model_B,
            noisy_trainloader,
            data_size,
            all_loss,
            clean_labels,
            epoch,
        )

        pred_A = split_prob(prob_A, opt.p_threshold)
        pred_B = split_prob(prob_B, opt.p_threshold)

        clean_probs_A = prob_A[pred_A]

        if len(clean_probs_A) > 0:
            avg_clean_prob_A = clean_probs_A.mean()
        else:
            avg_clean_prob_A = 0.0

        num_above_avg = (prob_A > avg_clean_prob_A).sum()
        ratio_above_avg = (num_above_avg / len(prob_A)) * 100

        logger.info(f"Average probability of Predicted Clean samples: {avg_clean_prob_A:.4f}")
        logger.info(f"Number of samples with prob > avg: {num_above_avg} ({ratio_above_avg:.2f}%)")

        model_A.avg_clean_prob = avg_clean_prob_A

        clean_probs_B = prob_B[pred_B]

        if len(clean_probs_B) > 0:
            avg_clean_prob_B = clean_probs_B.mean()
        else:
            avg_clean_prob_B = 0.0

        num_above_avg = (prob_B > avg_clean_prob_B).sum()
        ratio_above_avg = (num_above_avg / len(prob_B)) * 100

        logger.info(f"Average probability of Predicted Clean samples: {avg_clean_prob_B:.4f}")
        logger.info(f"Number of samples with prob > avg: {num_above_avg} ({ratio_above_avg:.2f}%)")

        model_B.avg_clean_prob = avg_clean_prob_B

        logger.info("\nModel A training ...")
        # train model_A
        labeled_trainloader, unlabeled_trainloader = get_loader(
            captions_train,
            images_train,
            vocab,
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            pred=pred_B,
            prob=prob_B,
        )
        train(opt, model_A, model_B, labeled_trainloader, unlabeled_trainloader, epoch)

        logger.info("\nModel B training ...")
        # train model_B
        labeled_trainloader, unlabeled_trainloader = get_loader(
            captions_train,
            images_train,
            vocab, 
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            pred=pred_A,
            prob=prob_A,
        )
        train(opt, model_B, model_A, labeled_trainloader, unlabeled_trainloader, epoch)

        logger.info("\nValidattion ...")
        # evaluate on validation set
        rsum = validate(opt, val_loader, [model_A, model_B])

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_A": model_A.state_dict(),
                "model_B": model_B.state_dict(),
                "best_rsum": best_rsum,
                "opt": opt,
            },
            is_best,
            filename="checkpoint_{}.pth.tar".format(epoch),
            prefix=opt.output_dir + "/",
        )


def train(opt, net, net2, labeled_trainloader, unlabeled_trainloader=None, epoch=None):
    """
    One epoch training.
    """
    losses_l = AverageMeter("Loss L", ":.4e")  
    losses_u = AverageMeter("Loss U", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    
    max_len = max(len(labeled_trainloader), 0)

    progress = ProgressMeter(
        # len(labeled_trainloader),
        max_len, 
        [batch_time, data_time, losses_l, losses_u],
        prefix=f"Epoch[{epoch}/{opt.num_epochs}] Training Step",
    )

    if not hasattr(net, 'memory_queue'):
        net.memory_queue = MemoryQueue(queue_size=opt.queue_size, dim=opt.embed_size)
    
    if not hasattr(net, 'i2t_refiner_node'):
        net.i2t_refiner_node = GraphRefiner(dim=opt.embed_size, k=opt.k).cuda()
        net.i2t_refiner_node.train()
        net.optimizer.add_param_group({
            "params": net.i2t_refiner_node.parameters(),
            "lr": opt.learning_rate   # 或者一个更小的 lr
        })
    if not hasattr(net, 't2i_refiner_node'):
        net.t2i_refiner_node = GraphRefiner(dim=opt.embed_size, k=opt.k).cuda()
        net.t2i_refiner_node.train()
        net.optimizer.add_param_group({
            "params": net.t2i_refiner_node.parameters(),
            "lr": opt.learning_rate   # 或者一个更小的 lr
        })

    if not hasattr(net2, 'memory_queue'):
        net2.memory_queue = MemoryQueue(queue_size=opt.queue_size, dim=opt.embed_size)
        
    if not hasattr(net2, 'i2t_refiner_node'):
        net2.i2t_refiner_node = GraphRefiner(dim=opt.embed_size, k=opt.k).cuda()
        net2.i2t_refiner_node.train()
        net2.optimizer.add_param_group({
            "params": net2.i2t_refiner_node.parameters(),
            "lr": opt.learning_rate    # 或者一个更小的 lr
        })
    if not hasattr(net2, 't2i_refiner_node'):  
        net2.t2i_refiner_node = GraphRefiner(dim=opt.embed_size, k=opt.k).cuda()
        net2.t2i_refiner_node.train()
        net2.optimizer.add_param_group({
            "params": net2.t2i_refiner_node.parameters(),
            "lr": opt.learning_rate    # 或者一个更小的 lr
        })
    criterion = net.criterion
    
    # fix one network and train the other
    net.train_start()
    net2.val_start()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    labeled_train_iter = iter(labeled_trainloader)
    labels_l = []
    pred_labels_l = []
    labels_u = []
    pred_labels_u = []
    end = time.time()
    for i in range(max_len):        
        try: 
            (
                batch_images_l,
                batch_img_len_l, 
                batch_text_l,
                batch_lengths_l,
                _,
                batch_labels_l,
                batch_prob_l,
                batch_clean_labels_l,
            ) = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            (
                batch_images_l,
                batch_img_len_l, 
                batch_text_l,
                batch_lengths_l,
                _,
                batch_labels_l,
                batch_prob_l,
                batch_clean_labels_l,
            ) = next(labeled_train_iter)
        batch_size = batch_images_l.size(0)
        labels_l.append(batch_clean_labels_l)

        # unlabeled data
        try:
            (
                batch_images_u,
                batch_img_len_u,
                batch_text_u,
                batch_lengths_u,
                _,
                batch_clean_labels_u,
            # ) = unlabeled_train_iter.next()
            ) = next(unlabeled_train_iter)

        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (
                batch_images_u,
                batch_img_len_u,
                batch_text_u,
                batch_lengths_u,
                _,
                batch_clean_labels_u,
            # ) = unlabeled_train_iter.next()
            ) = next(unlabeled_train_iter)
        labels_u.append(batch_clean_labels_u)

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            batch_prob_l = batch_prob_l.cuda()
            batch_labels_l = batch_labels_l.cuda()

        net.optimizer.zero_grad()

        img_emb_l, cap_emb_l, cap_len_l = net.forward_emb(batch_images_l, batch_text_l, batch_lengths_l, imgae_lengths=batch_img_len_l)

        # 计算标准 Contrastive Loss
        sims_l = net.forward_sim(img_emb_l, cap_emb_l, cap_len_l)
        loss_1 = criterion(sims_l, labels=None) # Standard loss

        img_emb_l2, cap_emb_l2, cap_len_l2 = net.forward_emb(batch_images_l, batch_text_l, batch_lengths_l, imgae_lengths=batch_img_len_l)
        img_sims = img_emb_l.mm(img_emb_l2.t())
        cap_sims = cap_emb_l.mm(cap_emb_l2.t())
        loss_2 = criterion(img_sims, margin=0.3)
        loss_3 = criterion(cap_sims, margin=0.3)

        loss_l = loss_1 + loss_2 + loss_3

        loss_l.backward()

        high_conf_idx = (batch_prob_l > net.avg_clean_prob).nonzero(as_tuple=True)[0]
        if high_conf_idx.numel() > 0:
            net.memory_queue.enqueue(img_emb_l[high_conf_idx], cap_emb_l[high_conf_idx])

        img_emb_u, cap_emb_u, cap_len_u = net.forward_emb(batch_images_u, batch_text_u, batch_lengths_u, imgae_lengths=batch_img_len_u)

        if opt.i2t:

            corrected_txt_emb = net2.memory_queue.retrieve_nearest_text(img_emb_u, k=opt.k, refiner_node=net.i2t_refiner_node, img_refiner_node=net.t2i_refiner_node)
            # corrected_txt_emb = net2.memory_queue.retrieve_nearest_text(img_emb_u, k=opt.k, refiner_node=None)

            loss_u = torch.tensor(0.0).cuda()
            if corrected_txt_emb is not None:
                sims_u = img_emb_u.mm(corrected_txt_emb.t()) 
                loss_u = infonce(sims_u) 
                
                correction_weight = 0.5
                loss_u = loss_u * correction_weight
                loss_u.backward()
            else:
                pass
            
        if opt.t2i:
            corrected_img_emb = net2.memory_queue.retrieve_nearest_img(cap_emb_u, k=opt.k, refiner_node=net.t2i_refiner_node, txt_refiner_node=net.i2t_refiner_node)
            # corrected_img_emb = net2.memory_queue.retrieve_nearest_img(cap_emb_u, k=opt.k, refiner_node=None)

            loss_u = torch.tensor(0.0).cuda()
            if corrected_img_emb is not None:
                sims_u = cap_emb_u.mm(corrected_img_emb.t()) 
                loss_u = infonce(sims_u) 
                
                correction_weight = 0.5
                loss_u = loss_u * correction_weight
                loss_u.backward()
            else:
                pass
        
        # 梯度裁剪和更新
        if net.grad_clip > 0:
            clip_grad_norm_(net.params, net.grad_clip)
        net.optimizer.step()

        # loss = loss_l + loss_u
        losses_l.update(loss_l.item(), batch_images_l.size(0))

        # 只有在进行了噪声矫正训练时才更新 losses_u
        if 'loss_u' in locals(): 
            losses_u.update(loss_u.item(), batch_images_u.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info log info
        if i % opt.log_step == 0:
            progress.display(i)


def warmup(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses], prefix="Warmup Step"
    )

    end = time.time()
    for i, (images, image_lengths, captions, lengths, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break

        model.train_start()

        # Update the model
        loss = model.train(images, captions, lengths, image_lengths=image_lengths, mode="warmup")
        losses.update(loss, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            progress.display(i)


def validate(opt, val_loader, models=[]):
    # compute the encoding for all the validation images and captions
    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    Eiters = models[0].Eiters
    sims_mean = 0
    count = 0
    for ind in range(len(models)):
        count += 1
        logger.info("Encoding with model {}".format(ind))
        img_embs, cap_embs, cap_lens = encode_data(
            models[ind], val_loader, opt.log_step
        )
        logger.info(f"image shape:{img_embs.shape}, cap shape: {cap_embs.shape}")
        # clear duplicate 5*images and keep 1*images FIXME
        img_embs = np.array(
            [img_embs[i] for i in range(0, len(img_embs), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        logger.info("Computing similarity from model {}".format(ind))
        sim = shard_attn_scores(
            models[ind], img_embs, cap_embs, cap_lens, opt, shard_size=100
        )
        end = time.time()
        logger.info(
            "Calculate similarity time with model {}: {:.2f} s".format(ind, end - start)
        )
        logger.info(
            f"Model_{ind}:"
        )
        # caption retrieval
        (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sim, per_captions)
        logger.info(
            "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
                r1, r5, r10, medr, meanr
            )
        )

        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sim, per_captions)
        logger.info(
            "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
                r1i, r5i, r10i, medri, meanr
            )
        )

        sims_mean += sim
    # average the sims
    sims_mean = sims_mean / count

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions)
    logger.info(
        "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1, r5, r10, medr, meanr
        )
    )

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions)
    logger.info(
        "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1i, r5i, r10i, medri, meanr
        )
    )

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info(f"R_sum: {r_sum}")
    return r_sum


def eval_train(
    opt, model_A, model_B, data_loader, data_size, all_loss, clean_labels, epoch
):
    """
    Compute per-sample loss and prob
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(data_loader), [batch_time, data_time], prefix="Computinng losses"
    )

    model_A.val_start()
    model_B.val_start()
    losses_A = torch.zeros(data_size)
    losses_B = torch.zeros(data_size)

    end = time.time()
    for i, (images, image_lengths, captions, lengths, ids) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        with torch.no_grad():
            # compute the loss
            loss_A = model_A.train(images, captions, lengths, image_lengths=image_lengths, mode="eval_loss")
            loss_B = model_B.train(images, captions, lengths, image_lengths=image_lengths, mode="eval_loss")
            for b in range(images.size(0)):
                losses_A[ids[b]] = loss_A[b]
                losses_B[ids[b]] = loss_B[b]

            batch_time.update(time.time() - end)
            end = time.time()
            if i % opt.log_step == 0:
                progress.display(i)

    losses_A = (losses_A - losses_A.min()) / (losses_A.max() - losses_A.min())
    all_loss[0].append(losses_A)
    losses_B = (losses_B - losses_B.min()) / (losses_B.max() - losses_B.min())
    all_loss[1].append(losses_B)

    input_loss_A = losses_A.reshape(-1, 1)
    input_loss_B = losses_B.reshape(-1, 1)

    logger.info("\nFitting GMM ...")
    # fit a two-component GMM to the loss
    gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]

    gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]

    return prob_A, prob_B, all_loss


def split_prob(prob, threshld):
    print(prob.min(), prob.max(), prob.mean())
    if prob.min() > threshld:
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        logger.info(
            "No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled."
        )
        threshld = np.sort(prob)[len(prob) // 100]

    pred = prob > threshld
    return pred
