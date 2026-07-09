"""SGRAF model"""
import math
from collections import OrderedDict
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()




def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features, pool_weights

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe
        

class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.gpool = GPO(32, 32)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths=None):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        features, pool_weights = self.gpool(features, image_lengths)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """
    Build local word representations by common-used Bi-GRU or GRU.
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """

    def __init__(
        self,
        opt,
        vocab_size,
        word_dim,
        embed_size,
        num_layers,
        use_bi_gru=False,
        no_txtnorm=False,
        word2idx=None,
    ):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.opt = opt
        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.cap_rnn = nn.GRU(
            word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru
        )
        self.gpool = GPO(32, 32)
        self.init_weights(word2idx)

    def init_weights(self, word2idx):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        path = os.path.join(self.opt.data_path, 'vector_cache')
        print(path)
        wemb = GloVe(
            name='6B',
            dim=300,
            cache=path,
            max_vectors=None
        )

        # quick-and-dirty trick to improve word-hit rate
        missing_words = []
        for word, idx in word2idx.items():
            if word not in wemb.stoi:
                word = word.replace('-', '').replace('.', '').replace("'", '')
                if '/' in word:
                    word = word.split('/')[0]
            if word in wemb.stoi:
                self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
            else:
                missing_words.append(word)
        print('Words: {}/{} found in vocabulary; {} words missing'.format(
            len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, captions, lengths):
        """Handles variable size captions"""
        # embed word ids to vectors
        cap_emb = self.embed(captions)
        cap_emb = self.dropout(cap_emb)

        # pack the caption
        packed = pack_padded_sequence(
            cap_emb, lengths, batch_first=True, enforce_sorted=False
        )

        # forward propagate RNN
        out, _ = self.cap_rnn(packed)

        # reshape output to (batch_size, hidden_size)
        cap_emb, cap_len = pad_packed_sequence(out, batch_first=True)

        if self.use_bi_gru:
            cap_emb = (
                cap_emb[:, :, : cap_emb.size(2) // 2]
                + cap_emb[:, :, cap_emb.size(2) // 2 :]
            ) / 2

        pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))


        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(pooled_features, dim=-1)

        return cap_emb



class RCE(nn.Module):
    def __init__(self, tau=0.1):
        super(RCE, self).__init__()
        self.tau = tau

    def forward(self, scores):
        eps = 1e-7
        mask = torch.eye(scores.shape[0])+eps
        mask = mask.cuda()
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True))
        t2i = scores.t() / (scores.t().sum(1, keepdim=True))

        cost_i2t_r = - (mask.log()*i2t).sum(1).mean()
        cost_t2i_r = - (mask.log()*t2i).sum(1).mean()
        cost_i2t = -i2t.diag().log().mean()
        cost_t2i = -t2i.diag().log().mean()

        return 0.5*(cost_i2t_r + cost_t2i_r + cost_i2t + cost_t2i)
        #return cost_i2t_r+cost_t2i_r



class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(
        self,
        scores,
        hard_negative=True,
        labels=None,
        soft_margin="linear",
        mode="train",
    ):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if labels is None:
            margin = self.margin
        elif soft_margin == "linear":
            margin = self.margin * labels
        elif soft_margin == "exponential":
            s = (torch.pow(10, labels) - 1) / 9
            margin = self.margin * s
        elif soft_margin == "sin":
            s = torch.sin(math.pi * labels - math.pi / 2) / 2 + 1 / 2
            margin = self.margin * s

        # compare every diagonal score to scores in its column: caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row: image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        mask = mask.to(cost_s.device)
        cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

        # maximum and mean
        cost_s_max, cost_im_max = cost_s.max(1)[0], cost_im.max(0)[0]
        cost_s_mean, cost_im_mean = cost_s.mean(1), cost_im.mean(0)

        if mode == "predict":
            p = margin - (cost_s_mean + cost_im_mean) / 2
            p = p.clamp(min=0, max=margin)
            idx = torch.argsort(p)
            ratio = scores.size(0) // 10 + 1
            p = p / torch.mean(p[idx[-ratio:]])
            return p

        elif mode == "warmup":
            return cost_s_mean.sum() + cost_im_mean.sum()
        elif mode == "train":
            if hard_negative:
                return cost_s_max.sum() + cost_im_max.sum()
            else:
                return cost_s_mean.sum() + cost_im_mean.sum()

        elif mode == "eval_loss":
            return cost_s_mean + cost_im_mean


class VSE(nn.Module):
    def __init__(self, opt, word2idx):
        super().__init__()
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(
            opt.img_dim, opt.embed_size, no_imgnorm=opt.no_imgnorm
        )
        self.txt_enc = EncoderText(opt,
            opt.vocab_size,
            opt.word_dim,
            opt.embed_size,
            opt.num_layers,
            use_bi_gru=opt.bi_gru,
            no_txtnorm=opt.no_txtnorm,
            word2idx=word2idx
        )

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin)

        self.K = opt.K
        # create the queue
        self.register_buffer("img_queue", torch.randn(self.K, self.opt.embed_size))
        self.img_queue = nn.functional.normalize(self.img_queue, dim=1)
        self.register_buffer("img_queue_ptr", torch.zeros(1, dtype=torch.long))

         # create the queue
        self.register_buffer("txt_queue", torch.randn(self.K, self.opt.embed_size))
        self.txt_queue = nn.functional.normalize(self.txt_queue, dim=1)
        self.register_buffer("txt_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.Eiters = 0

    def _dequeue_and_enqueue(self, img_embed, text_embed):

        batch_size = img_embed.shape[0]

        ptr = int(self.img_queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.img_queue[:, ptr : ptr + batch_size] = img_embed
        self.txt_queue[:, ptr : ptr + batch_size] = text_embed

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.img_queue_ptr[0] = ptr


    def forward_emb(self, images, captions, lengths, imgae_lengths=None):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            imgae_lengths = imgae_lengths.cuda()
        # Forward feature encoding
        img_embs = self.img_enc(images, imgae_lengths)
        cap_embs = self.txt_enc(captions, lengths)
        return img_embs, cap_embs, lengths

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sims = img_embs.mm(cap_embs.t())
        return sims

    def train(
        self,
        images,
        captions,
        lengths,
        image_lengths=None,
        hard_negative=True,
        labels=None,
        soft_margin=None,
        mode="train",
    ):
        """One epoch training.
        """
        self.Eiters += 1

        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths, imgae_lengths=image_lengths)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        if mode == 'warmup':
            loss = infonce(sims)
        else:
            loss = self.criterion(
                sims,
                hard_negative=hard_negative,
                labels=labels,
                soft_margin=soft_margin,
                mode=mode,
            )

        # return per-sample loss
        if mode == "eval_loss":
            return loss

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        return loss.item()

    def predict(self, images, captions, lengths, image_lengths=None):
        """
        predict the given samples
        """
        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths, imgae_lengths=image_lengths)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        I = self.criterion(sims, mode="predict")
        p = I.clamp(0, 1)

        return p

    def predict_freq(self, images, captions, lengths, image_lengths, 
                 low_ratio=0.3, lambda_freq=0.5):

        img_emb, txt_emb, _ = self.forward_emb(
            images, captions, lengths, imgae_lengths=image_lengths
        )

        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        # ===== FFT =====
        img_fft = torch.fft.fft(img_emb, dim=-1)
        txt_fft = torch.fft.fft(txt_emb, dim=-1)

        D = img_emb.size(-1)
        K = int(D * low_ratio)

        # low-frequency components
        img_low = img_fft[..., :K].real
        txt_low = txt_fft[..., :K].real

        sim_low = F.cosine_similarity(img_low, txt_low, dim=-1)

        # high-frequency instability
        img_mag = torch.abs(img_fft)
        txt_mag = torch.abs(txt_fft)

        img_ratio = img_mag[..., K:].sum(-1) / (img_mag[..., :K].sum(-1) + 1e-6)
        txt_ratio = txt_mag[..., K:].sum(-1) / (txt_mag[..., :K].sum(-1) + 1e-6)

        instability = torch.abs(img_ratio - txt_ratio)

        score = sim_low - lambda_freq * instability
        score = score.clamp(min=0, max=1)

        return score.detach()


def infonce(scores, tau=0.05):
    # scores = (scores / tau).exp()
    # i2t = scores / (scores.sum(1, keepdim=True))
    # t2i = scores.t() / (scores.t().sum(1, keepdim=True))

    # cost_i2t = -i2t.diag().log().mean()
    # cost_t2i = -t2i.diag().log().mean()
    # return cost_i2t + cost_t2i

    eps = 1e-7
    mask = torch.eye(scores.shape[0])+eps
    mask = mask.cuda()
    scores = (scores / tau).exp()
    i2t = scores / (scores.sum(1, keepdim=True))
    t2i = scores.t() / (scores.t().sum(1, keepdim=True))

    cost_i2t_r = - (mask.log()*i2t).sum(1).mean()
    cost_t2i_r = - (mask.log()*t2i).sum(1).mean()
    cost_i2t = -i2t.diag().log().mean()
    cost_t2i = -t2i.diag().log().mean()

    return 0.5*(cost_i2t_r + cost_t2i_r + cost_i2t + cost_t2i)

