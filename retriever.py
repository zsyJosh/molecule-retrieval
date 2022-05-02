import torch
from torchdrug.core import Registry as R
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torchdrug import data, layers, utils, core
from collections.abc import Sequence
from torchdrug.data import PackedGraph, Graph
import pickle

class Retriever(nn.Module, core.Configurable):
    """
        Parameters:
            dim (int): dim of embeddings
            emb_path (str): path of the stored embeddings
            emb_lable (tensor): [num_labels, N] specify the labels of N evidence graphs
    """
    def __init__(self, dim, emb_path, emb_label, num_label_types, topk, num_evidence, out_fation='sum'):
        super(Retriever, self).__init__()
        self.evidence_emb = torch.nn.Embedding(num_evidence, dim)
        with open(emb_path, 'rb') as f:
            pretrained_evi_emb = pickle.load(f)
        assert pretrained_evi_emb.shape == self.evidence_emb.weight.shape
        self.evidence_emb.from_pretrained(pretrained_evi_emb)
        self.evidence_emb.weight.requires_grad = True
        self.emb_label = emb_label
        self.num_label_types = emb_label.shape[0]
        self.num_classes_per_label = emb_label.max(dim=-1).values + 1
        self.label_embedding = []
        for i in range(self.num_label_types):
            num_classes = self.num_classes_per_label[i]
            class_emb = torch.nn.Embedding(num_classes, dim)
            class_emb.cuda()
            self.label_embedding.append(class_emb)
        self.k = topk
        self.num_evidence = num_evidence
        self.sfm = torch.nn.Softmax(dim=-1)
        self.out_fation = out_fation
        if self.out_fation == 'sum':
            self.output_dim = dim
        elif self.out_fation == 'concat':
            self.output_dim = dim * self.num_label_types
        else:
            raise NotImplementedError

        assert self.num_evidence == self.emb_label.shape[1]

    def forward(self, graph_feature, mode):
        """
        retrieve top k evidence graphs, average in the representation space
        :param  graph_feature: [batch_size, dim], a batch of query graph representation
                mode: 'train' or 'test'. If training, masking the embedding of its own to prevent information leakage.
        :return: [batch_size, dim], a batch of averaged top-k evidence graph embedding
        """
        batch_size = graph_feature.shape[0]
        score = torch.mm(graph_feature, self.evidence_emb.weight.T)
        if mode == 'train':
            kvalue, kind = torch.topk(score, self.k + 1, dim=-1)
            kvalue = kvalue[:, 1:]
            kind = kind[:, 1:]
            '''
            kself = kind[:, 0]
            kself = kself.unsqueeze(-1)
            for i in range(self.emb_label.shape[0]):
                label_self = torch.gather(self.emb_label[i].expand(batch_size, self.num_evidence), -1, kself).cuda()
                klabel = torch.gather(self.emb_label[i].expand(batch_size, self.num_evidence), -1, kind).cuda()
                same_label = label_self.expand(batch_size, self.k) == klabel
                hit_rate = same_label.sum(-1)
                hit_rate = hit_rate / self.k
                b_s = len(hit_rate)
                rate = hit_rate.sum() / b_s
                print(rate)
            '''
        elif mode == 'test':
            kvalue, kind = torch.topk(score, self.k, dim=-1)
        else:
            raise NotImplementedError
        kvalue /= torch.sqrt(kvalue)
        kvalue = self.sfm(kvalue)
        # TODO: whether or not to add representations of different labels together? Or concatenate?
        res = []
        for i in range(self.emb_label.shape[0]):
            klabel = torch.gather(self.emb_label[i].expand(batch_size, self.num_evidence), -1, kind).cuda()
            klabel_emb = self.label_embedding[i](klabel)
            weighted_label_emb = klabel_emb * kvalue.unsqueeze(-1)
            res_emb = weighted_label_emb.sum(dim=1)
            res.append(res_emb)
        if self.out_fation == 'sum':
            return sum(res)
        elif self.out_fation == 'concat':
            return torch.cat(res, dim=1)
        else:
            raise NotImplementedError

