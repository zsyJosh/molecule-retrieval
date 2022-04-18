import torch
from torchdrug.core import Registry as R
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torchdrug import data, layers, utils, core
from collections.abc import Sequence
from torchdrug.data import PackedGraph, Graph
import pickle
#from deepset import DeepSet

class Retriever(nn.Module, core.Configurable):
    """
        Parameters:
            dim (int): dim of embeddings
            emb_path (str): path of the stored embeddings
            emb_lable (tensor): [num_labels, N] specify the labels of N evidence graphs
    """
    def __init__(self, dim, emb_path, emb_label, num_label_types, topk, num_evidence, label_concat=True, each_concat=False):
        super(Retriever, self).__init__()
        self.dim = dim
        self.evidence_emb = torch.nn.Embedding(num_evidence, dim)
        with open(emb_path, 'rb') as f:
            pretrained_evi_emb = pickle.load(f)
        assert pretrained_evi_emb.shape == self.evidence_emb.weight.shape
        self.evidence_emb.from_pretrained(pretrained_evi_emb)
        self.evidence_emb.weight.requires_grad = True
        self.emb_label = emb_label
        self.num_label_types = emb_label.shape[0]
        print('self.num_label_types', self.num_label_types)
        print('num_label_types', num_label_types)
        self.num_classes_per_label = emb_label.max(dim=-1).values + 1
        self.label_embedding = []
        for i in range(self.num_label_types):
            num_classes = self.num_classes_per_label[i]
            class_emb = torch.nn.Embedding(num_classes, dim)
            class_emb.cuda()
            self.label_embedding.append(class_emb)
        self.k = topk
        self.num_evidence = num_evidence
        self.label_concat = label_concat
        if label_concat and not each_concat:
            self.deepset = DeepSet(input_dim=dim * (self.num_label_types + 1), hidden_dim=dim)
        elif not label_concat and not each_concat:
            self.deepset = DeepSet(input_dim=dim, hidden_dim=dim)
        elif label_concat and each_concat:
            self.deepset = DeepSet(input_dim=dim * (self.num_label_types + 2), hidden_dim=dim)
        elif not label_concat and each_concat:
            self.deepset = DeepSet(input_dim=dim * 2, hidden_dim=dim)
        self.sfm = torch.nn.Softmax(dim=-1)
        self.each_concat = each_concat
        if self.each_concat:
            self.output_dim = dim
        else:
            self.output_dim = dim * 2

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
            #kvalue = kvalue[:, 1:]
            kind = kind[:, 1:]
        elif mode == 'test':
            kvalue, kind = torch.topk(score, self.k, dim=-1)
        else:
            raise NotImplementedError
        if self.each_concat:
            kevidence_emb = self.evidence_emb(kind)
            stacked_graph_feature = graph_feature.repeat(1, 1, self.k).reshape(batch_size, self.k, self.dim)
            if self.label_concat:
                label_emb = []
                for i in range(self.emb_label.shape[0]):
                    klabel = torch.gather(self.emb_label[i].expand(batch_size, self.num_evidence), -1, kind).cuda()
                    klabel_emb = self.label_embedding[i](klabel)
                    label_emb.append(klabel_emb)
                label_emb = torch.cat(label_emb, dim=-1)
                input = torch.cat([stacked_graph_feature, kevidence_emb, label_emb], dim=-1)
            else:
                input = torch.cat([stacked_graph_feature, kevidence_emb], dim=-1)
            output = self.deepset(input)
        else:
            kevidence_emb = self.evidence_emb(kind)
            if self.label_concat:
                label_emb = []
                for i in range(self.emb_label.shape[0]):
                    klabel = torch.gather(self.emb_label[i].expand(batch_size, self.num_evidence), -1, kind).cuda()
                    klabel_emb = self.label_embedding[i](klabel)
                    label_emb.append(klabel_emb)
                label_emb = torch.cat(label_emb, dim=-1)
                input = torch.cat([kevidence_emb, label_emb], dim=-1)
            else:
                input = kevidence_emb
            deepset_out = self.deepset(input)
            output = torch.cat([graph_feature, deepset_out], dim=-1)
        return output
