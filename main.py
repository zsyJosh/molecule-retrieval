import torch
from torchdrug import data, datasets
from torchdrug import core, models, tasks, utils
from torch.nn import functional as F
import time
import pickle
from pathlib import Path
import os
import numpy as np
import GINE
from retrieval_task import RetrievalReader
from retriever2 import Retriever
from encoder_retriever import encoder_retriever


# dataset
def dataset_download(path):
    """
    download the dataset and split it
    """
    dataset = datasets.HIV(path)
    trans = True
    if trans:
        from ogb.graphproppred import GraphPropPredDataset
        dataset2 = GraphPropPredDataset(name="ogbg-molhiv", root='dataset/')
        split_idx = dataset2.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        train_set = torch.utils.data.Subset(dataset, train_idx.tolist())
        valid_set = torch.utils.data.Subset(dataset, valid_idx.tolist())
        test_set = torch.utils.data.Subset(dataset, test_idx.tolist())
        return dataset, train_set, valid_set, test_set
    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    lengths += [len(dataset) - sum(lengths)]
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
    return dataset, train_set, valid_set, test_set


# encoder pretraining
def encoder_pretraining(dataset, task, train_set, valid_set, test_set, num_evidence, evidence_dim, hidden_dims, encoder_path=None, load=False, load_path=None):
    """
    pretrain the encoder (GINE here)

    :param dataset: dataset to use
    :param task: target to learn (the classification label here)
    :param train_set: train set
    :param valid_set: valid set
    :param test_set: test set
    :param num_evidence: number of evidence graph in the training set
    :param evidence_dim: number of dim of evidence representation
    :param hidden_dims: dims of the hidden layers (can be different)
    :param encoder_path: the path to store the encoder after training
    :param load: whether to load it or train it
    :param load_path: the path to load the trained encoder
    """

    ef_dim = valid_set[0].pop('graph').edge_feature.shape[1]

    '''    
    encoder = models.GIN(input_dim=dataset.node_feature_dim,
                       hidden_dims=[256, 256, 256, 256],
                       short_cut=True, batch_norm=True, concat_hidden=True)
    '''
    encoder = GINE.GINE(num_evidence= num_evidence, evidence_dim=evidence_dim, input_dim=dataset.node_feature_dim,
                       hidden_dims=hidden_dims, edge_input_dim=ef_dim,
                       short_cut=True, batch_norm=True, concat_hidden=True)
    print(task)
    encoder_task = tasks.PropertyPrediction(encoder, task=task,
                                criterion="bce", metric=("auprc", "auroc"))

    optimizer = torch.optim.Adam(encoder_task.parameters(), lr=1e-3)
    encoder_solver = core.Engine(encoder_task, train_set, valid_set, test_set, optimizer, gpus=[0], batch_size=1024)
    if load:
        assert load_path is not None
        encoder_solver.load(load_path)
        encoder_solver.evaluate("valid")
    else:
        assert encoder_path is not None
        encoder_solver.train(num_epoch=100)
        encoder_solver.evaluate("valid")
        tm = '_'.join(time.ctime().split())
        encoder_solver.save(encoder_path + tm + '_encoder.pt')
    return encoder_solver, encoder


# number the molecules in train_set
def evidence_enumeration(dataset, train_set):
    """
    Give the evidence graph in the training set an order. The number label (the order) is used to retrieve the evidence representation.
    """

    # get the indices of training set
    train_ind = np.array(train_set.indices)

    # initialize for all graphs
    emb_ind = [len(train_set)] * len(dataset)
    emb_ind = np.array(emb_ind)
    emb_ind[train_ind] = np.arange(len(train_set)).tolist()

    # add the order for training graphs only, others' order is set to len(train_set)
    dataset.targets['EMB'] = emb_ind.tolist()
    return dataset


# retrieve evidence label
def evidence_label(dataset, task, train_set, device):
    """
    retrieve the ground truth label of the evidence graphs for each target task
    """
    # make sure we are not retrieving the order
    assert 'EMB' not in task

    # get the indices of training set
    train_ind = np.array(train_set.indices)
    targets = dataset.targets
    emb_label = []

    # get the ground truth label by the training set indices
    for label in task:
        full_label = np.array(targets[label])
        evidence_label = full_label[train_ind]
        emb_label.append(evidence_label.tolist())
    emb_label = torch.tensor(emb_label, device=device)
    return emb_label


# pretrained index
def pretrain_index(encoder, train_set, evidence_dim, evidence_dir):
    """
    encoder the evidence graph and store them
    :param encoder: evidence encoder
    :param train_set: training set
    :param evidence_dim: evidence representation dim
    :param evidence_dir: path to store the evidence representation
    :return: file name of the stored evidence representation
    """
    encoder.eval()
    pretrained_embedding = torch.zeros(len(train_set), evidence_dim)
    with torch.no_grad():
        for id, sample in enumerate(train_set):
            # train_set is subset subscription of dataset, "EMB" is added
            assert id == sample['EMB']
            graph = sample['graph'].cuda()
            emb = encoder(graph, graph.node_feature.float())['graph_feature']
            del graph
            torch.cuda.empty_cache()
            pretrained_embedding[id] = emb

    emb_path = Path(evidence_dir)
    if not emb_path.is_dir():
        os.mkdir(evidence_dir)
    tm = '_'.join(time.ctime().split())
    emb_file = evidence_dir + tm + '.pt'
    with open(emb_file, 'wb') as f:
        pickle.dump(pretrained_embedding, f)
    return emb_file

# retriever model
def retriever_model(dataset, task, train_set, valid_set, test_set, encoder, topk, emb_label, num_class, emb_file, retriever_path):
    """
    Training the retriever model as well as the evidence encoder.
    :param dataset: dataset to use.
    :param task: the label to be classified
    :param train_set: training set
    :param valid_set: validation set
    :param test_set: test set
    :param encoder: evidence/query encoder (they are the same)
    :param topk: top k retrieval
    :param emb_label: the ground truth of evidence graph
    :param num_class: number of classes to be classified
    :param emb_file: embedding file
    :param retriever_path: path to store the retriever model
    :return: trained retriever model
    """
    retrieval = Retriever(dim=1024, emb_path=emb_file, emb_label=emb_label, num_label_types=num_class, topk=topk, num_evidence=len(train_set))
    enc_retr = encoder_retriever(encoder=encoder, retriever=retrieval)
    assert 'EMB' not in task
    retrieval_task = RetrievalReader(model=enc_retr, task=task,
                                    criterion="bce", metric=("auprc", "auroc"), num_class=num_class)

    retrieval_optimizer = torch.optim.Adam(retrieval_task.parameters(), lr=1e-3)
    retrieve_solver = core.Engine(retrieval_task, train_set, valid_set, test_set, retrieval_optimizer, gpus=[0], batch_size=1024)
    retrieve_solver.train(num_epoch=100)
    tm = '_'.join(time.ctime().split())
    retrieve_solver.save(retriever_path + tm + '_retriever.pt')
    return retrieve_solver, retrieval



def main():
    dataset, train_set, valid_set, test_set = dataset_download("/Users/zhaoshiyu/PycharmProjects/retrieval/tmp")

    num_evidence = len(train_set)
    encoder_path = '/content/chkpt/'
    load_path = None
    retriever_path = '/content/chkpt/'

    hidden_dims = [256, 256, 256, 256]
    evidence_dim = sum(hidden_dims)
    task = dataset.tasks.copy()
    num_class = 1
    print(task)
    encoder_solver, encoder = encoder_pretraining(dataset, task, train_set, valid_set, test_set, num_evidence, evidence_dim, hidden_dims,
                                                  encoder_path=encoder_path, load=False, load_path=load_path)
    device = encoder_solver.device
    dataset = evidence_enumeration(dataset, train_set)
    print(task)
    emb_label = evidence_label(dataset, task, train_set, device)
    emb_file = pretrain_index(encoder, train_set, evidence_dim, evidence_dir='/content/emb/')
    print(task)
    retriever_solver, retrieval = retriever_model(dataset, task, train_set, valid_set, test_set, encoder, 10, emb_label, num_class, emb_file, retriever_path)

if __name__ == '__main__':
    main()
