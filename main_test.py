import torch
from torchdrug import data, datasets
from torchdrug import core, models, tasks, utils
from torch.nn import functional as F
import GINE
import time
import pickle
from pathlib import Path
import os
from retrieval_task import RetrievalReader
import numpy as np
from retriever import Retriever
from encoder_retriever import encoder_retriever


# dataset
def dataset_download(path):
    dataset = datasets.ClinTox(path)
    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    lengths += [len(dataset) - sum(lengths)]
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
    return dataset, train_set, valid_set, test_set

# encoder pretraining
def encoder_pretraining(dataset, train_set, valid_set, test_set, num_evidence, evidence_dim, hidden_dims, encoder_path=None, load=False, load_path=None):
    ef_dim = valid_set[0].pop('graph').edge_feature.shape[1]

    encoder = models.GIN(input_dim=dataset.node_feature_dim,
                         hidden_dims=[256, 256, 256, 256],
                         short_cut=True, batch_norm=True, concat_hidden=True)
    task = tasks.PropertyPrediction(encoder, task=dataset.tasks,
                                criterion="bce", metric=("auprc", "auroc"))

    optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
    encoder_solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0], batch_size=1024)
    if load:
        assert load_path is not None
        encoder_solver.load(load_path)
    else:
        assert encoder_path is not None
        encoder_solver.train(num_epoch=100)
        encoder_solver.evaluate("valid")
        tm = '_'.join(time.ctime().split())
        encoder_solver.save(encoder_path + tm + '_encoder.pt')
        encoder_solver.evaluate("valid")
    return encoder_solver, encoder

# number the molecules in train_set
def evidence_enumeration(dataset, train_set):
    train_ind = np.array(train_set.indices)
    emb_ind = [len(train_set)] * len(dataset)
    emb_ind = np.array(emb_ind)
    emb_ind[train_ind] = np.arange(len(train_set)).tolist()
    dataset.targets['EMB'] = emb_ind.tolist()
    return dataset

# retrieve evidence label
def evidence_label(dataset, train_set, device):
    train_tasks = dataset.tasks
    train_ind = np.array(train_set.indices)
    targets = dataset.targets
    num_labels = len(train_tasks) - 1
    emb_label = []
    for i in range(num_labels):
        label = train_tasks[i]
        full_label = np.array(targets[label])
        evidence_label = full_label[train_ind]
        emb_label.append(evidence_label.tolist())
    emb_label = torch.tensor(emb_label, device=device)
    return emb_label

# pretrained index
def pretrain_index(encoder, train_set, evidence_dim, evidence_dir):
    encoder.eval()
    pretrained_embedding = torch.zeros(len(train_set), evidence_dim)
    for id, sample in enumerate(train_set):
        # train_set is subset subscription of dataset, "EMB" is added
        assert id == sample['EMB']
        emb = encoder(sample['graph'].cuda(), sample['graph'].node_feature.float().cuda())['graph_feature']
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
def retriever_model(dataset, train_set, valid_set, test_set, encoder, emb_label, emb_file, retriever_path):
    assert emb_label[emb_label > 1].sum() + emb_label[emb_label < 0].sum() == 0
    retrieval = Retriever(dim=1024, emb_path=emb_file, emb_label=emb_label, topk=10, num_evidence=len(train_set), out_fation='sum')
    enc_retr = encoder_retriever(encoder=encoder, retriever=retrieval)
    training_task = dataset.tasks[:-1]
    assert 'EMB' not in training_task
    retrieval_task = RetrievalReader(model=enc_retr, task=training_task,
                                    criterion="bce", metric=("auprc", "auroc"), num_class=2)

    retrieval_optimizer = torch.optim.Adam(retrieval_task.parameters(), lr=1e-3)
    retrieve_solver = core.Engine(retrieval_task, train_set, valid_set, test_set, retrieval_optimizer, gpus=[0], batch_size=1024)
    retrieve_solver.train(num_epoch=100)
    retrieve_solver.evaluate("valid")
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

    encoder_solver, encoder = encoder_pretraining(dataset, train_set, valid_set, test_set, num_evidence, evidence_dim, hidden_dims,
                                                  encoder_path=encoder_path, load=False, load_path=None)
    device = encoder_solver.device
    dataset = evidence_enumeration(dataset, train_set)
    emb_label = evidence_label(dataset, train_set, device)
    emb_file = pretrain_index(encoder, train_set, evidence_dim, evidence_dir='/content/emb')
    retriever_solver, retrieval = retriever_model(dataset, train_set, valid_set, test_set, encoder, emb_label, emb_file, retriever_path)

if __name__ == '__main__':
    main()
