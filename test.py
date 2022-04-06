import torch
from torchdrug import data, datasets
from torchdrug import core, models, tasks, utils
from torch.nn import functional as F
import time
import pickle
from pathlib import Path
import os

import numpy as np

# encoder pretraining
ef_dim = valid_set[0].pop('graph').edge_feature.shape[1]
num_evidence = len(train_set)
hidden_dims = [256, 256, 256, 256]
evidence_dim = sum(hidden_dims)
load = True
model = GINE(num_evidence= num_evidence, evidence_dim=evidence_dim, input_dim=dataset.node_feature_dim,
                   hidden_dims=hidden_dims, edge_input_dim=ef_dim,
                   short_cut=True, batch_norm=True, concat_hidden=True)
if load:
  task = tasks.PropertyPrediction(model, task=dataset.tasks[:-1],
                                criterion="bce", metric=("auprc", "auroc"))
else:
  task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                criterion="bce", metric=("auprc", "auroc"))

optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0], batch_size=1024)
#solver.train(num_epoch=100)
#solver.evaluate("valid")
#tm = '_'.join(time.ctime().split())
#solver.save('/content/chkpt/' + tm + '_encoder.pt')
print(dataset.tasks)
solver.load('/content/chkpt/Tue_Apr_5_10:00:24_2022_encoder.pt')

# produce train_set as molecule dataset
train_ind = np.array(train_set.indices)
emb_ind = [len(train_set)] * len(dataset)
emb_ind = np.array(emb_ind)
emb_ind[train_ind] = np.arange(len(train_set))
dataset.targets['EMB'] = emb_ind

# pretrained index
pretrained_embedding = torch.zeros(len(train_set), evidence_dim)
for id, sample in enumerate(train_set):
    assert id == sample['EMB']
    emb = model(sample['graph'].cuda(), sample['graph'].node_feature.float().cuda())['graph_feature']
    pretrained_embedding[id] = emb

emb_path = Path('/content/emb')
if not emb_path.is_dir():
    os.mkdir('/content/emb')
tm = '_'.join(time.ctime().split())
emb_file = '/content/emb/' + tm + '.pt'
with open(emb_file, 'wb') as f:
    pickle.dump(pretrained_embedding, f)

# retrieval model
# TODO: is encoder model frozen during the retrieval
# TODO: be attention with infomation leakage in retrieval model
retrieval = None
retrieval_task = RetrievalReader(encoder_model=model, retrieval_model=retrieval, task=dataset.tasks[:-1],
                                criterion="bce", metric=("auprc", "auroc"))

retrieval_optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(retrieval_task, train_set, valid_set, test_set, retrieval_optimizer, gpus=[0], batch_size=1024)
solver.train(num_epoch=100)

# demo
samples = []
categories = set()
for sample in valid_set:
    category = tuple([v for k, v in sample.items() if k != "graph"])
    if category not in categories:
        categories.add(category)
        samples.append(sample)
samples = data.graph_collate(samples)
samples = utils.cuda(samples)

preds = F.sigmoid(task.predict(samples))
targets = task.target(samples)

titles = []
for pred, target in zip(preds, targets):
    pred = ", ".join(["%.2f" % p for p in pred])
    target = ", ".join(["%d" % t for t in target])
    titles.append("predict: %s\ntarget: %s" % (pred, target))
graph = samples["graph"]
graph.visualize(titles, figure_size=(3, 3.5), num_row=1)