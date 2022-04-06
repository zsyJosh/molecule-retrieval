import torch
import torchdrug
from torchdrug import data
from torchdrug.transforms import VirtualNode, VirtualAtom
from torchdrug.data.dataloader import graph_collate
import GINE
from GINE import VN
import numpy as np
from torchdrug import tasks


mol = data.Molecule.from_smiles("C1=CC=CC=C1", node_feature="default",
                                edge_feature="default", graph_feature="ecfp")

dataset = data.MoleculeDataset()
dataset.load_csv('/Users/zhaoshiyu/PycharmProjects/retrieval/tmp/clintox.csv', edge_feature='default')
lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
sample = dataset[0]
sample2 = dataset[1]


print(dataset.targets)
def a():
    train_ind = np.array(train_set.indices)
    emb_ind = [len(train_set)] * len(dataset)
    emb_ind = np.array(emb_ind)
    emb_ind[train_ind] = np.arange(len(train_set))
    dataset.targets['EMB'] = emb_ind
a()
print(train_set[0])
print(train_set[1])
print(train_set[2])

'''
tasks = dataset.tasks
targets = dataset.targets
num_labels = len(tasks) - 1
emb_label = []
for i in range(num_labels):
    label = tasks[i]
    print(label)
    full_label = np.array(targets[label])
    evidence_label = full_label[train_ind]
    emb_label.append(evidence_label.tolist())
emb_label = torch.tensor(emb_label)
print(emb_label)
print(emb_label.shape)
'''
'''
task = ['FDA_APPROVED', 'CT_TOX']
print([sample[t] for t in task])
#target = torch.stack([sample[t] for t in task], dim=-1)
target = torch.tensor([sample[t] for t in task])
print(target)
labeled = sample.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
print(labeled)

print(sample['graph'].node_feature.float().shape)
ef_dim = valid_set[0].pop('graph').edge_feature.shape[1]
num_evidence = len(train_set)
hidden_dims = [256, 256, 256, 256]
evidence_dim = sum(hidden_dims)
model = GINE.GINE(num_evidence= num_evidence, evidence_dim=evidence_dim, input_dim=dataset.node_feature_dim,
                   hidden_dims=hidden_dims, edge_input_dim=ef_dim,
                   short_cut=True, batch_norm=True, concat_hidden=True, virtual_node=False)

print(model(sample['graph'], sample['graph'].node_feature.float()))
graph = sample['graph']
graph2 = sample2['graph']

new_graph = VN()
ng = new_graph(graph)
#new_graph = ng['graph']

new_graph2 = VN()
ng2 = new_graph2(graph2)
#new_graph2 = ng2['graph']

print('old', graph)
print('new', ng)
print(graph2.node_feature.shape)
print('old2', graph2.node_feature[-1, :])
print('new2', ng2.node_feature[-2, :])
print('new2', ng2.node_feature[-1, :])

print(sample.get("labeled", True))



#print('old el', graph.edge_list)
#print('new el', new_graph.edge_list)

#print('old atom_type', graph.atom_type)
#print('nen atom_type', new_graph.atom_type)
'''