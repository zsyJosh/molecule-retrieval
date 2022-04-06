import torch
from torchdrug.core import Registry as R
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter
from torchdrug import data, layers, utils, core
from torchdrug.layers import MessagePassingBase, SumReadout
from collections.abc import Sequence
from typing import Optional
from torchdrug.transforms import VirtualNode, VirtualAtom
from torchdrug.data.molecule import Molecule


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.main = [
            nn.Linear(in_dim, 2 * in_dim),
            nn.BatchNorm1d(2 * in_dim),
            nn.ReLU()
        ]
        self.main.append(nn.Linear(2 * in_dim, out_dim))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)

def global_add_pool(x: torch.Tensor, batch: Optional[torch.Tensor],
                    size: Optional[int] = None) -> torch.Tensor:
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    if batch is None:
        return x.sum(dim=0, keepdim=True)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')

class VNAgg(nn.Module):
    def __init__(self, dim, conv_type="gin"):
        super().__init__()
        self.conv_type = conv_type
        if "gin" in conv_type:
            self.mlp = nn.Sequential(
                MLP(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            )
        elif "gcn" in conv_type:
            self.W0 = nn.Linear(dim, dim)
            self.W1 = nn.Linear(dim, dim)
            self.nl_bn = nn.Sequential(
                nn.BatchNorm1d(dim),
                nn.ReLU()
            )
        else:
            raise NotImplementedError('Unrecognised model conv : {}'.format(conv_type))

    def forward(self, virtual_node, embeddings, batch_vector):
        if batch_vector.size(0) > 0:  # ...or the operation will crash for empty graphs
            G = global_add_pool(embeddings, batch_vector)
        else:
            G = torch.zeros_like(virtual_node)
        if "gin" in self.conv_type:
            virtual_node = virtual_node + G
            virtual_node = self.mlp(virtual_node)
        elif "gcn" in self.conv_type:
            virtual_node = self.W0(virtual_node) + self.W1(G)
            virtual_node = self.nl_bn(virtual_node)
        else:
            raise NotImplementedError('Unrecognised model conv : {}'.format(self.conv_type))
        return virtual_node

class GINE_layer(MessagePassingBase):
    """
    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        hidden_dims (list of int, optional): hidden dimensions
        eps (float, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, hidden_dims=None, eps=0, learn_eps=False,
                 batch_norm=False, activation="relu"):
        super(GINE_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim

        eps = torch.tensor([eps], dtype=torch.float32)
        if learn_eps:
            self.eps = nn.Parameter(eps)
        else:
            self.register_buffer("eps", eps)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if hidden_dims is None:
            hidden_dims = []
        self.mlp = layers.MLP(input_dim, list(hidden_dims) + [output_dim], activation)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None


    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message.relu()


    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        return update


    def message_and_aggregate(self, graph, input):
        adjacency = utils.sparse_coo_tensor(graph.edge_list.t()[:2], graph.edge_weight, (graph.num_node, graph.num_node))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_weight = graph.edge_weight.unsqueeze(-1)
            if self.edge_linear.in_features > self.edge_linear.out_features:
                edge_input = self.edge_linear(edge_input)
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0,
                                      dim_size=graph.num_node)
            if self.edge_linear.in_features <= self.edge_linear.out_features:
                edge_update = self.edge_linear(edge_update)
            update += edge_update

        return update


    def combine(self, input, update):
        output = self.mlp((1 + self.eps) * input + update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

class Convblock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, edge_input_dim=None, hidden_dims=None, eps=0, learn_eps=False,
                 batch_norm=False, activation="relu", conv_type='gin', last_layer=False, dropout=0.5,
                 virtual_node=True, virtual_node_agg=True):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == 'gin':
            self.conv = GINE_layer(input_dim, output_dim, edge_input_dim, hidden_dims, eps, learn_eps,
                 batch_norm, activation)
        self.norm = nn.BatchNorm1d(output_dim)
        self.act = activation or nn.Identity()
        self.last_layer = last_layer

        self.dropout_ratio = dropout

        self.virtual_node = virtual_node
        self.virtual_node_agg = virtual_node_agg
        if self.virtual_node and self.virtual_node_agg:
            self.vn_aggregator = VNAgg(input_dim, conv_type=conv_type)



@R.register("models.GINE")
class GINE(nn.Module, core.Configurable):
    """
    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        num_mlp_layer (int, optional): number of MLP layers
        eps (int, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, num_evidence, evidence_dim=1024, input_dim=None, hidden_dims=None, edge_input_dim=None, num_mlp_layer=2, eps=0, learn_eps=False,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False,
                 readout="sum"):
        super(GINE, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.edge_input_dim = edge_input_dim
        self.num_evidence = num_evidence
        self.evidence_dim = evidence_dim
        self.evidence_emb = torch.nn.Embedding(num_evidence, evidence_dim)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            layer_hidden_dims = [self.dims[i + 1]] * (num_mlp_layer - 1)
            self.layers.append(Convblock(self.dims[i], self.dims[i + 1], self.edge_input_dim,
                                                           layer_hidden_dims, eps, learn_eps, batch_norm, activation))
        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input

        for layer in self.layers:

            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }