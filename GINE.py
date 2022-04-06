import torch
from torchdrug.core import Registry as R
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torchdrug import data, layers, utils, core
from torchdrug.layers import MessagePassingBase
from collections.abc import Sequence
from torchdrug.data import PackedGraph, Graph

class VN(object):
    """
    Add a virtual node and connect it with every node in the graph.

    Parameters:
        relation (int, optional): relation of virtual edges.
            By default, use the maximal relation in the graph plus 1.
        weight (int, optional): weight of virtual edges
        node_feature (array_like, optional): feature of the virtual node
        edge_feature (array_like, optional): feature of virtual edges
        kwargs: other attributes of the virtual node or virtual edges
    """

    def __init__(self, relation=None, weight=1, node_feature=None, edge_feature=None, **kwargs):
        self.relation = relation
        self.weight = weight

        self.default = {k: torch.as_tensor(v) for k, v in kwargs.items()}
        if node_feature is not None:
            self.default["node_feature"] = torch.as_tensor(node_feature)
        if edge_feature is not None:
            self.default["edge_feature"] = torch.as_tensor(edge_feature)

    def __call__(self, old_graph):
        graph = old_graph.clone()
        edge_list = graph.edge_list
        edge_weight = graph.edge_weight
        num_node = graph.num_node
        num_relation = graph.num_relation

        existing_node = torch.arange(num_node, device=edge_list.device)
        virtual_node = torch.ones(num_node, dtype=torch.long, device=edge_list.device) * num_node
        node_in = torch.cat([existing_node])
        node_out = torch.cat([virtual_node])
        if edge_list.shape[1] == 2:
            new_edge = torch.stack([node_in, node_out], dim=-1)
        else:
            if self.relation is None:
                relation = num_relation
                num_relation = num_relation + 1
            else:
                relation = self.relation
            relation = relation * torch.ones(num_node, dtype=torch.long, device=edge_list.device)
            new_edge = torch.stack([node_in, node_out, relation], dim=-1)
        edge_list = torch.cat([edge_list, new_edge])
        new_edge_weight = self.weight * torch.ones(num_node, device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, new_edge_weight])

        # add default node/edge attributes
        data = graph.data_dict.copy()
        for key, value in graph.meta_dict.items():
            if value == "node":
                if key in self.default:
                    new_data = self.default[key].unsqueeze(0)
                else:
                    new_data = torch.zeros(1, *data[key].shape[1:], dtype=data[key].dtype, device=data[key].device)
                data[key] = torch.cat([data[key], new_data])
            elif value == "edge":
                if key in self.default:
                    repeat = [-1] * (data[key].ndim - 1)
                    new_data = self.default[key].expand(num_node, *repeat)
                else:
                    new_data = torch.zeros(num_node, *data[key].shape[1:],
                                           dtype=data[key].dtype, device=data[key].device)
                data[key] = torch.cat([data[key], new_data])

        graph = type(graph)(edge_list, edge_weight=edge_weight, num_node=num_node + 1,
                            num_relation=num_relation, meta=graph.meta_dict, **data)

        return graph

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
                 readout="sum", virtual_node=False):
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
        self.virtual_node = virtual_node

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            layer_hidden_dims = [self.dims[i + 1]] * (num_mlp_layer - 1)
            self.layers.append(GINE_layer(self.dims[i], self.dims[i + 1], self.edge_input_dim,
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
        if self.virtual_node:
            def add_virtual(molecule):
                molecule.cpu()
                vngraph = VN()
                return vngraph(molecule).cuda()
            upgraph = graph.unpack()
            vn_upgraph = [add_virtual(mol) for mol in upgraph]
            graph = data.Graph.pack(vn_upgraph)


        for layer in self.layers:
            if self.virtual_node:
                layer_input += graph.node_feature[-1, :]
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


