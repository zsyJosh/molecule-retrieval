from torch import nn
from torchdrug import data, layers, utils, core

class encoder_retriever(nn.Module, core.Configurable):
    """
    Parameters:
        encoder(nn.Module): the graph representation encoder
        retriever(nn.Module): the graph retriever model
    """

    def __init__(self, encoder, retriever):
        super(encoder_retriever, self).__init__()
        self.encoder = encoder
        self.retriever = retriever
        self.output_dim = retriever.output_dim

    def forward(self, graph, input, mode):
        """
        concatenate encoder and retriever for core.Engine
        :param graph: the batched query molecule
        :param input: node representation of query molecule
        :param mode: 'train' or 'test'. If 'train', mask the evidence embedding of its own
        :return: graph representation of the query graph
        """
        enc_out = self.encoder(graph, input)
        graph_rep = enc_out['graph_feature']
        res = self.retriever(graph_rep, mode)
        return res
