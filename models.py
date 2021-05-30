import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB, SpGraphAttentionLayer_rel
import lstmlayers
import copy
import pdb

CUDA = torch.cuda.is_available()  # checking cuda availability


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads, use_2hop, use_bi, rnn_type, n_path,path_type):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True,
                                                 n_path=n_path)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # pdb.set_trace()
        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.W1 = nn.Parameter(torch.zeros(size=(relation_dim*2, nheads * nhid * 2)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        # self.W2 = nn.Parameter(torch.zeros(size=(relation_dim,  nheads * nhid)))
        # nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        if use_2hop:
            self.W_1 = nn.Parameter(torch.Tensor(relation_dim), requires_grad=True)
            self.W_2 = nn.Parameter(torch.Tensor(relation_dim), requires_grad=True)
            nn.init.constant_(self.W_1, 1.0)
            nn.init.constant_(self.W_2, 1.0)
            self.W_1f = nn.Parameter(torch.Tensor(nheads * nhid), requires_grad=True)
            self.W_2f = nn.Parameter(torch.Tensor(nheads * nhid), requires_grad=True)
            nn.init.constant_(self.W_1f, 1.0)
            nn.init.constant_(self.W_2f, 1.0)


        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False,
                                             n_path=n_path
                                             )
        self.path_type=path_type
        # pdb.set_trace()
        # Set dropout
        if n_path>0:
            lstmlayers.set_my_dropout_prob(0.4)
            lstmlayers.set_seq_dropout(True)
            if use_bi:
                if rnn_type=="lstm":
                    self.nodeRNN = lstmlayers.StackedBRNN(nfeat, nfeat // 2, num_layers=1)
                    self.relRNN = lstmlayers.StackedBRNN(nfeat, nfeat // 2, num_layers=1)
                elif rnn_type=="gru":
                    self.nodeRNN = lstmlayers.StackedBRNN(nfeat, nfeat // 2, num_layers=1, rnn_type=nn.GRU)
                    self.relRNN = lstmlayers.StackedBRNN(nfeat, nfeat // 2, num_layers=1, rnn_type=nn.GRU)
            else:
                if rnn_type == "lstm":
                    self.nodeRNN = lstmlayers.StackedBRNN(nfeat, nfeat, num_layers=1, bidir=False)
                    self.relRNN = lstmlayers.StackedBRNN(nfeat, nfeat, num_layers=1, bidir=False)
                elif rnn_type=="gru":
                    self.nodeRNN = lstmlayers.StackedBRNN(nfeat, nfeat, num_layers=1, rnn_type=nn.GRU, bidir=False)
                    self.relRNN = lstmlayers.StackedBRNN(nfeat, nfeat, num_layers=1, rnn_type=nn.GRU, bidir=False)

            # self.node_self_att = lstmlayers.GetAttentionHiddens(nfeat, nfeat)
            # self.rel_self_att = lstmlayers.GetAttentionHiddens(nfeat, nfeat)
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.nodeRNN.to(self.device)
            # self.nodeRNN = torch.nn.DataParallel(self.nodeRNN)
            # self.relRNN.to(self.device)
            # self.relRNN = torch.nn.DataParallel(self.relRNN)


    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_reverse, edge_embed_reverse,
                edge_list_nhop, edge_type_nhop, edge_list_reverse_nhop, edge_type_reverse_nhop,
                edge_path, all_node, all_node_mask, all_rel, all_rel_mask):

        if len(edge_path)>0:
            # find embedding for nodes and edges in path
            # pdb.set_trace()
            if self.path_type=="or":
                all_node_toedge_elsepad, all_rel_elsepad=self.nodeoredge(all_node,all_node_mask, all_rel, all_rel_mask, entity_embeddings, relation_embed)
            else:
                all_node_toedge_elsepad, all_rel_elsepad = self.nodeandedge(all_node, all_node_mask, all_rel,all_rel_mask, entity_embeddings, relation_embed)

        else:
            all_node_toedge_elsepad = torch.LongTensor([])
            all_rel_elsepad = torch.LongTensor([])
        # end path
        # pdb.set_trace()
        x = entity_embeddings

        # pdb.set_trace()
        edge_embed_nhop=torch.LongTensor([])
        edge_embed_reverse_nhop = torch.LongTensor([])
        if len(edge_type_nhop)>0:
            edge_embed_nhop = relation_embed[edge_type_nhop[:, 0]] * self.W_1 + relation_embed[edge_type_nhop[:, 1]] * self.W_2
            edge_embed_reverse_nhop = relation_embed[edge_type_reverse_nhop[:, 0]] * self.W_1 + relation_embed[edge_type_reverse_nhop[:, 1]] * self.W_2

        x = torch.cat([att(x, edge_list, edge_embed, edge_list_reverse, edge_embed_reverse,
                           edge_list_nhop, edge_embed_nhop, edge_list_reverse_nhop, edge_embed_reverse_nhop,
                           edge_path, all_node_toedge_elsepad, all_rel_elsepad)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        # pdb.set_trace()
        out_relation_1 = relation_embed.mm(self.W)
        if len(edge_path) > 0:
            all_node_toedge_elsepad = all_node_toedge_elsepad.mm(self.W1)
            all_rel_elsepad = all_rel_elsepad.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        edge_embed_reverse = edge_embed
        if len(edge_type_nhop) > 0:
            edge_embed_nhop = out_relation_1[edge_type_nhop[:, 0]] * self.W_1f + out_relation_1[edge_type_nhop[:, 1]] * self.W_2f
            edge_embed_reverse_nhop = out_relation_1[edge_type_reverse_nhop[:, 0]] * self.W_1f + out_relation_1[edge_type_reverse_nhop[:, 1]] * self.W_2f

        x = F.elu(self.out_att(x, edge_list, edge_embed, edge_list_reverse, edge_embed_reverse,
                               edge_list_nhop, edge_embed_nhop, edge_list_reverse_nhop, edge_embed_reverse_nhop,
                               edge_path, all_node_toedge_elsepad, all_rel_elsepad))
        return x, out_relation_1

    def nodeandedge(self,all_node,all_node_mask, all_rel, all_rel_mask, entity_embeddings, relation_embed):
        # cat the node and edge as a only path
        all_node = all_node.view(-1)
        all_rel = all_rel.view(-1)
        all_node_embed = entity_embeddings[all_node]
        all_rel_embed = relation_embed[all_rel]
        all_node_embed = all_node_embed.view(all_node_mask.size(0), all_node_mask.size(1), all_node_embed.size(1))
        all_rel_embed = all_rel_embed.view(all_rel_mask.size(0), all_rel_mask.size(1), all_rel_embed.size(1))

        # pdb.set_trace()
        # cat the node path and edge path
        all_node_embed_tmp=all_node_embed[:,:-1,:].contiguous().view(-1,all_node_embed.size(2)).unsqueeze(0)
        all_rel_embed=all_rel_embed.view(-1, all_node_embed.size(2)).unsqueeze(0)
        all_node_embed_tmp=torch.cat((all_node_embed_tmp,all_rel_embed),dim=0)
        all_node_embed_tmp=all_node_embed_tmp.transpose(1,0).contiguous().view(all_node_embed.size(0),-1,all_node_embed.size(2))
        all_node_embed=all_node_embed[:, -1, :].unsqueeze(0).transpose(1,0)
        all_node_embed=torch.cat((all_node_embed_tmp,all_node_embed),dim=1)

        all_node_embed_tmp = all_node_mask[:, :-1].contiguous().view(-1).unsqueeze(0)
        all_rel_mask = all_rel_mask.view(-1).unsqueeze(0)
        all_node_embed_tmp = torch.cat((all_node_embed_tmp, all_rel_mask), dim=0)
        all_node_embed_tmp = all_node_embed_tmp.transpose(1, 0).contiguous().view(all_node_embed.size(0), -1)
        all_node_mask = all_node_mask[:, -1].unsqueeze(0).transpose(1, 0)
        all_node_mask = torch.cat((all_node_embed_tmp, all_node_mask), dim=1)

        # use BiLSTM for each path
        all_node_BiLSTM = self.nodeRNN(all_node_embed, all_node_mask)
        # all_rel_BiLSTM = self.relRNN(all_rel_embed, all_rel_mask)

        # pdb.set_trace()
        # # self attention  but adding it will exist nan in subsequent procedures
        # all_node_BiLSTM = self.node_self_att(all_node_BiLSTM, all_node_BiLSTM, all_node_mask)
        # all_rel_BiLSTM = self.rel_self_att(all_rel_BiLSTM, all_rel_BiLSTM, all_rel_mask)
        all_node_toedge_elsepad = []
        all_rel_elsepad = []
        for i_path in range(len(all_node_mask)):
            i_len = sum(all_node_mask[i_path])
            source_ids = all_node_BiLSTM[i_path][:i_len]
            i_len_j=0
            while i_len_j < i_len-1:
                source_ids_tmp= torch.cat([source_ids[i_len_j], source_ids[i_len_j+2]], dim=0)
                all_node_toedge_elsepad.append(source_ids_tmp)
                all_rel_elsepad.append(source_ids[i_len_j+1])
                i_len_j = i_len_j + 2

        # pdb.set_trace()
        all_node_toedge_elsepad = torch.stack(all_node_toedge_elsepad, dim=0)
        all_rel_elsepad = torch.stack(all_rel_elsepad, dim=0)

        return all_node_toedge_elsepad, all_rel_elsepad

    def nodeoredge(self,all_node,all_node_mask, all_rel, all_rel_mask, entity_embeddings, relation_embed):
        # divide the path to node path and edge path
        all_node = all_node.view(-1)
        all_rel = all_rel.view(-1)
        all_node_embed = entity_embeddings[all_node]
        all_rel_embed = relation_embed[all_rel]
        all_node_embed = all_node_embed.view(all_node_mask.size(0), all_node_mask.size(1), all_node_embed.size(1))
        all_rel_embed = all_rel_embed.view(all_rel_mask.size(0), all_rel_mask.size(1), all_rel_embed.size(1))

        # use BiLSTM for each path
        all_node_BiLSTM = self.nodeRNN(all_node_embed, all_node_mask)
        all_rel_BiLSTM = self.relRNN(all_rel_embed, all_rel_mask)

        # pdb.set_trace()
        # # self attention  but adding it will exist nan in subsequent procedures
        # all_node_BiLSTM = self.node_self_att(all_node_BiLSTM, all_node_BiLSTM, all_node_mask)
        # all_rel_BiLSTM = self.rel_self_att(all_rel_BiLSTM, all_rel_BiLSTM, all_rel_mask)
        all_node_toedge_elsepad = []
        all_rel_elsepad = []
        for i_path in range(len(all_node_mask)):
            i_len = sum(all_node_mask[i_path])
            source_ids = all_node_BiLSTM[i_path][:i_len]
            source_ids = torch.cat([source_ids[1:i_len], source_ids[:i_len - 1]],dim=1)  # adj is reverse with original edges
            all_node_toedge_elsepad.extend(source_ids)

            edge_ids = all_rel_BiLSTM[i_path][:sum(all_rel_mask[i_path])]
            all_rel_elsepad.extend(edge_ids)

        # pdb.set_trace()
        all_node_toedge_elsepad = torch.stack(all_node_toedge_elsepad, dim=0)
        all_rel_elsepad = torch.stack(all_rel_elsepad, dim=0)

        return all_node_toedge_elsepad, all_rel_elsepad


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT, use_2hop, use_bi, rnn_type, n_path, path_type):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.use_2hop = use_2hop
        self.use_bi = use_bi
        self.rnn_type = rnn_type
        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        # self.entity_out_dim_1 = entity_out_dim[0]
        self.entity_out_dim_1 = entity_out_dim
        self.nheads_GAT_1 = nheads_GAT[0]
        # self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        # self.relation_out_dim_1 = relation_out_dim[0]
        self.relation_out_dim_1 = relation_out_dim

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1, self.use_2hop, self.use_bi, self.rnn_type, n_path,path_type)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

        self.W_relations = nn.Parameter(torch.zeros(
            size=(self.relation_dim, self.relation_out_dim_1*2)))
        nn.init.xavier_uniform_(self.W_relations.data, gain=1.414)


        self.rel_att = SpGraphAttentionLayer_rel(self.relation_out_dim_1*2,
                                                 self.relation_out_dim_1*2,
                                                 dropout=self.drop_GAT,
                                                 alpha=alpha)


    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop, train_indices_reverse_nhop,
                edge_path, all_node, all_node_mask, all_rel, all_rel_mask):
        # getting edge list
        edge_list = adj[0]
        edge_type = adj[1]
        rel_num = adj[2]
        edge_list_reverse = torch.cat((edge_list[1, :].unsqueeze(0), edge_list[0, :].unsqueeze(0)), dim=0)
        # pdb.set_trace()
        edge_list_nhop = torch.LongTensor([])
        edge_list_reverse_nhop = torch.LongTensor([])
        edge_type_nhop = torch.LongTensor([])
        edge_type_reverse_nhop = torch.LongTensor([])
        if len(train_indices_nhop)>0:
            edge_list_nhop = torch.cat((train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
            edge_type_nhop = torch.cat([train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)
            edge_list_reverse_nhop = torch.cat((train_indices_reverse_nhop[:, 3].unsqueeze(-1), train_indices_reverse_nhop[:, 0].unsqueeze(-1)), dim=1).t()
            edge_type_reverse_nhop = torch.cat([train_indices_reverse_nhop[:, 1].unsqueeze(-1), train_indices_reverse_nhop[:, 2].unsqueeze(-1)], dim=1)

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_list_reverse = edge_list_reverse.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_list_reverse_nhop = edge_list_reverse_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()
            edge_type_reverse_nhop = edge_type_reverse_nhop.cuda()

            rel_num = rel_num.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed, edge_list_reverse, edge_embed,
            edge_list_nhop, edge_type_nhop, edge_list_reverse_nhop, edge_type_reverse_nhop,
            edge_path, all_node, all_node_mask, all_rel, all_rel_mask)

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)


        # pdb.set_trace()
        #For out_relation_1, get all pairs es-eo from relevant triples, and do attention on them
        edge_num=torch.cat((edge_type.unsqueeze(0), rel_num.unsqueeze(0)),dim=0)
        edge_num_embed=out_entity_1[edge_list[0,:]]-out_entity_1[edge_list[1,:]]

        out_relation_1 = self.rel_att(out_relation_1, edge_num, edge_num_embed)
        # out_relation_1 = self.rel_avg(out_relation_1, edge_num, edge_num_embed, Corpus_.relation2id_num)
        out_relation_1 =self.relation_embeddings.mm(self.W_relations)+out_relation_1
        out_relation_1 = F.normalize(out_relation_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1

    def rel_avg(self, input, edge, edge_embed, relation2id_num):
        N1 = input.size()[0]
        N2 = max(edge[1, :]).tolist() + 1

        a = torch.sparse_coo_tensor(
            edge, edge_embed, torch.Size([N1, N2, self.relation_out_dim_1*2]))
        b = torch.sparse.sum(a, dim=1)
        b = b.to_dense()

        N3 = [relation2id_num[i] for i in relation2id_num]
        N3 = torch.FloatTensor(N3).unsqueeze(1).cuda()
        # pdb.set_trace()
        b = b.div(N3)

        return b


class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        # self.entity_out_dim_1 = entity_out_dim[0]
        self.entity_out_dim_1 = entity_out_dim
        self.nheads_GAT_1 = nheads_GAT[0]
        # self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        # self.relation_out_dim_1 = relation_out_dim[0]
        self.relation_out_dim_1 = relation_out_dim

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv
