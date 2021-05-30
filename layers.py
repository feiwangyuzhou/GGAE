import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


CUDA = torch.cuda.is_available()


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None

class SpecialSpmmFunctionFinal_rel(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N1, N2, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N1, N2, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)

class SpecialSpmmFinal_rel(nn.Module):
    def forward(self, edge, edge_w, N1, N2, E, out_features):
        return SpecialSpmmFunctionFinal_rel.apply(edge, edge_w, N1, N2, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True, n_path=1):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        # self.a_reverse = nn.Parameter(torch.zeros(
        #     size=(out_features, 2 * in_features + nrela_dim)))
        # nn.init.xavier_normal_(self.a_reverse.data, gain=1.414)

        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)
        # self.a_2_reverse = nn.Parameter(torch.zeros(size=(1, out_features)))
        # nn.init.xavier_normal_(self.a_2_reverse.data, gain=1.414)


        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()


        # pdb.set_trace()
        if n_path>0:
            self.W_h_e = nn.Parameter(torch.zeros(
                size=(self.out_features * 2, self.out_features*2)))
            nn.init.xavier_uniform_(self.W_h_e.data, gain=1.414)
            self.W_h_p = nn.Parameter(torch.zeros(
                size=(self.out_features * 2, self.out_features*2)))
            nn.init.xavier_uniform_(self.W_h_e.data, gain=1.414)
            self.W_h_b = nn.Parameter(torch.zeros(
                size=(1, self.out_features*2)))
            nn.init.xavier_uniform_(self.W_h_b.data, gain=1.414)


            self.W_h = nn.Parameter(torch.zeros(
                size=(self.out_features * 2, self.out_features)))
            nn.init.xavier_uniform_(self.W_h.data, gain=1.414)
        else:
            self.W_h = nn.Parameter(torch.zeros(
                size=(self.out_features * 2, self.out_features)))
            nn.init.xavier_uniform_(self.W_h.data, gain=1.414)


    def forward(self, input, edge, edge_embed, edge_reverse, edge_embed_reverse,
                edge_list_nhop, edge_embed_nhop, edge_list_reverse_nhop, edge_embed_reverse_nhop,
                edge_path, all_node_path_toedge, all_rel_path):
        N = input.size()[0]
        # pdb.set_trace()
        # Self-attention on the nodes - Shared attention mechanism
        if len(edge_list_nhop)>0:
            edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
            edge_embed = torch.cat((edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)
        edge_h = torch.cat((input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()

        if len(edge_list_reverse_nhop) > 0:
            edge_reverse = torch.cat((edge_reverse[:, :], edge_list_reverse_nhop[:, :]), dim=1)
            edge_embed_reverse = torch.cat((edge_embed_reverse[:, :], edge_embed_reverse_nhop[:, :]), dim=0)
        edge_h_reverse = torch.cat((input[edge_reverse[0, :], :], input[edge_reverse[1, :], :], edge_embed_reverse[:, :]), dim=1).t()


        if len(edge_path)>0:
            # for path, a path attention or path + neighbor attention with neighbor
            edge_path_h = torch.cat((all_node_path_toedge[:,:], all_rel_path[:,:]),dim=1).t()
            edge_path_h_reverse = edge_path_h
            edge_path = edge_path.t()
            edge_path_reverse = torch.cat((edge_path[1, :].unsqueeze(0), edge_path[0, :].unsqueeze(0)), dim=0)

            edge_path_m = self.a.mm(edge_path_h)
            edge_path_m_reverse = self.a.mm(edge_path_h_reverse)

            powers_path = -self.leakyrelu(self.a_2.mm(edge_path_m).squeeze())
            powers_path_reverse = -self.leakyrelu(self.a_2.mm(edge_path_m_reverse).squeeze())

            edge_path_e = torch.exp(powers_path).unsqueeze(1)
            edge_path_e_reverse = torch.exp(powers_path_reverse).unsqueeze(1)
            assert not torch.isnan(edge_path_e).any()
            assert not torch.isnan(edge_path_e_reverse).any()

            e_path_rowsum = self.special_spmm_final(
                edge_path, edge_path_e, N, edge_path_e.shape[0], 1)
            e_path_rowsum[e_path_rowsum == 0.0] = 1e-12

            e_path_rowsum_reverse = self.special_spmm_final(
                edge_path_reverse, edge_path_e_reverse, N, edge_path_e_reverse.shape[0], 1)
            e_path_rowsum_reverse[e_path_rowsum_reverse == 0.0] = 1e-12

            e_path_rowsum = e_path_rowsum
            e_path_rowsum_reverse = e_path_rowsum_reverse

            edge_path_e = edge_path_e.squeeze(1)
            edge_path_e = self.dropout(edge_path_e)

            edge_path_e_reverse = edge_path_e_reverse.squeeze(1)
            edge_path_e_reverse = self.dropout(edge_path_e_reverse)

            edge_path_w = (edge_path_e * edge_path_m).t()
            edge_path_w_reverse = (edge_path_e_reverse * edge_path_m_reverse).t()

            h_path_prime = self.special_spmm_final(
                edge_path, edge_path_w, N, edge_path_w.shape[0], self.out_features)
            h_path_prime_reverse = self.special_spmm_final(
                edge_path_reverse, edge_path_w_reverse, N, edge_path_w_reverse.shape[0], self.out_features)
            assert not torch.isnan(h_path_prime).any()
            assert not torch.isnan(h_path_prime_reverse).any()

            h_path_prime = h_path_prime.div(e_path_rowsum)
            h_path_prime_reverse = h_path_prime_reverse.div(e_path_rowsum_reverse)


        # edge_h: (2*in_dim + nrela_dim) x E
        edge_m = self.a.mm(edge_h)
        edge_m_reverse = self.a.mm(edge_h_reverse)
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        powers_reverse = -self.leakyrelu(self.a_2.mm(edge_m_reverse).squeeze())

        edge_e = torch.exp(powers).unsqueeze(1)
        edge_e_reverse = torch.exp(powers_reverse).unsqueeze(1)

        assert not torch.isnan(edge_e).any()
        assert not torch.isnan(edge_e_reverse).any()
        # edge_e: E

        # pdb.set_trace()
        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum_reverse = self.special_spmm_final(
            edge_reverse, edge_e_reverse, N, edge_e_reverse.shape[0], 1)
        e_rowsum_reverse[e_rowsum_reverse == 0.0] = 1e-12

        e_rowsum = e_rowsum
        e_rowsum_reverse = e_rowsum_reverse

        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)
        edge_e = self.dropout(edge_e)
        # edge_e: E
        edge_e_reverse = edge_e_reverse.squeeze(1)
        edge_e_reverse = self.dropout(edge_e_reverse)

        edge_w = (edge_e * edge_m).t()
        edge_w_reverse = (edge_e_reverse * edge_m_reverse).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)
        h_prime_reverse = self.special_spmm_final(
            edge_reverse, edge_w_reverse, N, edge_w_reverse.shape[0], self.out_features)

        assert not torch.isnan(h_prime).any()
        assert not torch.isnan(h_prime_reverse).any()

        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        h_prime_reverse = h_prime_reverse.div(e_rowsum_reverse)

        # pdb.set_trace()
        # h_prime: N x out
        if len(edge_path) > 0:
            h_prime_e = torch.cat((h_prime, h_prime_reverse),dim=-1)
            h_prime_p = torch.cat((h_path_prime,h_path_prime_reverse), dim=-1)
            ht=torch.sigmoid(h_prime_e.mm(self.W_h_e)+h_prime_p.mm(self.W_h_p)+self.W_h_b)
            h_prime = ht*h_prime_e + (1-ht)*h_prime_p
            # h_prime = h_prime_p

            # h_prime = torch.cat((h_prime, h_prime_reverse, h_path_prime,h_path_prime_reverse), dim=-1)
        else:
            h_prime = torch.cat((h_prime, h_prime_reverse), dim=-1)
        h_prime = h_prime.mm(self.W_h)

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphAttentionLayer_rel(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(SpGraphAttentionLayer_rel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, in_features+out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final_rel = SpecialSpmmFinal_rel()

    def forward(self, input, edge, edge_embed):
        N1 = input.size()[0]
        N2 = max(edge[1,:]).tolist()+1

        # Self-attention on the nodes - Shared attention mechanism
        # pdb.set_trace()
        edge_h = torch.cat((input[edge[0, :], :], edge_embed[:, :]), dim=1).t()
        # edge_h: (2*in_dim + nrela_dim) x E

        edge_m = self.a.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        edge_e = torch.exp(powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final_rel(
            edge, edge_e, N1, N2, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final_rel(
            edge, edge_w, N1, N2, edge_w.shape[0], self.out_features)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        # if this layer is last layer,
        return h_prime
