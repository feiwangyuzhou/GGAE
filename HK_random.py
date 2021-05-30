import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import queue
import pdb


class HKrandom(nn.Module):
    def __init__(self, nodes, edges):
        # pdb.set_trace()
        self.nodes = nodes
        self.edges = edges
        self.nodeCount = len(self.nodes)
        self.edgeCount = len(self.edges)
        self.cx = [-1]*self.nodeCount
        self.cy = [-1]*self.nodeCount
        self.edge1 = {}
        self.edge2 = {}
        self.distx = {}
        self.disty = {}
        self.que = queue.Queue()
        self.first = [-1]*self.nodeCount
        self.rand = [r for r in range(self.nodeCount)]
        self.edge_num = 0
        self.ans = 0
        self.matching={}
        self.drivernode=[]
        self.allpath=[]


    def AddEdge(self, a, b):
        # pdb.set_trace()
        self.edge1[self.edge_num] = b  # target for each edge
        self.edge2[self.edge_num] = self.first[a]  #the next edge with the same source
        self.first[a] = self.edge_num
        self.edge_num=self.edge_num+1

    def BFS(self):
        # print("bfs start")
        flag=False
        self.distx = [0] * self.nodeCount
        self.disty = [0] * self.nodeCount
        # add to queue
        for i_node in range(self.nodeCount):
            if self.cx[self.rand[i_node]] == -1:
                self.que.put(self.rand[i_node])

        while (not self.que.empty()):
            i_node=self.que.get()
            # print("bfs ing")
            k_node=self.first[i_node]
            while k_node != -1:
                j_node=self.edge1[k_node]
                if self.disty[j_node] == 0:
                    self.disty[j_node]=self.distx[i_node]+1
                    if self.cy[j_node] == -1:
                        flag = True
                    elif self.cy[j_node]!=-1:
                        self.distx[self.cy[j_node]] = self.disty[j_node] + 1
                        self.que.put(self.cy[j_node])
                k_node=self.edge2[k_node]
        # print("bfs end")
        return flag

    def DFS(self,i_node):
        # print("dfs start")
        k_node=self.first[i_node]
        while k_node != -1:
            j_node=self.edge1[k_node]
            if self.disty[j_node] == self.distx[i_node] + 1:  #j_node is after of i_node, but may not be augmented path
                self.disty[j_node]=0  # j_node has been used
                if (self.cy[j_node] == -1) or self.DFS(self.cy[j_node]):
                    self.cx[i_node] = j_node
                    self.cy[j_node] = i_node
                    # print("dfs true end")
                    return True
            k_node=self.edge2[k_node]
        # print("dfs false end")
        return False

    def HK(self):
        # pdb.set_trace()
        random.shuffle(self.rand)
        random.shuffle(self.edges)

        for i_node in range(self.edgeCount):
            self.AddEdge(self.edges[i_node][0],self.edges[i_node][2])
        # pdb.set_trace()
        while self.BFS():
            for i_node in range(self.nodeCount):
                if self.cx[self.rand[i_node]] == -1 and self.DFS(self.rand[i_node]):
                    self.ans=self.ans+1
        # pdb.set_trace()
        # find all matching edges
        for i_edge in range(self.edgeCount):
            source_id= self.edges[i_edge][0]
            target_id= self.edges[i_edge][2]
            if self.cx[source_id] !=-1:
                if (self.cx[source_id]==target_id) and (self.cy[target_id]==source_id):
                    self.matching[source_id]=self.edges[i_edge]
        # pdb.set_trace()
        # find all driver nodes
        for i_node in self.nodes:
            if self.cy[i_node] == -1:
                self.drivernode.append(i_node)

        # find all path from driver node
        # pdb.set_trace()
        max_len=0
        for d_node in self.drivernode:
            path=[]
            dd_node=d_node
            while(dd_node in self.matching):
                edge_mm=self.matching.pop(dd_node)
                path.append(edge_mm)
                dd_node=edge_mm[2]
            if len(path)>1 and len(path)<=4:
                self.allpath.append(path)
                if len(path)>max_len:
                    max_len=len(path)
        # pdb.set_trace()
        # find all path for ring
        if len(self.matching)>0:
            for i_node in self.nodes:
                path=[]
                dd_node=i_node
                while (dd_node in self.matching):
                    edge_mm = self.matching.pop(dd_node)
                    path.append(edge_mm)
                    dd_node = edge_mm[2]
                if len(path) > 1 and len(path) <= 4:
                    self.allpath.append(path)
                    if len(path) > max_len:
                        max_len = len(path)
        if len(self.matching) > 0:
            print("error")
            pdb.set_trace()
        # pdb.set_trace()
        return self.allpath, max_len
















