import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
import torch as t
class GTLayer(nn.Module):
    def __init__(self, config):
        super(GTLayer, self).__init__()
        self.config = config
        self.qTrans = nn.Parameter(init(t.empty(config['embedding_size'], config['embedding_size'])))
        self.kTrans = nn.Parameter(init(t.empty(config['embedding_size'], config['embedding_size'])))
        self.vTrans = nn.Parameter(init(t.empty(config['embedding_size'], config['embedding_size'])))

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise = -t.log(-t.log(noise))
        return scores + 0.01*noise

    def forward(self, adj, embeds, flag=False):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, self.config['K'], self.config['embedding_size'] // self.config['K']])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, self.config['K'], self.config['embedding_size'] // self.config['K']])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, self.config['K'], self.config['embedding_size'] // self.config['K']])

        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        tem = t.zeros([adj.shape[0], self.config['K']]).cuda()
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)

        resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, self.config['embedding_size']])
        tem = t.zeros([adj.shape[0], self.config['embedding_size']]).cuda()
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd
        return resEmbeds, att