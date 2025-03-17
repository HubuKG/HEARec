# coding: utf-8
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from models.GTlayer import GTLayer
import random
import torch as t

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class HEARec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(HEARec, self).__init__(config, dataset)
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.cf_model = config['cf_model']
        self.cl_layer = config['cl_layer']
        self.n_ui_layers = config['n_ui_layers']
        self.n_hyper_layer = config['n_hyper_layer']
        self.hyper_num = config['hyper_num']
        self.keep_rate = config['keep_rate']
        self.alpha = config['alpha']
        self.cl_weight = config['cl_weight']
        self.reg_weight = config['reg_weight']
        self.env_weight = config['chi']
        self.align_weight = config['lambda']
        self.eps = config['eps']
        self.depth_cl = config['depth_cl']
        self.gama = config['gama']
        self.beta = config['beta']
        self.theta = config['theta']
        self.P = config['P']
        self.tau = 0.2
        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()

        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.drop = nn.Dropout(p=1 - self.keep_rate)

        # graph transformer
        self.gtLayer = GTLayer(config).cuda()
        self.hgnnLayer = HGNNLayer(self.n_hyper_layer)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.item_image_trs = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))
            self.v_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.hyper_num)))
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.item_text_trs = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))
            self.t_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.hyper_num)))

        self.align = MSE()
        self.warm_missing_item_index = dataset.warm_missing_item_index

    #         self.gating_weightub = nn.Parameter(
    #             torch.FloatTensor(1, hide_dim))
    #         nn.init.xavier_normal_(self.gating_weightub.data)
    #         self.gating_weightu = nn.Parameter(
    #             torch.FloatTensor(hide_dim, hide_dim))
    #         nn.init.xavier_normal_(self.gating_weightu.data)
    #         self.gating_weightib = nn.Parameter(
    #             torch.FloatTensor(1, hide_dim))
    #         nn.init.xavier_normal_(self.gating_weightib.data)
    #         self.gating_weighti = nn.Parameter(
    #             torch.FloatTensor(hide_dim, hide_dim))
    #         nn.init.xavier_normal_(self.gating_weighti.data)

    #     def self_gatingu(self, em):
    #         return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weightu) + self.gating_weightub))

    #     def self_gatingi(self, em):
    #         return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weighti) + self.gating_weightib))

    #     def get_multimedia_emb(self, image_feat, text_feat):
    #         image_emb = self.image_linear(image_feat)
    #         text_emb = self.text_linear(text_feat)
    #         return image_emb, text_emb

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size((self.n_nodes, self.n_nodes)))

    # collaborative graph embedding
    def cge(self):
        if self.cf_model == 'mf':
            cge_embs = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)

        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            cge_embs = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
                cge_embs += [ego_embeddings]
            cge_embs = torch.stack(cge_embs, dim=1)
            cge_embs = cge_embs.mean(dim=1, keepdim=False)
        return cge_embs
        # collaborative graph embedding

    #     def cge(self, u_emb, i_emb):
    #         if self.cf_model == 'mf':
    #             cge_embs = torch.cat((u_emb, i_emb), dim=0)

    #         if self.cf_model == 'lightgcn':
    #             ego_embeddings = torch.cat((u_emb, i_emb), dim=0)
    #             cge_embs = [ego_embeddings]
    #             for _ in range(self.n_ui_layers):
    #                 ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
    #                 cge_embs += [ego_embeddings]
    #             cge_embs = torch.stack(cge_embs, dim=1)
    #             cge_embs = cge_embs.mean(dim=1, keepdim=False)
    #         return cge_embs

    # modality graph embedding
    def mge(self, perturbed, str='v'):
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)
        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[:self.n_users]
        # user_feats = self.user_embedding.weight
        mge_feats = torch.concat([user_feats, item_feats], dim=0)

        # ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        all_embeddings_cl = mge_feats
        for k in range(self.cl_layer):
            mge_feats = torch.sparse.mm(self.norm_adj, mge_feats)
            if perturbed:
                random_noise = torch.rand_like(mge_feats).cuda()
                mge_feats += torch.sign(mge_feats) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(mge_feats)
            if k == self.depth_cl - 1:
                all_embeddings_cl = mge_feats

        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        return final_embeddings

    def get_multimedia_emb(self, image_feat, text_feat):
        image_emb = self.image_linear(image_feat)
        text_emb = self.text_linear(text_feat)
        return image_emb, text_emb

    def get_env_emb(self, mix_ration, env, image_feat, text_feat):
        """
        propagate methods for lightGCN
        """
        # user_emb = self.user_embedding.weight
        mm_emb = mix_ration[0] * self.image_linear(image_feat) + mix_ration[1] * self.text_linear(
            text_feat)  # + mix_ration[env][2] *  self.audio_linear(self.audio_feat)
        # item_emb = self.fusion_linear(mm_emb)
        # item_emb = mm_emb
        user_emb, item_emb = torch.split(mm_emb, [self.n_users, self.n_items], dim=0)
        assert torch.isnan(user_emb).sum() == 0
        assert torch.isnan(item_emb).sum() == 0
        return user_emb, item_emb

    def bpr(self, user_embedding, item_embedding, u, i, j, user_embedding_ego=None, item_embedding_ego=None):
        user = user_embedding[u]
        pos_item = item_embedding[i]
        neg_item = item_embedding[j]
        assert torch.isnan(user).sum() == 0
        assert torch.isnan(pos_item).sum() == 0
        assert torch.isnan(neg_item).sum() == 0
        # -------- BPR loss

        # prediction_i = (user * pos_item).sum(dim=-1)
        # prediction_j = (user * neg_item).sum(dim=-1)
        # assert torch.isnan(prediction_i).sum() == 0
        # assert torch.isnan(prediction_j).sum() == 0

        # bpr_loss = -((prediction_i - prediction_j).sigmoid().log().mean())
        pos_scores = torch.mul(user, pos_item)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user, neg_item)
        neg_scores = torch.sum(neg_scores, dim=1)
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        if user_embedding_ego == None or item_embedding_ego == None:
            reg_loss = ((user ** 2).sum(dim=-1) + (pos_item ** 2 + neg_item ** 2).sum(dim=-1)).mean()
        else:
            userEmb0 = user_embedding_ego[u]
            posEmb0 = item_embedding_ego[i]
            negEmb0 = item_embedding_ego[j]
            reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                                  posEmb0.norm(2).pow(2) +
                                  negEmb0.norm(2).pow(2)) / float(len(u))

        return bpr_loss, reg_loss

    def forward(self, perturbed=False):

        # u_emb0 = self.self_gatingu(self.user_embedding.weight)
        # i_emb0 = self.self_gatingi(self.item_id_embedding.weight)

        # hyperedge dependencies constructing
        if self.v_feat is not None:
            iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper)
            uv_hyper = torch.mm(self.adj, iv_hyper)
            iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)
        if self.t_feat is not None:
            it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)
            ut_hyper = torch.mm(self.adj, it_hyper)
            it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)

        E_sta = self.cge()
        # E_sta = self.cge(u_emb0, i_emb0)

        if self.v_feat is not None and self.t_feat is not None:
            v_feats = self.mge(perturbed, 'v')
            t_feats = self.mge(perturbed, 't')

            self.image_linear = torch.nn.Linear(v_feats.shape[1], self.embedding_dim).to(self.device)
            self.text_linear = torch.nn.Linear(t_feats.shape[1], self.embedding_dim).to(self.device)

            mcl_embs = F.normalize(v_feats) + F.normalize(t_feats)
            # GHE: global hypergraph embedding
            user_indices = torch.arange(self.n_users, dtype=torch.long).cuda()
            user_tensor = self.user_embedding(user_indices).cuda()
            item_indices = torch.arange(self.n_items, dtype=torch.long).cuda()
            item_tensor = self.item_id_embedding(item_indices).cuda()

            embs = torch.cat([user_tensor, item_tensor], dim=0).cuda()
            E_dyn, _ = self.gtLayer(self.norm_adj, embs)

            E_Interest = self.alpha * F.normalize(E_dyn) + E_sta

            uv_hyper_embs, iv_hyper_embs = self.hgnnLayer(self.drop(iv_hyper), self.drop(uv_hyper),
                                                          E_sta[self.n_users:])
            ut_hyper_embs, it_hyper_embs = self.hgnnLayer(self.drop(it_hyper), self.drop(ut_hyper),
                                                          E_sta[self.n_users:])
            av_hyper_embs = torch.cat([uv_hyper_embs, iv_hyper_embs], dim=0)
            at_hyper_embs = torch.cat([ut_hyper_embs, it_hyper_embs], dim=0)
            E_h = av_hyper_embs + at_hyper_embs

            mm_emb = self.beta * F.normalize(E_h) + mcl_embs

            all_embs = E_Interest + mm_emb
        else:
            all_embs = E_sta
        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)
        return u_embs, i_embs, [uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs], [v_feats, t_feats]

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss

    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)

        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def calculate_loss(self, interaction, sign):
        ua_embeddings, ia_embeddings, hyper_embeddings, [image_emb, text_emb] = self.forward(sign)
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        [uv_embs, iv_embs, ut_embs, it_embs] = hyper_embeddings

        batch_hcl_loss = self.ssl_triple_loss(uv_embs[users], ut_embs[users], ut_embs) + self.ssl_triple_loss(
            iv_embs[pos_items], it_embs[pos_items], it_embs)

        if hasattr(self, 'warm_missing_item_index') and self.v_feat is not None and self.t_feat is not None:
            temp_item = torch.unique(pos_items)
            mask = torch.tensor([i.item() not in self.warm_missing_item_index for i in temp_item], device=self.device)
            unique_item = temp_item[mask]


        #     if unique_item.numel() > 0:
        #         align_loss = self.align(image_emb[unique_item], text_emb[unique_item]) * 0.01
        #     else:
        #         align_loss = torch.tensor(0.0, device=self.device)
        # else:
        #         align_loss = torch.tensor(0.0, device=self.device)
        align_loss = self.align(image_emb, text_emb) * 0.01


        mix_ration = [[self.P, self.P]]
        for i in range(1):
            lam_1, lam_2 = np.random.dirichlet([self.theta, self.theta])
            mix_ration.append([lam_1, lam_2])
            mix_ration.append([lam_2, lam_1])

        env_penalty = []
        env_reg = []
        for mix in mix_ration:
            env_user_emb, env_item_emb = self.get_env_emb(mix, 0, image_emb, text_emb)
            env_bpr_loss, env_reg_loss = self.bpr(env_user_emb, env_item_emb, users, pos_items, neg_items)
            env_penalty.append(env_bpr_loss)
            env_reg.append(env_reg_loss)


        env_penalty_loss = torch.mean(torch.stack(env_penalty)) * self.env_weight
        env_reg_loss = torch.mean(torch.stack(env_reg)) * self.env_weight

        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        loss = batch_bpr_loss + self.reg_weight * batch_reg_loss + self.cl_weight * batch_hcl_loss + align_loss * self.align_weight + env_penalty_loss + env_reg_loss

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, _, _ = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores


class HGNNLayer(nn.Module):
    def __init__(self, n_hyper_layer):
        super(HGNNLayer, self).__init__()
        self.h_layer = n_hyper_layer

    def forward(self, i_hyper, u_hyper, embeds):
        i_ret = embeds
        for _ in range(self.h_layer):
            lat = torch.mm(i_hyper.T, i_ret)
            i_ret = torch.mm(i_hyper, lat)
            u_ret = torch.mm(u_hyper, lat)
        return u_ret, i_ret


class MSE(torch.nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse = self.mse = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, embedding_1, embedding_2):
        return self.mse(embedding_1, embedding_2)