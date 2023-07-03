import pandas
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.random.seed(0)
from torch import optim

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
from utils import EMA, set_requires_grad, init_weights, update_moving_average, loss_fn, repeat_1d_tensor, currentTime
import copy

from data import Dataset
from embedder import embedder
from utils import config2string, show_info
from embedder import Encoder
import faiss
from sklearn.cluster import KMeans

from torch_geometric.nn.models.autoencoder import InnerProductDecoder
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   remove_self_loops)
import time
EPS = 1e-15


class scAFGCC_ModelTrainer(embedder):

    def __init__(self, args):
        embedder.__init__(self, args)
        self._args = args
        self._init()
        self.config_str = config2string(self._args)
        print("\n[Config] {}\n".format(self.config_str))
        # self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))

    def _init(self):
        args = self._args
        self._task = args.task
        print("Downstream Task : {}".format(self._task))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self._device)
        self._dataset = Dataset(root=args.root, input_h5ad_path=args.dataset)
        args.num_centroids = self._dataset.num_centroids
        self._args.num_centroids = args.num_centroids
        self._loader = DataLoader(dataset=self._dataset)  # default batch_size=1
        layers = [self._dataset.data.x.shape[1]] + self.hidden_layers
        self._model = scAFGCC(layers, args).to(self._device)
        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay=1e-5)

    def train(self, dataset: str, df: pandas.DataFrame, start_memory):

        self.best_test_acc, self.best_dev_acc_ari, self.best_dev_acc_nmi, self.best_test_std, self.best_dev_std, self.best_epoch = 0, 0, 0, 0, 0, 0
        self.best_dev_accs = []
        print("Pre-training Start!")
        self._model.train()
        start = time.time()
        for epoch in range(50):
            for bc, batch_data in enumerate(self._loader):  # one batch
                batch_data.to(self._device)
                loss = self._model.pre_train(x=batch_data.x, edge_index=batch_data.edge_index,
                                             edge_weight=batch_data.edge_attr)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                st = '[Pre-training][{}][Epoch {}/{}] Loss: {:.4f}'.format(currentTime(), epoch, 50,
                                                                           loss.item())
                print(st)
            if (epoch) % self._args.eval_freq == 0:
                # 推理embeddings
                self.infer_embeddings(epoch)
                # 评估
                self.evaluate(self._task, epoch)


        # 初始化模型参数（重构完之后）
        self._model.initialize_teacher_by_student_decoder()
        self.infer_embeddings(0)
        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.args.num_centroids, n_init=20)  ###定义聚类方法是kmeans，类别数目=10,选20次最优的
        kmeans.fit_predict(self._embeddings.data.cpu().numpy())
        self._model.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        f_final = open("results/{}.txt".format(self._args.embedder), "a")

        # Start Model Training
        print("Training Start!")
        self._model.train()
        for epoch in range(self._args.epochs):
            for bc, batch_data in enumerate(self._loader):  # one batch
                batch_data.to(self._device)
                _, loss, ind, k = self._model(x=batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                                              neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                                              edge_weight=batch_data.edge_attr, epoch=epoch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._model.update_moving_average()

                st = '[Training][{}][Epoch {}/{}] Loss: {:.4f}'.format(currentTime(), epoch, self._args.epochs,
                                                                       loss.item())
                print(st)

            if (epoch) % self._args.eval_freq == 0:
                # 推理embeddings
                self.infer_embeddings(epoch)
                # 评估
                self.evaluate(self._task, epoch)
        total = time.time() - start
        end_memory = show_info()
        df.loc[df.shape[0]] = [dataset.split("_")[0], end_memory-start_memory,  total, self.best_dev_acc_ari, self.best_dev_acc_nmi]
        print("\nPre-training and Training Done!")
        print("[Final] {}".format(self.st_best))

        df.to_csv("results/{}.csv".format(self._args.embedder))
        f_final.write("{} -> {}\n".format(self.config_str, self.st_best))


class scAFGCC(nn.Module):
    def __init__(self, layer_config, args, **kwargs):
        super().__init__()
        # GCN encoder
        self.student_encoder = Encoder(layer_config=layer_config, dropout=args.dropout, **kwargs)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.epochs)
        self.neighbor = Neighbor(args)

        rep_dim = layer_config[-1]  # args.layers

        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid),
                                               nn.PReLU(), nn.Linear(args.pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)

        self.topk = args.topk
        # kl loss
        self.mu = nn.Parameter(torch.Tensor(args.num_centroids, rep_dim))  ###生成一个10*32的质心坐标，即每个簇的坐标都是32维
        self.alpha = 1.0
        # linear decoder
        self.decoder_nn = nn.Sequential(
            nn.Linear(in_features=rep_dim, out_features=args.pred_hid),
            nn.BatchNorm1d(args.pred_hid),
            nn.PReLU(),
            nn.Linear(in_features=args.pred_hid, out_features=layer_config[0]),
        )
        # innder product decoder
        self.inner_decoder = InnerProductDecoder()

    def initialize_teacher_by_student_decoder(self):
        for param_s, param_t in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            param_t.data.copy_(param_s.data)  # initialize

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x, y, edge_index, neighbor, edge_weight=None, epoch=None):

        student = self.student_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        pred = self.student_predictor(student)

        with torch.no_grad():
            teacher = self.teacher_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)

        if edge_weight == None:
            adj = torch.sparse.FloatTensor(neighbor[0], torch.ones_like(neighbor[0][0]), [x.shape[0], x.shape[0]])
        else:
            adj = torch.sparse.FloatTensor(neighbor[0], neighbor[1], [x.shape[0], x.shape[0]])

        ind, k = self.neighbor(adj, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk,
                               epoch)

        loss1 = loss_fn(pred[ind[0]], teacher[ind[1]].detach())
        loss2 = loss_fn(pred[ind[1]], teacher[ind[0]].detach())
        grl_loss = (loss1 + loss2).mean()
        # kl loss
        kl_loss = self.soft_cluster(student)
        # overall loss
        loss = 0.8 * grl_loss + kl_loss * 0.2

        return student, loss, ind, k

    def pre_train(self, x, edge_index, edge_weight=None):
        student = self.student_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        # reconstruct
        reconstructed_features = self.decoder_nn(student)
        recon_loss_linear = F.mse_loss(reconstructed_features, x)
        recon_loss_inner = self.recon_loss(student, edge_index)
        recon_loss = 0.8 * recon_loss_inner + recon_loss_linear * 0.2
        return recon_loss

    def soft_cluster(self, student):
        q = self.soft_assign(student)
        p = self.target_distribution(q).data
        kl_loss = self.cluster_loss(p, q)

        return kl_loss

    def soft_assign(self, z):  ####算z所属的软标签，即用学生分布度量z与self.mu的相似度
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2,
                                   dim=2) / self.alpha)  ###对于每一个cell都有一个10维度的q,因为属于每个簇的概率不同
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q  ###对于每一个cell都有一个10维度的q,因为属于每个簇的概率不同，所以q是268*10的矩阵

    def target_distribution(self, q):  ###
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(
                torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))  ####这里求了均值，所以后面会* len(inputs)

        kldloss = kld(p, q)
        return kldloss

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.inner_decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.inner_decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


class Neighbor(nn.Module):
    def __init__(self, args):
        super(Neighbor, self).__init__()
        self.device = args.device
        # 质心的数量
        self.num_centroids = args.num_centroids
        # kmeans执行次数
        self.num_kmeans = args.num_kmeans
        # 每次kmeans迭代次数
        self.clus_num_iters = args.clus_num_iters

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei

    # ind, k = self.neighbor(adj, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk, epoch)
    def forward(self, adj, student, teacher, top_k, epoch):
        n_data, d = student.shape
        similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
        similarity += torch.eye(n_data, device=self.device) * 10

        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
        tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(self.device)

        knn_neighbor = self.create_sparse(I_knn)
        locality = knn_neighbor * adj

        ncentroids = self.num_centroids
        niter = self.clus_num_iters

        pred_labels = []

        for seed in range(self.num_kmeans):
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=seed + 1234)
            kmeans.train(teacher.cpu().numpy())
            _, I_kmeans = kmeans.index.search(teacher.cpu().numpy(), 1)

            clust_labels = I_kmeans[:, 0]

            pred_labels.append(clust_labels)

        pred_labels = np.stack(pred_labels, axis=0)
        cluster_labels = torch.from_numpy(pred_labels).long()

        all_close_nei_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn,
                                                              I_knn.shape[1])

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        all_close_nei_in_back = all_close_nei_in_back.to(self.device)

        globality = self.create_sparse_revised(I_knn, all_close_nei_in_back)

        pos_ = locality + globality

        return pos_.coalesce()._indices(), I_knn.shape[1]

    def create_sparse(self, I):

        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])

        assert len(similar) == len(index)
        indices = torch.tensor([index, similar]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result

    def create_sparse_revised(self, I, all_close_nei_in_back):
        n_data, k = I.shape[0], I.shape[1]

        index = []
        similar = []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(int(j))
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_nei_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_nei_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [n_data, n_data])

        return result
