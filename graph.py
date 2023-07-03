from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
import scipy.sparse as sp

def get_adj(count, k=10, pca=50):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, metric="cosine", include_self=False)
    adj = A.toarray()  # 转换成ndarray
    adj = sp.coo_matrix(adj)
    return adj  # 返回的是ndarray

def dopca(X, dim=50):
    pcaten = PCA(n_components=dim)
    X_pca = pcaten.fit_transform(X)
    return X_pca
