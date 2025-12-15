import multiprocessing
import pathlib
from collections import OrderedDict

import anndata
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import copy

import pynndescent
from scipy.cluster.hierarchy import linkage
from scipy.sparse import issparse
from scipy.stats import zscore
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, normalize, StandardScaler
from sklearn.utils.extmath import safe_sparse_dot

# from .cca import cca, lsi_cca
# from ..clustering.lsi import tf_idf



# get_cca_adata_list
def get_cca_adata_list(
    adatas,
    all_nfeatures=3000,
    single_nfeatures=3000
):
    """
    Select highly variable gene features for integration based on dispersion ranking and rank sum across multiple datasets.
    
    Parameters:
    - adatas: A list of AnnData objects
    - all_nfeatures: Total number of features to select after integration
    - single_nfeatures: Number of highly variable genes to select per dataset
    
    Returns:
    - adata_subs: A list of subset AnnData objects containing only the selected features
    """
    
    # 1. Convert raw data into new AnnData objects
    adata_raw_list = []
    for idx, adata in enumerate(adatas, start=1):
        adata_raw = adata.raw.to_adata()
        adata_raw.raw = adata_raw
        adata_raw_list.append(adata_raw)
    
    # 2. Find common genes across all datasets
    common_genes = set(adata_raw_list[0].var_names)
    for adata_raw in adata_raw_list[1:]:
        common_genes &= set(adata_raw.var_names)
    common_genes = pd.Index(list(common_genes))
    print(f"Number of common genes: {len(common_genes)}")
        
    # 3. Initialize gene_rank_df with common genes as rows and datasets as columns
    gene_rank_df = pd.DataFrame(index=common_genes)
    
    # 4. Perform highly variable gene selection and ranking for each dataset
    for idx, adata_raw in enumerate(adata_raw_list, start=1):
        dataset_name = f'adata{idx}_rank'
        sc.pp.highly_variable_genes(
            adata_raw,
            flavor='seurat',
            n_top_genes=single_nfeatures,
            inplace=True
        )
        hvgs = adata_raw.var.loc[common_genes]
        hvgs_sorted = hvgs.sort_values(by='dispersions_norm', ascending=False)
        hvgs_sorted = hvgs_sorted.reset_index()
        hvgs_sorted['rank'] = np.arange(1, len(hvgs_sorted) + 1)
        gene_rank_df[dataset_name] = hvgs_sorted.set_index('index')['rank']
    
    # 5. Compute the rank sum for each gene
    gene_rank_df['rank_sum'] = gene_rank_df.sum(axis=1)
    gene_rank_df = gene_rank_df.sort_values(by='rank_sum', ascending=True)
    # print(f"gene_rank_df.shape: {gene_rank_df.shape}")
    # print(f"gene_rank_df.head():\n{gene_rank_df.head()}")
    
    # 6. Select the top all_nfeatures genes with the smallest rank sum
    selected_features = gene_rank_df.nsmallest(all_nfeatures, 'rank_sum').index.tolist()
    
    # 7. Return subset objects containing only the selected features
    adata_subs = []
    for adata_raw in adata_raw_list:
        adata_sub = adata_raw[:, selected_features].copy()
        adata_subs.append(adata_sub)
    
    return adata_subs



################ tf_idf ################
def tf_idf(data, scale_factor=100000, idf=None):
    sparse_input = issparse(data)

    if idf is None:
        # add small value in case down sample creates empty feature
        _col_sum = data.sum(axis=0)
        if sparse_input:
            col_sum = _col_sum.A1.astype(np.float32) + 0.00001
        else:
            col_sum = _col_sum.ravel().astype(np.float32) + 0.00001
        idf = np.log(1 + data.shape[0] / col_sum).astype(np.float32)
    else:
        idf = idf.astype(np.float32)

    _row_sum = data.sum(axis=1)
    if sparse_input:
        row_sum = _row_sum.A1.astype(np.float32) + 0.00001
    else:
        row_sum = _row_sum.ravel().astype(np.float32) + 0.00001

    tf = data.astype(np.float32)

    if sparse_input:
        tf.data = tf.data / np.repeat(row_sum, tf.getnnz(axis=1))
        tf.data = np.log1p(np.multiply(tf.data, scale_factor, dtype="float32"))
        tf = tf.multiply(idf)
    else:
        tf = tf / row_sum[:, np.newaxis]
        tf = np.log1p(np.multiply(tf, scale_factor, dtype="float32"))
        tf = tf * idf
    return tf, idf





################ cca ################
def top_features_idx(data, n_features):
    """
    Select top features with the highest importance in CCs.

    Parameters
    ----------
    data
        data.shape = (n_cc, total_features)
    n_features
        number of features to select

    Returns
    -------
    features_idx : np.array
    """
    # data.shape = (n_cc, total_features)
    n_cc = data.shape[0]
    n_features_per_dim = n_features * 10 // n_cc
    n_features_per_dim = min(n_features_per_dim, data.shape[1] - 1)

    sample_range = np.arange(n_cc)[:, None]

    # get idx of n_features_per_dim features with the highest absolute loadings
    data = np.abs(data)
    idx = np.argpartition(-data, n_features_per_dim, axis=1)[:, :n_features_per_dim]
    # idx.shape = (n_cc, n_features_per_dim)

    # make sure the order of first n_features_per_dim is ordered by loadings
    idx = idx[sample_range, np.argsort(-data[sample_range, idx], axis=1)]

    for i in range(n_features // n_cc + 1, n_features_per_dim):
        features_idx = np.unique(idx[:, :i].flatten())
        if len(features_idx) > n_features:
            return features_idx
    else:
        features_idx = np.unique(idx[:, :n_features_per_dim].flatten())
        return features_idx


def cca(
    data1,
    data2,
    scale1=True,
    scale2=True,
    n_components=50,
    max_cc_cell=20000,
    chunk_size=50000,
    random_state=0,
    svd_algorithm="randomized",
    k_filter=None,
    n_features=200,
):
    np.random.seed(random_state)
    tf_data1, tf_data2, scaler1, scaler2 = downsample(
        data1=data1,
        data2=data2,
        todense=True,
        scale1=scale1,
        scale2=scale2,
        max_cc_cell=max_cc_cell,
        random_state=random_state,
    )

    # CCA decomposition
    model = TruncatedSVD(n_components=n_components, algorithm=svd_algorithm, random_state=random_state)
    tf_data2_t = tf_data2.T.copy()
    U = model.fit_transform(tf_data1.dot(tf_data2_t))

    # select dimensions with non-zero singular values
    sel_dim = model.singular_values_ != 0
    print("non zero dims", sel_dim.sum())

    V = model.components_[sel_dim].T
    U = U[:, sel_dim] / model.singular_values_[sel_dim]

    # compute ccv feature loading
    if k_filter:
        high_dim_feature = top_features_idx(
            np.concatenate([U, V], axis=0).T.dot(np.concatenate([tf_data1, tf_data2], axis=0)), n_features=n_features
        )
    else:
        high_dim_feature = None

    # transform CC
    if data2.shape[0] > max_cc_cell:
        V = []
        for chunk_start in np.arange(0, data2.shape[0], chunk_size):
            if issparse(data2):
                tmp = data2[chunk_start : (chunk_start + chunk_size)].toarray()
            else:
                tmp = data2[chunk_start : (chunk_start + chunk_size)]
            if scale2:
                tmp = scaler2.transform(tmp)
            V.append(np.dot(np.dot(U.T, tf_data1), tmp.T).T)
        V = np.concatenate(V, axis=0)
        V = V / model.singular_values_[sel_dim]

    if data1.shape[0] > max_cc_cell:
        U = []
        for chunk_start in np.arange(0, data1.shape[0], chunk_size):
            if issparse(data1):
                tmp = data1[chunk_start : (chunk_start + chunk_size)].toarray()
            else:
                tmp = data1[chunk_start : (chunk_start + chunk_size)]
            if scale1:
                tmp = scaler1.transform(tmp)
            U.append(np.dot(tmp, np.dot(model.components_[sel_dim], tf_data2).T))
        U = np.concatenate(U, axis=0)
        U = U / model.singular_values_[sel_dim]

    return U, V, high_dim_feature


def adata_cca(adata, group_col, separate_scale=True, n_components=50, random_state=42):
    groups = adata.obs[group_col].unique()
    if len(groups) != 2:
        raise ValueError(f"CCA only handle 2 groups, " f"adata.obs[{group_col}] has {len(groups)} different groups.")
    group_a, group_b = groups
    a = adata[adata.obs[group_col] == group_a, :].X
    b = adata[adata.obs[group_col] == group_b, :].X

    pc, loading, _ = cca(
        data1=a,
        data2=b,
        scale1=separate_scale,
        scale2=separate_scale,
        n_components=n_components,
        random_state=random_state,
    )
    total_cc = np.concatenate([pc, loading], axis=0)
    adata.obsm["X_cca"] = total_cc
    return


# def incremental_cca(a, b, max_chunk_size=10000, random_state=0):
#     """
#     Perform Incremental CCA by chunk dot product and IncrementalPCA
#
#     Parameters
#     ----------
#     a
#         dask.Array of dataset a
#     b
#         dask.Array of dataset b
#     max_chunk_size
#         Chunk size for Incremental fit and transform, the larger the better as long as MEM is enough
#     random_state
#
#     Returns
#     -------
#     Top CCA components
#     """
#     raise NotImplementedError
#     # TODO PC is wrong
#     pca = dIPCA(n_components=50,
#                 whiten=False,
#                 copy=True,
#                 batch_size=None,
#                 svd_solver='auto',
#                 iterated_power=0,
#                 random_state=random_state)
#
#     # partial fit
#     n_sample = a.shape[0]
#     n_chunks = n_sample // max_chunk_size + 1
#     chunk_size = int(n_sample / n_chunks) + 1
#     for chunk_start in range(0, n_sample, chunk_size):
#         print(chunk_start)
#         X_chunk = a[chunk_start:chunk_start + chunk_size, :].dot(b.T)
#         pca.partial_fit(X_chunk)
#
#     # transform
#     pcs = []
#     for chunk_start in range(0, n_sample, chunk_size):
#         print(chunk_start)
#         X_chunk = a[chunk_start:chunk_start + chunk_size, :].dot(b.T)
#         pc_chunk = pca.transform(X_chunk).compute()
#         pcs.append(pc_chunk)
#     pcs = np.concatenate(pcs)
#
#     # concatenate CCA
#     total_cc = np.concatenate([pcs, pca.components_.T])
#     return total_cc


def lsi_cca(
    data1,
    data2,
    scale_factor=100000,
    n_components=50,
    max_cc_cell=20000,
    chunk_size=50000,
    svd_algorithm="randomized",
    min_cov_filter=5,
    random_state=0,
):
    np.random.seed(random_state)

    # down sample data1 and data2 to run tf_idf and CCA
    if max_cc_cell < data1.shape[0]:
        sel1 = np.sort(np.random.choice(np.arange(data1.shape[0]), max_cc_cell, False))
        tf_data1 = data1[sel1, :]
    else:
        tf_data1 = data1
    if max_cc_cell < data2.shape[0]:
        sel2 = np.sort(np.random.choice(np.arange(data2.shape[0]), max_cc_cell, False))
        tf_data2 = data2[sel2, :]
    else:
        tf_data2 = data2

    # filter bin to make sure the min_cov_filter is satisfied
    col_sum1 = tf_data1.sum(axis=0).A1
    col_sum2 = tf_data2.sum(axis=0).A1
    # the same bin_filter will also be used
    # in the chunk transfer below
    bin_filter = np.logical_and(col_sum1 > min_cov_filter, col_sum2 > min_cov_filter)
    tf1, idf1 = tf_idf(tf_data1[:, bin_filter], scale_factor=scale_factor)
    tf2, idf2 = tf_idf(tf_data2[:, bin_filter], scale_factor=scale_factor)

    # CCA part
    model = TruncatedSVD(n_components=n_components, algorithm=svd_algorithm, random_state=0)
    tf = tf1.dot(tf2.T)
    U = model.fit_transform(tf)

    # select non-zero singular values
    # transform the whole dataset 2 to get V
    sel_dim = model.singular_values_ != 0
    nnz_singular_values = model.singular_values_[sel_dim]
    nnz_components = model.components_[sel_dim]
    if max_cc_cell > data2.shape[0]:
        V = nnz_components.T
    else:
        # use the safe_sparse_dot to avoid memory error
        # safe_sparse_dot take both sparse and dense matrix,
        # for dense matrix, it just uses normal numpy dot product
        V = np.concatenate(
            [
                safe_sparse_dot(
                    safe_sparse_dot(U.T[sel_dim], tf1),
                    tf_idf(
                        data2[chunk_start : (chunk_start + chunk_size)][:, bin_filter],
                        scale_factor=scale_factor,
                        idf=idf2,
                    )[
                        0
                    ].T,  # [0] is the tf
                ).T
                for chunk_start in np.arange(0, data2.shape[0], chunk_size)
            ],
            axis=0,
        )
        V = V / np.square(nnz_singular_values)

    # transform the whole dataset 1 to get U
    if max_cc_cell > data1.shape[0]:
        U = U[:, sel_dim] / nnz_singular_values
    else:
        U = np.concatenate(
            [
                safe_sparse_dot(
                    tf_idf(
                        data1[chunk_start : (chunk_start + chunk_size)][:, bin_filter],
                        scale_factor=scale_factor,
                        idf=idf1,
                    )[
                        0
                    ],  # [0] is the tf
                    safe_sparse_dot(nnz_components, tf2).T,
                )
                for chunk_start in np.arange(0, data1.shape[0], chunk_size)
            ],
            axis=0,
        )
        U = U / nnz_singular_values
    return U, V


class LSI:
    def __init__(
        self,
        scale_factor=100000,
        n_components=100,
        algorithm="arpack",
        random_state=0,
        idf=None,
        model=None,
    ):
        self.scale_factor = scale_factor
        if idf is not None:
            self.idf = idf.copy()
        if idf is not None:
            self.model = model
        else:
            self.model = TruncatedSVD(n_components=n_components, algorithm=algorithm, random_state=random_state)

    def fit(self, data):
        tf, idf = tf_idf(data, self.scale_factor)
        self.idf = idf.copy()
        n_rows, n_cols = tf.shape
        self.model.n_components = min(n_rows, n_cols, self.model.n_components)
        self.model.fit(tf)
        return self

    def fit_transform(self, data):
        tf, idf = tf_idf(data, self.scale_factor)
        self.idf = idf.copy()
        n_rows, n_cols = tf.shape
        self.model.n_components = min(n_rows, n_cols, self.model.n_components)
        tf_reduce = self.model.fit_transform(tf)
        return tf_reduce / self.model.singular_values_

    def transform(self, data, chunk_size=50000, scaler=None):
        tf_reduce = []
        for chunk_start in np.arange(0, data.shape[0], chunk_size):
            tf, _ = tf_idf(data[chunk_start : (chunk_start + chunk_size)], self.scale_factor, self.idf)
            tf_reduce.append(self.model.transform(tf))
        return np.concatenate(tf_reduce, axis=0) / self.model.singular_values_


class SVD:
    def __init__(
        self,
        n_components=100,
        algorithm="randomized",
        random_state=0,
    ):
        self.model = TruncatedSVD(n_components=n_components, algorithm=algorithm, random_state=random_state)

    def fit(self, data):
        self.model.fit(data)
        return self

    def fit_transform(self, data):
        return self.model.fit_transform(data)

    def transform(self, data, chunk_size=50000, scaler=None):
        tf_reduce = []
        for chunk_start in np.arange(0, data.shape[0], chunk_size):
            if issparse(data):
                tmp = data[chunk_start : (chunk_start + chunk_size)].toarray()
            else:
                tmp = data[chunk_start : (chunk_start + chunk_size)]
            if scaler:
                tmp = scaler.transform(tmp)
            tf_reduce.append(self.model.transform(tmp))
        return np.concatenate(tf_reduce, axis=0)


def downsample(data1, data2, scale1, scale2, todense, max_cc_cell=20000, random_state=0):
    scaler1, scaler2 = [None, None]
    np.random.seed(random_state)
    if data1.shape[0] > max_cc_cell:
        sel1 = np.random.choice(np.arange(data1.shape[0]), min(max_cc_cell, data1.shape[0]), False)
        tf1 = data1[sel1]
    else:
        tf1 = data1.copy()
    if todense:
        if issparse(tf1):
            tf1 = tf1.toarray()

    if data2.shape[0] > max_cc_cell:
        sel2 = np.random.choice(np.arange(data2.shape[0]), min(max_cc_cell, data2.shape[0]), False)
        tf2 = data2[sel2]
    else:
        tf2 = data2.copy()
    if todense:
        if issparse(tf2):
            tf2 = tf2.toarray()

    if scale1:
        scaler1 = StandardScaler()
        tf1 = scaler1.fit_transform(tf1)
    if scale2:
        scaler2 = StandardScaler()
        tf2 = scaler2.fit_transform(tf2)
    return tf1, tf2, scaler1, scaler2













################ seurat_class ################
CPU_COUNT = multiprocessing.cpu_count()


def find_neighbor(cc1, cc2, k, random_state=0, n_jobs=-1):
    """
    Find all four way of neighbors for two datasets.
    
    Parameters
    ----------
    cc1
        cc for dataset 1
    cc2
        cc for dataset 2
    k
        number of neighbors
    random_state
        random seed
    n_jobs
        number of jobs to run in parallel

    Returns
    -------
    11, 12, 21, 22 neighbor matrix in shape (n_cell, k)
    """
    index = pynndescent.NNDescent(
        cc1,
        metric="euclidean",
        n_neighbors=k + 1,
        random_state=random_state,
        parallel_batch_queries=True,
        n_jobs=n_jobs,
    )
    G11 = index.neighbor_graph[0][:, 1 : k + 1]
    G21 = index.query(cc2, k=k)[0]
    index = pynndescent.NNDescent(
        cc2,
        metric="euclidean",
        n_neighbors=k + 1,
        random_state=random_state,
        parallel_batch_queries=True,
        n_jobs=n_jobs,
    )
    G22 = index.neighbor_graph[0][:, 1 : k + 1]
    G12 = index.query(cc1, k=k)[0]
    return G11, G12, G21, G22


def find_mnn(G12, G21, kanchor):
    """Calculate mutual nearest neighbor for two datasets."""
    anchor = [[i, G12[i, j]] for i in range(G12.shape[0]) for j in range(kanchor) if (i in G21[G12[i, j], :kanchor])]
    return np.array(anchor)


def min_max(tmp, q_left=1, q_right=90):
    """Normalize to q_left, q_right quantile to 0, 1, and cap extreme values."""
    tmin, tmax = np.percentile(tmp, [q_left, q_right])
    tmp = (tmp - tmin) / (tmax - tmin)
    tmp[tmp > 1] = 1
    tmp[tmp < 0] = 0
    return tmp


def filter_anchor(
    anchor,
    adata_ref=None,
    adata_qry=None,
    scale_ref=False,
    scale_qry=False,
    high_dim_feature=None,
    k_filter=200,
    random_state=0,
    n_jobs=-1,
):
    """
    Check if an anchor is still an anchor when only using the high_dim_features to construct KNN graph.

    If not, remove the anchor.
    """
    if issparse(adata_ref.X):
        ref_data = adata_ref.X[:, high_dim_feature].toarray()
    else:
        ref_data = adata_ref.X[:, high_dim_feature].copy()
    if scale_ref:
        ref_data = zscore(ref_data, axis=0)
    ref_data = normalize(ref_data, axis=1)

    if issparse(adata_qry.X):
        qry_data = adata_qry.X[:, high_dim_feature].toarray()
    else:
        qry_data = adata_qry.X[:, high_dim_feature].copy()
    if scale_qry:
        qry_data = zscore(qry_data, axis=0)
    qry_data = normalize(qry_data, axis=1)

    index = pynndescent.NNDescent(
        ref_data,
        metric="euclidean",
        n_neighbors=k_filter,
        random_state=random_state,
        parallel_batch_queries=True,
        n_jobs=n_jobs,
    )
    G = index.query(qry_data, k=k_filter)[0]
    input_anchors = anchor.shape[0]
    anchor = np.array([xx for xx in anchor if (xx[0] in G[xx[1]])])
    print(f"Anchor selected with high CC feature graph: {anchor.shape[0]} / {input_anchors}")
    return anchor


def score_anchor(anchor, G11, G12, G21, G22, k_score=30, Gp1=None, Gp2=None, k_local=50):
    """
    Score the anchor by the number of shared neighbors.

    Parameters
    ----------
    anchor
        anchor in shape (n_anchor, 2)
    G11
        neighbor graph of dataset 1
    G12
        neighbor graph of dataset 1 to 2
    G21
        neighbor graph of dataset 2 to 1
    G22
        neighbor graph of dataset 2
    k_score
        number of neighbors to score the anchor
    Gp1
        Intra-dataset1 kNN graph
    Gp2
        Intra-dataset2 kNN graph
    k_local
        number of neighbors to calculate the local score

    Returns
    -------
    anchor with score in shape (n_anchor, 3): pd.DataFrame
    """
    tmp = [
        len(set(G11[x, :k_score]).intersection(G21[y, :k_score]))
        + len(set(G12[x, :k_score]).intersection(G22[y, :k_score]))
        for x, y in anchor
    ]
    anchor_df = pd.DataFrame(anchor, columns=["x1", "x2"])
    anchor_df["score"] = min_max(tmp)

    if k_local:
        # if k_local is not None, then use local KNN to adjust the score
        share_nn = np.array([len(set(Gp1[i]).intersection(G11[i, :k_local])) for i in range(len(Gp1))])
        tmp = [share_nn[xx] for xx in anchor_df["x1"].values]
        anchor_df["score_local1"] = min_max(tmp)

        share_nn = np.array([len(set(Gp2[i]).intersection(G22[i, :k_local])) for i in range(len(Gp2))])
        tmp = [share_nn[xx] for xx in anchor_df["x2"].values]
        anchor_df["score_local2"] = min_max(tmp)

        anchor_df["score"] = anchor_df["score"] * anchor_df["score_local1"] * anchor_df["score_local2"]
    return anchor_df


def find_order(dist, ncell):
    """Use dendrogram to find the order of dataset pairs."""
    D = linkage(1 / dist, method="average")
    node_dict = {i: [i] for i in range(len(ncell))}
    alignment = []
    for xx in D[:, :2].astype(int):
        if ncell[xx[0]] < ncell[xx[1]]:
            xx = xx[::-1]
        alignment.append([node_dict[xx[0]], node_dict[xx[1]]])
        node_dict[len(ncell)] = node_dict[xx[0]] + node_dict[xx[1]]
        ncell.append(ncell[xx[0]] + ncell[xx[1]])
    return alignment


class SeuratIntegration:
    """Main class for Seurat integration."""

    def __init__(self, n_jobs=-1, random_state=0):
        self.n_jobs = n_jobs

        # intra-dataset KNN graph
        self.k_local = None
        self.key_local = None
        self.local_knn = []

        self.adata_dict = OrderedDict()
        self.n_dataset = 0
        self.n_cells = []
        self.alignments = None
        self.all_pairs = np.array([])
        self._get_all_pairs()

        self.anchor = {}
        self.mutual_knn = {}
        self.raw_anchor = {}
        self.label_transfer_results = {}

        self.random_state = random_state

    def _calculate_local_knn(self):
        """
        Calculate local kNN graph for each dataset.

        If klocal is provided, we calculate the local knn graph to
        evaluate whether the anchor preserves local structure within the dataset.
        One can use a different obsm with key_local to compute knn for each dataset.
        """
        if self.k_local is not None:
            print("Find neighbors within datasets")
            for adata in self.adata_dict.values():
                index = pynndescent.NNDescent(
                    adata.obsm[self.key_local],
                    metric="euclidean",
                    n_neighbors=self.k_local + 1,
                    random_state=self.random_state,
                    parallel_batch_queries=True,
                    n_jobs=self.n_jobs,
                )
                self.local_knn.append(index.neighbor_graph[0][:, 1:])
        else:
            self.local_knn = [None for _ in self.adata_dict.values()]

    def _get_all_pairs(self):
        if self.alignments is not None:
            all_pairs = []
            for pair in self.alignments:
                for xx in pair[0]:
                    for yy in pair[1]:
                        if xx < yy:
                            all_pairs.append(f"{xx}-{yy}")
                        else:
                            all_pairs.append(f"{yy}-{xx}")
            self.all_pairs = np.unique(all_pairs)
        else:
            self.all_pairs = np.array([])

    def _prepare_matrix(self, i, j, key_anchor):
        adata_dict = self.adata_dict
        adata1 = adata_dict[i]
        adata2 = adata_dict[j]

        if key_anchor == "X":
            # in case the adata var is not in the same order
            # select and order the var to make sure it is matched
            if (adata1.shape[1] != adata2.shape[1]) or ((adata1.var.index == adata2.var.index).sum() < adata1.shape[1]):
                sel_b = adata1.var.index & adata2.var.index
                U = adata1[:, sel_b].X.copy()
                V = adata2[:, sel_b].X.copy()
            else:
                U = adata1.X.copy()
                V = adata2.X.copy()
        else:
            U = adata1.obsm[key_anchor]
            V = adata2.obsm[key_anchor]

        return U, V

    def _calculate_mutual_knn_and_raw_anchors(self, i, j, U, V, k, k_anchor):
        """
        Calculate the mutual knn graph and raw anchors.

        The results are saved to self.mutual_knn and self.raw_anchor.
        """
        G11, G12, G21, G22 = find_neighbor(U, V, k=k, n_jobs=self.n_jobs)
        raw_anchors = find_mnn(G12, G21, k_anchor)
        self.mutual_knn[(i, j)] = (G11, G12, G21, G22)
        self.raw_anchor[(i, j)] = raw_anchors
        return G11, G12, G21, G22, raw_anchors

    def _pairwise_find_anchor(
        self,
        i,  # Index of the dataset to be paired
        i_sel,  # Optional cell selection indices
        j,  # Index of the dataset to be paired
        j_sel,  # Optional cell selection indices
        dim_red,    # Dimensionality reduction method
        key_anchor, # Data key used for anchor calculation
        svd_algorithm,
        scale1,
        scale2,
        k_anchor,   # Number of k-nearest neighbors used for finding anchors
        k_local,    # Number of k-nearest neighbors used for local scoring
        k_score,    # Number of k-nearest neighbors used for scoring anchors
        ncc,    # Number of dimensions after reduction
        max_cc_cell,    # Maximum number of cells processed by CCA
        k_filter,   # Number of k-nearest neighbors used for anchor filtering
        n_features, # Number of high-dimensional features used for anchor filtering
        chunk_size,
        random_state,
        signorm,
    ):
        """Pairwise anchor between two datasets."""
        adata1 = self.adata_dict[i]
        adata2 = self.adata_dict[j]

        min_sample = min(adata1.shape[0], adata2.shape[0])

        if i_sel is not None:
            adata1 = adata1[i_sel, :]
        if j_sel is not None:
            adata2 = adata2[j_sel, :]

        if dim_red in ("cca", "pca", "lsi", "lsi-cca"):
            # 1. prepare input matrix for CCA
            # print("———————————————————————————————————"*2)
            print("1. prepare input matrix for CCA")
            U, V = self._prepare_matrix(i, j, key_anchor=key_anchor)
            # print(f"U.shape:{U.shape}")
            # print(f"V.shape:{V.shape}")
            # print(f"V:{V}")

            # 2. run cca between datasets
            # print("———————————————————————————————————"*2)
            print("2. run cca between datasets")
            if dim_red in ("cca", "pca"):
                print("Run CCA")
                U, V, high_dim_feature = cca(
                    data1=U,
                    data2=V,
                    scale1=scale1,
                    scale2=scale2,
                    n_components=ncc,
                    max_cc_cell=max_cc_cell,
                    k_filter=k_filter,
                    n_features=n_features,
                    chunk_size=chunk_size,
                    svd_algorithm=svd_algorithm,
                    random_state=random_state,
                )
            elif dim_red in ("lsi", "lsi-cca"):
                print("Run LSI-CCA")
                U, V = lsi_cca(
                    data1=U,
                    data2=V,
                    scale_factor=100000,
                    n_components=ncc,
                    max_cc_cell=max_cc_cell,
                    chunk_size=chunk_size,
                    svd_algorithm=svd_algorithm,
                    min_cov_filter=5,
                    random_state=random_state,
                )
                high_dim_feature = None
            else:
                raise ValueError(f"Dimension reduction method {dim_red} is not supported.")
            # print(f"U.shape:{U.shape}")
            # print(f"V.shape:{V.shape}")
            # if high_dim_feature is not None:
            #     print(f"len(high_dim_feature):{len(high_dim_feature)}")
            # else:
            #     print(f"high_dim_feature:{high_dim_feature}")

            # 3. normalize CCV per sample/row
            # print("———————————————————————————————————"*2)
            print("3. normalize CCV per sample/row")
            U = normalize(U, axis=1)
            V = normalize(V, axis=1)
            # print(f"U.shape:{U.shape}")
            # print(f"V.shape:{V.shape}")

            # 4. find MNN of U and V to find anchors
            # print("———————————————————————————————————"*2)
            print("4. find MNN of U and V to find anchors")
            _k = max(_temp for _temp in [k_anchor, k_local, k_score] if _temp is not None)
            _k = min(min_sample - 2, _k)
            print(f"Find Anchors using k={_k}")
            G11, G12, G21, G22, raw_anchors = self._calculate_mutual_knn_and_raw_anchors(
                i=i, j=j, U=U, V=V, k=_k, k_anchor=k_anchor
            )

            # 5. filter anchors by high dimensional neighbors
            # print("———————————————————————————————————"*2)
            print("5. filter anchors by high dimensional neighbors")
            if k_filter is not None and high_dim_feature is not None:
                # compute ccv feature loading
                if self.n_cells[i] >= self.n_cells[j]:
                    raw_anchors = filter_anchor(
                        anchor=raw_anchors,
                        adata_ref=adata1,
                        adata_qry=adata2,
                        scale_ref=scale1,
                        scale_qry=scale2,
                        high_dim_feature=high_dim_feature,
                        k_filter=k_filter,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    )
                else:
                    raw_anchors = filter_anchor(
                        anchor=raw_anchors[:, ::-1],
                        adata_ref=adata2,
                        adata_qry=adata1,
                        scale_ref=scale2,
                        scale_qry=scale1,
                        high_dim_feature=high_dim_feature,
                        k_filter=k_filter,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    )[:, ::-1]
                    
        elif dim_red in ("rpca", "rlsi"):

            adata1, adata2 = adata1.X, adata2.X
            k = max(i for i in [k_anchor, k_local, k_score, 50] if i is not None)
            if dim_red == "rpca":
                print("Run rPCA")
                model = SVD(n_components=ncc, random_state=random_state)
            elif dim_red == "rlsi":
                print("Run rLSI")
                model = LSI(n_components=ncc, random_state=random_state)
            else:
                raise ValueError(f"Dimension reduction method {dim_red} is not supported.")
            tf1, tf2, scaler1, scaler2 = downsample(
                adata1,
                adata2,
                todense=True if dim_red == "rpca" else False,
                scale1=scale1,
                scale2=scale2,
                max_cc_cell=max_cc_cell,
            )

            # project adata2 to adata1
            model.fit(tf1)
            U = model.transform(adata1, chunk_size=chunk_size, scaler=scaler1)
            V = model.transform(adata2, chunk_size=chunk_size, scaler=scaler2)
            if (dim_red == "pca") and signorm:
                U = U / model.model.singular_values_
                V = V / model.model.singular_values_
            index = pynndescent.NNDescent(
                U, metric="euclidean", n_neighbors=k + 1, random_state=random_state, n_jobs=-1
            )
            G11 = index.neighbor_graph[0][:, 1 : k + 1]
            G21 = index.query(V, k=k)[0]

            # project adata1 to adata2
            model.fit(tf2)
            U = model.transform(adata1, chunk_size=chunk_size, scaler=scaler1)
            V = model.transform(adata2, chunk_size=chunk_size, scaler=scaler2)
            if (dim_red == "pca") and signorm:
                U = U / model.model.singular_values_
                V = V / model.model.singular_values_
            index = pynndescent.NNDescent(
                V, metric="euclidean", n_neighbors=k + 1, random_state=random_state, n_jobs=-1
            )
            G22 = index.neighbor_graph[0][:, 1 : k + 1]
            G12 = index.query(U, k=k)[0]

            raw_anchors = find_mnn(G12, G21, k_anchor)

        elif dim_red == "precomputed":
            # no CCA or PCA, but directly use `key_anchor` data
            # 1. get U,V
            print("———————————————————————————————————"*2)
            print("1. get U,V")
            U, V = self._prepare_matrix(i, j, key_anchor=key_anchor)
            print(f"U.shape:{U.shape}")
            print(f"V.shape:{V.shape}")

            # 4. find MNN of U and V to find anchors
            print("———————————————————————————————————"*2)
            print("4. find MNN of U and V to find anchors")
            # 直接寻找MNN
            _k = max(_temp for _temp in [k_anchor, k_local, k_score] if _temp is not None)
            _k = min(min_sample - 2, _k)
            print(f"Find Anchors using k={_k}")
            G11, G12, G21, G22, raw_anchors = self._calculate_mutual_knn_and_raw_anchors(i, j, U, V, k=_k, k_anchor=k_anchor)

            # 5. filter anchors by high dimensional neighbors
            if k_filter is not None and high_dim_feature is not None:
                print("———————————————————————————————————"*2)
                print("5. filter anchors by high dimensional neighbors")
                # compute ccv feature loading
                if self.n_cells[i] >= self.n_cells[j]:
                    raw_anchors = filter_anchor(
                        anchor=raw_anchors,
                        adata_ref=adata1,
                        adata_qry=adata2,
                        scale_ref=scale1,
                        scale_qry=scale2,
                        high_dim_feature=high_dim_feature,
                        k_filter=k_filter,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    )
                else:
                    raw_anchors = filter_anchor(
                        anchor=raw_anchors[:, ::-1],
                        adata_ref=adata2,
                        adata_qry=adata1,
                        scale_ref=scale2,
                        scale_qry=scale1,
                        high_dim_feature=high_dim_feature,
                        k_filter=k_filter,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    )[:, ::-1]
        else:
            raise ValueError(f"Dimension reduction method {dim_red} is not supported.")

        # 6. score anchors with snn and local structure preservation
        print("6. score anchors with snn and local structure preservation")
        # print("———————————————————————————————————"*2)
        anchor_df = score_anchor(
            anchor=raw_anchors,
            G11=G11,
            G12=G12,
            G21=G21,
            G22=G22,
            k_score=k_score,
            k_local=k_local,
            Gp1=self.local_knn[i],
            Gp2=self.local_knn[j],
        )
        self.U=U
        self.V=V
        return anchor_df

    def find_anchor(
        self,
        adata_list,
        adata_names=None,
        k_local=None,   # k for local knn
        key_local="X_pca",  # key for local knn
        key_anchor="X", # key for cca
        dim_red="pca",
        svd_algorithm="randomized",
        scale1=True,
        scale2=True,
        scale_list=None,
        k_filter=None,  # k for anchor filter
        n_features=200, # high dim number
        n_components=None,  # cca dim number
        max_cc_cells=50000, # max cca cells
        chunk_size=50000,   # chunk_size
        k_anchor=5, # k for find anchors
        k_score=30, # k for scoring anchors
        alignments=None,    # alignment order
        random_state=0,
        signorm=True,   # whether regularize pca
        key_match=None, # None: Directly search for anchors between the entire datasets; 
        # Non-None: Group based on a specific key and search for anchors within each group separately.
    ):
        """Find anchors for each dataset pair."""
        valid_dim_red_name = ["pca", "cca", "lsi", "lsi-cca", "rpca", "rlsi",'precomputed']
        if dim_red not in valid_dim_red_name:
            raise ValueError(f"Dimension reduction method {dim_red} is not supported.")

        if adata_names is None:
            adata_names = list(range(len(adata_list)))
        try:
            assert len(adata_names) == len(adata_list)
        except AssertionError:
            print("length of adata_names does not match length of adata_list.")

        self.adata_dict = {k: v for k, v in zip(adata_names, adata_list)}
        self.n_dataset = len(adata_list)
        self.n_cells = [adata.shape[0] for adata in adata_list]

        # intra-dataset KNN for scoring the anchors
        self.k_local = k_local
        self.key_local = key_local
        ## 1. _calculate_local_knn
        self._calculate_local_knn()

        # alignments and all_pairs
        self.alignments = alignments
        ## 2. _get_all_pairs
        self._get_all_pairs()
        
        ## 3. _pairwise_find_anchor
        print("Find anchors across datasets.")
        for i in range(self.n_dataset - 1):
            for j in range(i + 1, self.n_dataset):
                if scale_list is not None:
                    scale1 = scale_list[i]
                    scale2 = scale_list[j]
                    print("Get scale1 and scale2 from scale_list")
                    print(f"dataset {i} scale: {scale1}")
                    print(f"dataset {j} scale: {scale2}")

                if key_match is None:
                    anchor_df = self._pairwise_find_anchor(
                        i=i,
                        i_sel=None,
                        j=j,
                        j_sel=None,
                        dim_red=dim_red,
                        key_anchor=key_anchor,
                        svd_algorithm=svd_algorithm,
                        scale1=scale1,
                        scale2=scale2,
                        k_anchor=k_anchor,
                        k_local=k_local,
                        k_score=k_score,
                        ncc=n_components,
                        max_cc_cell=max_cc_cells,
                        k_filter=k_filter,
                        n_features=n_features,
                        chunk_size=chunk_size,
                        random_state=random_state,
                        signorm=signorm,
                    )
                else:
                    tissue = [xx.obs[key_match].unique() for xx in adata_list]
                    sharet = list(set(tissue[i]).intersection(tissue[j]))
                    if len(sharet) > 0:
                        anchor_df_list = []
                        for t in sharet:
                            print(t)
                            adata1 = adata_list[i].copy()
                            adata2 = adata_list[j].copy()

                            idx1 = np.where(adata1.obs[key_match] == t)[0]
                            idx2 = np.where(adata2.obs[key_match] == t)[0]
                            tmp = self._pairwise_find_anchor(
                                i=i,
                                i_sel=idx1,
                                j=j,
                                j_sel=idx2,
                                dim_red=dim_red,
                                key_anchor=key_anchor,
                                svd_algorithm=svd_algorithm,
                                scale1=scale1,
                                scale2=scale2,
                                k_anchor=k_anchor,
                                k_local=k_local,
                                k_score=k_score,
                                ncc=n_components,
                                max_cc_cell=max_cc_cells,
                                k_filter=k_filter,
                                n_features=n_features,
                                chunk_size=chunk_size,
                                random_state=random_state,
                                signorm=signorm,
                            )
                            tmp["x1"] = idx1[tmp["x1"].values]
                            tmp["x2"] = idx2[tmp["x2"].values]
                            anchor_df_list.append(tmp)
                        anchor_df = pd.concat(anchor_df_list, axis=0)
                    else:
                        anchor_df = self._pairwise_find_anchor(
                            i=i,
                            i_sel=None,
                            j=j,
                            j_sel=None,
                            dim_red="rpca",
                            key_anchor=key_anchor,
                            svd_algorithm=svd_algorithm,
                            scale1=scale1,
                            scale2=scale2,
                            k_anchor=k_anchor,
                            k_local=k_local,
                            k_score=k_score,
                            ncc=n_components,
                            max_cc_cell=max_cc_cells,
                            k_filter=k_filter,
                            n_features=n_features,
                            chunk_size=chunk_size,
                            random_state=random_state,
                            signorm=signorm,
                        )

                # save anchors
                self.anchor[(i, j)] = anchor_df.copy()
                # print(f"Identified {len(self.anchor[i, j])} anchors between datasets {i} and {j}.")
        return

    def find_nearest_anchor(
        self, data, data_qry, ref, qry, key_correct="X_pca", npc=30, k_weight=100, sd=1, random_state=0
    ):
        """Find the nearest anchors for each cell in data."""
        print("Initialize")
        cum_ref, cum_qry = [0], [0]
        for xx in ref:
            cum_ref.append(cum_ref[-1] + data[xx].shape[0])
        for xx in qry:
            cum_qry.append(cum_qry[-1] + data[xx].shape[0])

        anchor = []
        for i, xx in enumerate(ref):
            for j, yy in enumerate(qry):
                if xx < yy:
                    tmp = self.anchor[(xx, yy)].copy()
                else:
                    tmp = self.anchor[(yy, xx)].copy()
                    tmp[["x1", "x2"]] = tmp[["x2", "x1"]]
                tmp["x1"] += cum_ref[i]
                tmp["x2"] += cum_qry[j]
                anchor.append(tmp)
        anchor = pd.concat(anchor)
        score = anchor["score"].values
        anchor = anchor[["x1", "x2"]].values

        if key_correct == "X":
            model = PCA(n_components=npc, svd_solver="arpack", random_state=random_state)
            reduce_qry = model.fit_transform(data_qry)
        else:
            reduce_qry = data_qry[:, :npc]

        print("Find nearest anchors", end=". ")
        index = pynndescent.NNDescent(
            reduce_qry[anchor[:, 1]],
            metric="euclidean",
            n_neighbors=k_weight,
            random_state=random_state,
            parallel_batch_queries=True,
            n_jobs=self.n_jobs,
        )
        k_weight = min(k_weight, anchor.shape[0] - 5)
        k_weight = max(5, k_weight)
        print("k_weight: ", k_weight, end="\n")
        G, D = index.query(reduce_qry, k=k_weight)

        print("Normalize graph")
        cell_filter = D[:, -1] == 0
        D = (1 - D / D[:, -1][:, None]) * score[G]
        D[cell_filter] = score[G[cell_filter]]
        D = 1 - np.exp(-D * (sd**2) / 4)
        D = D / (np.sum(D, axis=1) + 1e-6)[:, None]
        return anchor, G, D, cum_qry

    def transform(
        self,
        data,
        ref,
        qry,
        key_correct,
        npc=30,
        k_weight=100,
        sd=1,
        chunk_size=50000,
        random_state=0,
        row_normalize=True,
    ):
        """Transform query data to reference space."""
        data_ref = np.concatenate(data[ref])
        data_qry = np.concatenate(data[qry])

        anchor, G, D, cum_qry = self.find_nearest_anchor(
            data=data,
            data_qry=data_qry,
            key_correct=key_correct,
            ref=ref,
            qry=qry,
            npc=npc,
            k_weight=k_weight,
            sd=sd,
            random_state=random_state,
        )

        print("Transform data")
        bias = data_ref[anchor[:, 0]] - data_qry[anchor[:, 1]]
        data_prj = np.zeros(data_qry.shape)

        for chunk_start in np.arange(0, data_prj.shape[0], chunk_size):
            data_prj[chunk_start : (chunk_start + chunk_size)] = data_qry[chunk_start : (chunk_start + chunk_size)] + (
                D[chunk_start : (chunk_start + chunk_size), :, None] * bias[G[chunk_start : (chunk_start + chunk_size)]]
            ).sum(axis=1)
        for i, xx in enumerate(qry):
            _data = data_prj[cum_qry[i] : cum_qry[i + 1]]
            if row_normalize:
                _data = normalize(_data, axis=1)
            data[xx] = _data
        return data

    def integrate(self, key_correct, row_normalize=True, n_components=30, k_weight=100, sd=1, alignments=None):
        """Integrate datasets by transform data matrices from query to reference data using the MNN information."""
        if alignments is not None:
            self.alignments = alignments

        # find order of pairwise dataset merging with hierarchical clustering
        if self.alignments is None:
            dist = []
            for i in range(self.n_dataset - 1):
                for j in range(i + 1, self.n_dataset):
                    dist.append(len(self.anchor[(i, j)]) / min([self.n_cells[i], self.n_cells[j]]))
            self.alignments = find_order(np.array(dist), self.n_cells)
            print(f"Alignments: {self.alignments}")

        print("Merge datasets")
        adata_list = list(self.adata_dict.values())

        # initialize corrected with original data
        if key_correct == "X":
            # correct the original feature matrix
            corrected = [adata_list[i].X.copy() for i in range(self.n_dataset)]
        else:
            # correct dimensionality reduced matrix only
            corrected = [normalize(adata_list[i].obsm[key_correct], axis=1) for i in range(self.n_dataset)]

        for xx in self.alignments:
            print(xx)
            corrected = self.transform(
                data=np.array(corrected, dtype="object"),
                ref=xx[0],
                qry=xx[1],
                npc=n_components,
                k_weight=k_weight,
                sd=sd,
                random_state=self.random_state,
                row_normalize=row_normalize,
                key_correct=key_correct,
            )
        return corrected

    def label_transfer(
        self,
        ref,
        qry,
        categorical_key=None,
        continuous_key=None,
        key_dist="X_pca",
        k_weight=100,
        npc=30,
        sd=1,
        chunk_size=50000,
        random_state=0,
    ):
        """Transfer labels from query to reference space."""
        adata_list = list(self.adata_dict.values())

        data_qry = np.concatenate([normalize(adata_list[i].obsm[key_dist], axis=1) for i in qry])
        data_qry_index = np.concatenate([adata_list[i].obs_names for i in qry])

        anchor, G, D, cum_qry = self.find_nearest_anchor(
            data=adata_list,
            data_qry=data_qry,
            ref=ref,
            qry=qry,
            npc=npc,
            k_weight=k_weight,
            key_correct=key_dist,
            sd=sd,
            random_state=random_state,
        )
        print("Label transfer")
        label_ref = []
        columns = []
        cat_counts = []

        if categorical_key is None:
            categorical_key = []
        if continuous_key is None:
            continuous_key = []
        if len(categorical_key) == 0 and len(continuous_key) == 0:
            raise ValueError("No categorical or continuous key specified.")

        if len(categorical_key) > 0:
            tmp = pd.concat([adata_list[i].obs[categorical_key] for i in ref], axis=0)
            enc = OneHotEncoder()
            label_ref.append(enc.fit_transform(tmp[categorical_key].values.astype(np.str_)).toarray())
            # add categorical key to make sure col is unique
            columns += enc.categories_
            # enc.categories_ are a list of arrays, each array are categories in that categorical_key
            cat_counts += [cats.size for cats in enc.categories_]

        if len(continuous_key) > 0:
            tmp = pd.concat([adata_list[i].obs[continuous_key] for i in ref], axis=0)
            label_ref.append(tmp[continuous_key].values)
            columns += [[xx] for xx in continuous_key]
            cat_counts += [1 for _ in continuous_key]

        label_ref = np.concatenate(label_ref, axis=1)
        label_qry = np.zeros((data_qry.shape[0], label_ref.shape[1]))

        bias = label_ref[anchor[:, 0]]
        for chunk_start in np.arange(0, label_qry.shape[0], chunk_size):
            label_qry[chunk_start : (chunk_start + chunk_size)] = (
                D[chunk_start : (chunk_start + chunk_size), :, None] * bias[G[chunk_start : (chunk_start + chunk_size)]]
            ).sum(axis=1)

        all_column_names = np.concatenate(columns)  # these column names might be duplicated
        all_column_variables = np.repeat(categorical_key + continuous_key, cat_counts)
        label_qry = pd.DataFrame(label_qry, index=data_qry_index, columns=all_column_names)
        result = {}
        for key in categorical_key + continuous_key:
            result[key] = label_qry.iloc[:, all_column_variables == key]
        return result

    def save(self, output_path, save_local_knn=False, save_raw_anchor=False, save_mutual_knn=False, save_adata=False):
        """Save the model and results to disk."""
        # save each adata in a separate dir
        output_path = pathlib.Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        if save_adata:
            # save adata and clear the self.adata_dict
            adata_dir = output_path / "adata"
            adata_dir.mkdir(exist_ok=True)
            with open(f"{adata_dir}/order.txt", "w") as f:
                for k, v in self.adata_dict.items():
                    for col, val in v.obs.items():
                        if val.dtype == "O":
                            v.obs[col] = val.fillna("nan").astype(str)
                        elif val.dtype == "category":
                            v.obs[col] = val.fillna("nan").astype(str)
                        else:
                            pass
                    v.write_h5ad(f"{adata_dir}/{k}.h5ad")
                    f.write(f"{k}\n")

        # clear the adata in integrator
        self.adata_dict = {}

        if not save_local_knn:
            self.local_knn = []
        if not save_raw_anchor:
            self.raw_anchor = {}
        if not save_mutual_knn:
            self.mutual_knn = {}

        joblib.dump(self, f"{output_path}/model.lib")
        return

    @classmethod
    def load(cls, input_path):
        """Load integrator from file."""
        adata_dir = f"{input_path}/adata"
        model_path = f"{input_path}/model.lib"

        obj = joblib.load(model_path)

        orders = pd.read_csv(f"{adata_dir}/order.txt", header=None, index_col=0).index
        adata_dict = OrderedDict()
        for k in orders:
            adata_path = f"{adata_dir}/{k}.h5ad"
            if pathlib.Path(adata_path).exists():
                adata_dict[k] = anndata.read_h5ad(f"{adata_dir}/{k}.h5ad")
        obj.adata_dict = adata_dict
        return obj

    @classmethod
    def save_transfer_results_to_adata(cls, adata, transfer_results, new_label_suffix="_transfer"):
        """Save transfer results to adata."""
        for key, df in transfer_results.items():
            adata.obs[key + new_label_suffix] = adata.obs[key].copy()
            adata.obs.loc[df.index, key + new_label_suffix] = df.idxmax(axis=1).values
        return






