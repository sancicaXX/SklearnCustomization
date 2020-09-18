# -*- coding: utf-8 -*-
# @Time: 2020/9/18 15:08
# @Author: sancica
# @File: evaluation.py.py
# functional description: evaluation for models including supervised and unsurpervised

# ---------------------------------聚类外在评估方法--------------------------------------
def rand_index(labels_true, labels_pred):
    """
    聚类评估：兰德系数，外在方法（指用于原始数据的真实标签）
    return: score,float, 最大值为1
    """
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels_true, labels_pred)


def mutual_info(labels_true, labels_pred):
    """
    聚类评估：互信息，外在方法
    return：score, float, 最大值为1
    """
    from sklearn.metrics import adjusted_mutual_info_score
    return adjusted_mutual_info_score(labels_true, labels_pred)


def fowlkes_mallows(labels_true, labels_pred):
    """
    聚类评估：TODO:
    """
    from sklearn.metrics import fowlkes_mallows_score
    return fowlkes_mallows_score(labels_true, labels_pred)


# ---------------------------- 聚类内在评估方法----------------------------------
def silhouette(X, labels, args={}):
    """
    聚类评估：轮廓系数，内在方法，没有真实标签
    return:float, -1--1
    """
    from sklearn.metrics import silhouette_score
    return silhouette_score(X, labels, **args)


def calinski_harabasz(X, labels, args={}):
    """
    聚类评估：CHI， 内在方法
    return: score, 分数越大越好
    """
    from sklearn.metrics import calinski_harabasz_score
    return calinski_harabasz_score(X, labels, **args)


def davies_bouldin(X, labels, args={}):
    """
    聚类评估：DBI， 内在方法
    return:score,越接近0越好
    """
    from sklearn.metrics import davies_bouldin_score
    return davies_bouldin_score(X, labels, **args)
