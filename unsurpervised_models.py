# -*- coding: utf-8 -*-
# @Time: 2020/9/18 11:48
# @Author: sancica
# @File: unsurpervised_models.py
# functional description: unserpervised models!!!


def kmeans(X, args={}):
    from sklearn.cluster import KMeans
    model = KMeans(**args)
    model.fit(X)
    return model


def affinity_propagation(X, args={}):
    """
    AffinityPropagation聚类：图聚类的一种
    """
    from sklearn.cluster import AffinityPropagation
    model = AffinityPropagation(**args)
    model.fit(X)
    return model


def spectral(X, args={}):
    """
    Spectral聚类：图聚类的一种
    """
    from sklearn.cluster import SpectralClustering
    model = SpectralClustering(**args)
    model.fit(X)
    return model


def agglomerative(X, args={}):
    """
    层次聚类：Agglomerative 聚类
    """
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(**args)
    model.fit(X)
    return model


def dbscan(X, args={}):
    """
    dbscan聚类算法
    """
    from sklearn.cluster import DBSCAN
    model = DBSCAN(**args)
    model.fit(X)
    return model

