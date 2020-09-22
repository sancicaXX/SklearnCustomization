# -*- coding: utf-8 -*-
# @Time: 2020/9/18 9:50
# @Author: sancica
# @File: feature.py
# functional description: feature selection method

# -------------------------------------------特征选择-------------------------------
def variance_threshold(data, threshold):
    """
    基于方差阈值进行特征选择
    """
    from sklearn.feature_selection import VarianceThreshold
    return VarianceThreshold(threshold=threshold).fit_transform(data)


def select_K_best(data, labels, selector, k):
    """
    通过单变量评分进行特征选择
    selector：单变量评分方法
    """
    from sklearn.feature_selection import SelectKBest
    return SelectKBest(selector, k=k).fit_transform(data, labels)


def L1(data, labels, args={}):
    """
    通过模型进行特征选择，这里选取L1权重为非0的特征
    """
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel
    lsvc = LinearSVC(**args).fit(data, labels)
    model = SelectFromModel(lsvc, prefit=True)
    return model.transform(data)


def extra_tree(data, labels, args={}):
    """
    通过模型进行特征选择，这里选择extra_tree选择特征
    """
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier(**args).fit(data, labels)
    model = SelectFromModel(clf, prefit=True)
    return model.transform(data)


def polynomial_feature(X, n, args={}):
    """
    生成多项式特征
    """
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(n, **args)
    poly.fit(X)
    return poly


# --------------------------------------特征提取----------------------------------------
def dict_vectorizer(X, args={}):
    """
    将dict数据转为numpy数据的格式，one-hot表示
    """
    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer(**args)
    return vec.fit_transform(X).toarray()


def count_vectorizer(X, args={}):
    """
    统计文本语句单词次数，并形成字典，并将原始文本数据转为向量频次特征
    这里也可以和n-gram进行结合，只要在args中提供 ngram_range属性即可
    """
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(**args)
    return vectorizer.fit_transform(X).toarray()


def tfidf_vectorizer(X, args={}):
    """
    TFIDF统计特征
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(**args)
    return vectorizer.fit_transform(X)


# -----------------------------------------特征预处理-------------------------
def scale(X, args={}):
    """
    标准化数据,直接返回标准化之后的数据
    """
    from sklearn.preprocessing import scale
    return scale(X)


def standard_scaler(X, args={}):
    """
    和scale一样，但是使用fit，之后可以用于测试集数据中
    """
    from sklearn.preprocessing import StandardScaler
    sds = StandardScaler(**args)
    sds.fit(X)
    return sds


def robust_scaler(X, args={}):
    """
    同样的标准化，但是能够处理outliers
    """
    from sklearn.preprocessing import RobustScaler
    rbs = RobustScaler(**args)
    rbs.fit(X)
    return rbs


def min_max_scaler(X, args={}):
    """
    最大最小归一化
    """
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler(**args)
    mms.fit(X)
    return mms


def normalize(X, norm='l2', args={}):
    """
    使用正则化转换数据，将每一个样本的范数转为单位范数
    使用场景：比如tfidf的两个样本normaliza之后进行点积，得到两个变量的余弦相似度
    """
    from sklearn.preprocessing import Normalizer
    nml = Normalizer(norm=norm, **args)
    return nml.transform(X)


# -------------------------------------------特征离散化-----------------------------------------

def KBins_discretizer(X, n_bins, encode, args={}):
    """
    分桶离散化
    """
    from sklearn.preprocessing import KBinsDiscretizer
    est = KBinsDiscretizer(n_bins, encode, **args)
    est.fit(X)
    return est


def power_transformer(X, method, args={}):
    """
    特征非线性变换，这里采用高斯分布
    """
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method, **args)
    pt.fit(X)
    return pt


# ----------------------------- 特征降维，非特征选择--------------------------
def pca(X, args={}):
    """
    PCA主成分分析，得到的降维数据是映射到主成分之后的数据
    """
    from sklearn.decomposition import PCA
    pca_m = PCA(**args)
    pca_m.fit(X)
    return pca


def lda(X, args={}):
    """
    潜在狄利克雷分布
    """
    from sklearn.decomposition import LatentDirichletAllocation
    lda_m = LatentDirichletAllocation(**args)
    lda_m.fit(X)
    return lda_m


def feature_agglomeration(X, args={}):
    """
    使用层次聚类对特征进行聚类，然后进行特征降维
    """
    from sklearn.cluster import FeatureAgglomeration
    fam = FeatureAgglomeration(**args)
    fam.fit(X)
    return fam


# ------------------------------------------特征重要性------------------------------------------
def permutation_performance(clf, X, y, args={}):
    """
    通过随机变换特征对目标的影响用来判断特征重要性
    """
    from sklearn.inspection import permutation_importance
    result = permutation_importance(clf, X, y, **args)
    return result


# ---------------------------------------- 标签变换------------------------------------------
def label_binarizer(y, args={}):
    """
    将标签转换为one-hot
    """
    from sklearn.preprocessing import LabelBinarizer
    lbr = LabelBinarizer(**args)
    lbr.fit(y)
    return lbr



