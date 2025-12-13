import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold

def run_clustering(X, df, base_dir):
    """聚类模块：在数值特征子空间上对多算法(kmeans/minibatch/gmm)与k进行5折CV，
    以轮廓系数均值选择最优模型，报告全量得分并保存PCA可视化。"""
    print("\n--- 3. 聚类与降维可视化 ---")
    num_cols = [c for c in X.columns if c.startswith('num__')]
    X_cluster = X[num_cols] if len(num_cols) > 0 else X  # 优先数值特征避免稀疏高维影响
    #封装构建算法函数
    def build(algo, k):
        if algo == 'kmeans':
            return KMeans(n_clusters=k, random_state=42, n_init=20)
        if algo == 'minibatch':
            return MiniBatchKMeans(n_clusters=k, random_state=42, n_init=20, batch_size=1024)
        if algo == 'gmm':
            return GaussianMixture(n_components=k, random_state=42, covariance_type='full', n_init=5)
        raise ValueError('unknown algo')
    #对算法与簇数 k∈[2,10] 进行 5 折交叉验证
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    sample_sz = min(5000, len(X_cluster))  # 限制评估采样规模提高速度
    cv_stats = {}
    for algo in ['kmeans', 'minibatch', 'gmm']:
        for k in range(2, 11):
            scores = []
            for train_idx, _ in cv.split(X_cluster):
                X_sub = X_cluster.iloc[train_idx]
                model = build(algo, k)
                labels = model.fit_predict(X_sub) if hasattr(model, 'fit_predict') else model.fit(X_sub).predict(X_sub)
                s = silhouette_score(X_sub, labels, sample_size=sample_sz, random_state=42)
                scores.append(s)
            cv_stats[(algo, k)] = (float(np.mean(scores)), float(np.std(scores)))
    #选择最优 (算法,k) 组合并输出 Top3
    best_algo, best_k = max(cv_stats.keys(), key=lambda t: cv_stats[t][0])
    mean_s, std_s = cv_stats[(best_algo, best_k)]
    print("CV轮廓系数 (mean±std) Top:")
    top = sorted(cv_stats.items(), key=lambda x: x[1][0], reverse=True)[:3]
    for (algo, k), (m, s) in top:
        print(f"  {algo}-k={k}: {m:.4f}±{s:.4f}")
    print(f"最终采用: {best_algo}-k={best_k}")

    model = build(best_algo, best_k)  # 在全量数据上拟合最佳模型
    clusters = model.fit_predict(X_cluster) if hasattr(model, 'fit_predict') else model.fit(X_cluster).predict(X_cluster)
    df['Cluster'] = clusters
    full_s = silhouette_score(X_cluster, clusters, sample_size=sample_sz, random_state=42)
    print(f"全量轮廓系数: {full_s:.4f}")
    #可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title('PCA 2D 聚类可视化')
    plt.tight_layout()
    path = base_dir / 'analysis_clustering_pca.png'
    plt.savefig(path)
    plt.close()
    level = '良好' if full_s >= 0.50 else ('一般' if full_s >= 0.30 else '较弱')
    print(f"图像: {path} - 聚类PCA散点图 | 评估: 轮廓系数={full_s:.3f}（{level}）")
