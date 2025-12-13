import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

def run_recommender(X, y):
    """推荐模块：对度量/算法进行5折CV，以“均值价格差”最小为准择优，
    在全量数据上检索示例并输出距离与价格差。"""
    print("\n--- 5. 相似车型推荐 ---")
    X_clean = X.drop(columns=[c for c in ['Cluster'] if c in X.columns])

    def build(metric, algo, metric_params=None, k=5):
        return NearestNeighbors(n_neighbors=k, metric=metric, algorithm=algo, metric_params=metric_params)
    #候选组合
    candidates = [  # 常用度量与算法组合
        ("euclidean", "auto", None),
        ("cosine", "brute", None),
        ("minkowski", "auto", {"p": 1}),
    ]
    #对候选组合进行 5 折交叉验证
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    k = 5
    stats = {}
    for metric, algo, mparams in candidates:
        fold_scores = []
        for tr_idx, te_idx in cv.split(X_clean):  # 以测试折的均值价格差作为评估指标
            X_tr, y_tr = X_clean.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X_clean.iloc[te_idx], y.iloc[te_idx]
            nn = build(metric, algo, mparams, k)
            nn.fit(X_tr)
            dists, inds = nn.kneighbors(X_te)
            price_diff = []
            for j in range(len(X_te)):
                nbr_prices = y_tr.iloc[inds[j]]
                price_diff.append(float(np.mean(np.abs(nbr_prices.values - y_te.iloc[j]))))
            fold_scores.append(float(np.mean(price_diff)))
        stats[(metric, algo)] = (float(np.mean(fold_scores)), float(np.std(fold_scores)))
    #选择均值价格差最小的组合
    best_cfg = min(stats.keys(), key=lambda kx: stats[kx][0])
    mean_s, std_s = stats[best_cfg]
    print("推荐CV(均值价格差) mean±std:")
    for (metric, algo), (m, s) in sorted(stats.items(), key=lambda x: x[1][0]):
        print(f"  {metric}/{algo}: {m:.0f}±{s:.0f}")
    print(f"最终采用: {best_cfg[0]}/{best_cfg[1]}")
    #全量数据上检索输出相似车辆
    nn = build(best_cfg[0], best_cfg[1], None, k)
    nn.fit(X_clean)
    q = 0
    dists, inds = nn.kneighbors(X_clean.iloc[[q]])
    print(f"查询车辆价格: {y.iloc[q]}")
    print("推荐相似车辆:")
    for d, i in zip(dists[0], inds[0]):
        if i == q:
            continue
        diff = abs(float(y.iloc[i] - y.iloc[q]))
        print(f"ID: {i}, 价格: {y.iloc[i]}, 距离: {d:.4f}, 价格差: {diff:.0f}")
