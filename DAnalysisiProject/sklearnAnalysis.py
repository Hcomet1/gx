import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Sklearn 模块
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score, classification_report, silhouette_score, confusion_matrix

# 关联规则模块
try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules
except ImportError:
    print("错误: 请先安装 mlxtend 库 (pip install mlxtend) 以运行关联规则分析")
    exit()

# --- 1. 全局设置 ---
sns.set(style="whitegrid")
# 字体容错处理：解决中文方框问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 路径处理：基于当前脚本位置动态定位，解决路径报错
base_dir = Path(__file__).resolve().parent
file_path = base_dir / 'car_sales_preprocessed.csv'

if not file_path.exists():
    print(f"错误：找不到文件 {file_path}")
    exit()

df = pd.read_csv(file_path)
X = df.drop('Price', axis=1)
y = df['Price']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. 回归分析 (Regression)
print("--- 1. 回归分析: 价格预测 ---")
reg = LinearRegression().fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred_reg):.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_reg, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('回归分析：真实值 vs 预测值')
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.tight_layout()
plt.savefig(base_dir / 'analysis_regression.png')
plt.show()

# 2. 分类分析 (Classification)
print("\n--- 2. 分类分析: 价格档次预测 ---")
y_class = pd.qcut(df['Price'], q=3, labels=['Low', 'Medium', 'High'])
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000).fit(X_train_c, y_train_c)
y_pred_clf = clf.predict(X_test_c)
print(classification_report(y_test_c, y_pred_clf))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_c, y_pred_clf), annot=True, fmt='d', cmap='Blues',
            xticklabels=['High', 'Low', 'Medium'], yticklabels=['High', 'Low', 'Medium'])
plt.title('分类分析：混淆矩阵')
plt.tight_layout()
plt.savefig(base_dir / 'analysis_classification.png')
plt.show()

# 3. 聚类与降维 (Clustering & PCA)
print("\n--- 3. 聚类与降维可视化 ---")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters
print(f"轮廓系数: {silhouette_score(X, clusters, sample_size=5000):.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='聚类 ID')
plt.title('PCA 降维可视化聚类结果')
plt.tight_layout()
plt.savefig(base_dir / 'analysis_clustering_pca.png')
plt.show()

# 4. 关联规则分析 (Association Rules)
print("\n--- 4. 关联规则分析: 车型特征关联 (FPGrowth) ---")

cat_cols = [c for c in X.columns if c.startswith('cat__')]
basket_data = X[cat_cols].astype(bool)

print(f"参与关联分析的特征数: {basket_data.shape[1]}")

# 挖掘频繁项集 (支持度 > 0.05，即出现在5%以上的车中)
frequent_itemsets = fpgrowth(basket_data, min_support=0.05, use_colnames=True)

if not frequent_itemsets.empty:
    # 生成规则 (置信度 > 0.5)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    # 排序并展示强规则
    top_rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False]).head(5)
    print("\nTop 5 强关联规则 (按提升度 Lift 排序):")
    for idx, row in top_rules.iterrows():
        print(f"规则: {set(row['antecedents'])} -> {set(row['consequents'])} | Lift: {row['lift']:.2f}")

    # 可视化：规则散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="support", y="confidence", size="lift", hue="lift",
                    data=rules, palette="viridis", sizes=(20, 200))
    plt.title('关联规则分布 (Support vs Confidence)')
    plt.tight_layout()
    plt.savefig(base_dir / 'analysis_association_rules.png')
    plt.show()
else:
    print("未找到满足条件的频繁项集，请尝试降低 min_support")

# 5. 推荐分析 (Recommender)
print("\n--- 5. 相似车型推荐 ---")
X_clean = X.drop(columns=[c for c in ['Cluster'] if c in X.columns])
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X_clean)
distances, indices = nbrs.kneighbors(X_clean.iloc[0].values.reshape(1, -1))

print(f"查询车辆价格: {y.iloc[0]}")
print("推荐相似车辆ID:", indices[0][1:])