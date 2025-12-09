import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score, classification_report, silhouette_score, confusion_matrix

# --- 1. 全局设置
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方框的问题

# 读取数据
df = pd.read_csv('car_sales_preprocessed.csv')
X = df.drop('Price', axis=1)
y = df['Price']

# 划分数据集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. 回归分析：预测具体价格 ---
print("--- 1. 回归分析 (Linear Regression) ---")
reg = LinearRegression().fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred_reg):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_reg):.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_reg, alpha=0.3, label='预测点')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='完美预测线')
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.title('回归分析：真实值 vs 预测值')
plt.legend()
plt.tight_layout()
plt.savefig('analysis_regression.png')
plt.show() # 显式展示

# --- 3. 分类分析：预测价格档次 ---
print("\n--- 2. 分类分析 (Logistic Regression) ---")
# 将价格离散化为三档：Low, Medium, High
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
plt.savefig('analysis_classification.png')
plt.show()

# --- 4. 聚类与降维可视化 ---
print("\n--- 3. 聚类 (KMeans) & 4. 降维 (PCA) ---")
# 聚类
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters
print(f"轮廓系数 (Silhouette Score): {silhouette_score(X, clusters, sample_size=5000):.4f}")

# 降维 (用于2D可视化)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
# 优化：使用 scatter 并在 c 参数传入聚类标签，实现自动着色
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='聚类 ID')
plt.xlabel('PCA 主成分 1')
plt.ylabel('PCA 主成分 2')
plt.title('KMeans 聚类结果可视化 (PCA降维)')
plt.tight_layout()
plt.savefig('analysis_clustering_pca.png')
plt.show()

# --- 5. 关联/推荐：寻找相似车型 ---
print("\n--- 5. 推荐分析 (Nearest Neighbors) ---")
# 使用 Ball Tree 算法加速搜索
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)

# 示例：查询第0辆车的相似车
query_idx = 0
distances, indices = nbrs.kneighbors(X.iloc[query_idx].values.reshape(1, -1))

print(f"查询目标 (Index {query_idx}) 价格: {y.iloc[query_idx]}")
print("推荐相似车辆:")
for i in indices[0]:
    if i != query_idx:
        print(f"ID: {i}, 价格: {y.iloc[i]}, 相似度距离: {distances[0][list(indices[0]).index(i)]:.4f}")