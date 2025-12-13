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
try:
    from regression_xgb import run_regression
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("提示: 未安装 xgboost，已跳过回归分析")
from classification_module import run_classification
from clustering_pca_module import run_clustering
from association_rules_module import run_association
from recommender_module import run_recommender

# 关联规则依赖在子模块内自检，主流程不强制退出

# --- 1. 全局设置 ---
sns.set(style="whitegrid")
# 字体容错处理：解决中文方框问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

base_dir = Path(__file__).resolve().parent
file_path = base_dir / 'car_sales_preprocessed.csv'

if not file_path.exists():
    print(f"错误：找不到文件 {file_path}")
    exit()

df = pd.read_csv(file_path)
X = df.drop('Price', axis=1)
y = df['Price']

def _show_image(path, title=None):
    try:
        img = plt.imread(path)
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()
    except Exception as e:
        print(f"显示图像失败: {path} - {e}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. 回归分析
if HAS_XGB:
    print("--- 1. 回归分析: 价格预测  ---")
    run_regression(X_train, y_train, X_test, y_test, base_dir)
    _show_image(base_dir / 'analysis_regression.png', '回归分析：真实值 vs 预测值')
else:
    print("--- 1. 回归分析: 已跳过 (缺少 xgboost) ---")

# 2. 分类分析 (Classification)
run_classification(df, base_dir)
_show_image(base_dir / 'analysis_classification.png', '分类分析：混淆矩阵')

# 3. 聚类与降维 (Clustering & PCA)
run_clustering(X, df, base_dir)
_show_image(base_dir / 'analysis_clustering_pca.png', 'KMeans 聚类 (PCA)')

# 4. 关联规则分析 (Association Rules)
run_association(X, y, base_dir)
_show_image(base_dir / 'analysis_association_rules.png', '关联规则分布')

# 5. 推荐分析 (Recommender)
run_recommender(X, y)
