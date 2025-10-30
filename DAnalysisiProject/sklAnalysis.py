import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from mlxtend.frequent_patterns import apriori, association_rules

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取处理后的数据
print("读取数据...")
df = pd.read_csv('car_sales_data_processed_all.csv')
print(f"数据形状: {df.shape}")
print(df.columns)

# 数据预处理
print("\n数据预处理...")
# 对分类变量进行编码
categorical_features = ['Manufacturer', 'Model', 'Fuel type']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    df[feature + '_encoded'] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# 选择数值特征用于分析
numeric_features = ['Engine size', 'Year of manufacture', 'Mileage', 'Price', 
                    'Manufacturer_encoded', 'Model_encoded', 'Fuel type_encoded']
X = df[numeric_features].copy()

# 检查缺失值
print(f"缺失值数量: {X.isnull().sum().sum()}")

# 1. K-means聚类分析
print("\n1. 执行K-means聚类分析...")
# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 确定最佳聚类数 (使用肘部法则)
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('聚类数量')
plt.ylabel('惯性值 (Inertia)')
plt.title('K-means聚类肘部法则图')
plt.savefig('kmeans_elbow.png')

# 选择合适的聚类数 (这里选择4个聚类)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 分析聚类结果
cluster_stats = df.groupby('cluster').agg({
    'Price': ['mean', 'min', 'max', 'count'],
    'Engine size': 'mean',
    'Year of manufacture': 'mean',
    'Mileage': 'mean',
    'Fuel type': lambda x: x.value_counts().index[0]  # 最常见的燃料类型
})
print("\n聚类统计结果:")
print(cluster_stats)

# 可视化聚类结果 (使用PCA降维到2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
for i in range(optimal_k):
    plt.scatter(X_pca[df['cluster'] == i, 0], X_pca[df['cluster'] == i, 1], 
                label=f'聚类 {i}', alpha=0.7)
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('K-means聚类结果 (PCA降维)')
plt.legend()
plt.savefig('kmeans_clusters.png')

# 2. 分类分析 - 预测燃料类型
print("\n2. 执行分类分析 - 预测燃料类型...")
# 准备数据
X_class = df[['Engine size', 'Year of manufacture', 'Mileage', 'Price', 'Manufacturer_encoded', 'Model_encoded']]
y_class = df['Fuel type_encoded']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

# 训练多个分类模型并比较
classifiers = {
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "决策树": DecisionTreeClassifier(random_state=42),
    "支持向量机": SVC(random_state=42),
    "K近邻": KNeighborsClassifier(n_neighbors=5)
}

# 评估分类模型
classification_results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_results[name] = accuracy
    print(f"{name} 准确率: {accuracy:.4f}")

# 使用最佳模型进行详细分析
best_classifier = max(classification_results, key=classification_results.get)
print(f"\n最佳分类模型: {best_classifier}")

best_clf = classifiers[best_classifier]
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 如果是随机森林，分析特征重要性
if best_classifier == "随机森林":
    feature_importance = pd.DataFrame({
        'Feature': X_class.columns,
        'Importance': best_clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('特征重要性 - 燃料类型预测')
    plt.tight_layout()
    plt.savefig('feature_importance_classification.png')

# 3. 回归分析 - 预测汽车价格
print("\n3. 执行回归分析 - 预测汽车价格...")
# 准备数据
X_reg = df[['Engine size', 'Year of manufacture', 'Mileage', 'Manufacturer_encoded', 'Model_encoded', 'Fuel type_encoded']]
y_reg = df['Price']

# 划分训练集和测试集
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# 训练多个回归模型并比较
regressors = {
    "线性回归": LinearRegression(),
    "随机森林回归": RandomForestRegressor(n_estimators=100, random_state=42),
    "神经网络回归": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# 评估回归模型
regression_results = {}
for name, reg in regressors.items():
    reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    regression_results[name] = r2
    print(f"{name} - MSE: {mse:.2f}, R²: {r2:.4f}")

# 使用最佳模型进行详细分析
best_regressor = max(regression_results, key=regression_results.get)
print(f"\n最佳回归模型: {best_regressor}")

best_reg = regressors[best_regressor]
best_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = best_reg.predict(X_test_reg)

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.title(f'{best_regressor} - 价格预测结果')
plt.savefig('regression_prediction.png')

# 如果是随机森林，分析特征重要性
if best_regressor == "随机森林回归":
    feature_importance_reg = pd.DataFrame({
        'Feature': X_reg.columns,
        'Importance': best_reg.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_reg)
    plt.title('特征重要性 - 价格预测')
    plt.tight_layout()
    plt.savefig('feature_importance_regression.png')

# 4. 主成分分析 (PCA)
print("\n4. 执行主成分分析 (PCA)...")
# 准备数据
X_pca_full = df[['Engine size', 'Year of manufacture', 'Mileage', 'Price', 
                'Manufacturer_encoded', 'Model_encoded', 'Fuel type_encoded']]
X_pca_scaled = StandardScaler().fit_transform(X_pca_full)

# 执行PCA
pca = PCA()
X_pca_transformed = pca.fit_transform(X_pca_scaled)

# 分析解释方差比
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# 可视化解释方差
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='单个方差')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='累积方差')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% 方差阈值')
plt.xlabel('主成分数量')
plt.ylabel('解释方差比')
plt.title('PCA解释方差分析')
plt.legend()
plt.savefig('pca_variance.png')

# 确定需要的主成分数量
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"解释95%方差所需的主成分数量: {n_components}")

# 使用前两个主成分可视化数据
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1], 
                     c=df['Fuel type_encoded'], cmap='viridis', alpha=0.7)
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('PCA降维可视化 (按燃料类型着色)')
plt.colorbar(scatter, label='燃料类型')
plt.savefig('pca_visualization.png')

# 分析主成分的组成
components = pd.DataFrame(pca.components_[:2].T, 
                         columns=['PC1', 'PC2'], 
                         index=X_pca_full.columns)
print("\nPCA主成分组成:")
print(components)

# 5. 特征选择和关联规则挖掘
print("\n5. 执行特征选择和关联规则挖掘...")

# 特征选择
selector = SelectKBest(f_classif, k=4)
X_selected = selector.fit_transform(X_class, y_class)
selected_features = X_class.columns[selector.get_support()]
print(f"选择的最重要特征: {selected_features}")

# 关联规则挖掘
# 为了应用关联规则，我们需要将数据离散化
# 创建离散化的数据框
df_discrete = pd.DataFrame()

# 离散化数值特征
df_discrete['Engine_size_cat'] = pd.cut(df['Engine size'], bins=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
df_discrete['Year_cat'] = pd.cut(df['Year of manufacture'], bins=5, labels=['Very Old', 'Old', 'Medium', 'New', 'Very New'])
df_discrete['Mileage_cat'] = pd.cut(df['Mileage'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
df_discrete['Price_cat'] = pd.cut(df['Price'], bins=5, labels=['Very Cheap', 'Cheap', 'Medium', 'Expensive', 'Very Expensive'])
df_discrete['Manufacturer'] = df['Manufacturer']
df_discrete['Fuel_type'] = df['Fuel type']

# 将数据转换为one-hot编码
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    return x

# 创建one-hot编码
df_onehot = pd.get_dummies(df_discrete)

# 应用Apriori算法
frequent_itemsets = apriori(df_onehot, min_support=0.1, use_colnames=True)
print(f"发现的频繁项集数量: {len(frequent_itemsets)}")

# 生成关联规则
if len(frequent_itemsets) > 0:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    if len(rules) > 0:
        print("\n发现的关联规则:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        
        # 可视化关联规则
        plt.figure(figsize=(10, 6))
        plt.scatter(rules['support'], rules['confidence'], alpha=0.5, s=rules['lift']*20)
        plt.xlabel('支持度 (Support)')
        plt.ylabel('置信度 (Confidence)')
        plt.title('关联规则 - 支持度 vs 置信度')
        plt.savefig('association_rules.png')
    else:
        print("未发现满足条件的关联规则")
else:
    print("未发现满足条件的频繁项集")

# 总结分析结果
print("\n分析总结:")
print(f"1. K-means聚类分析: 将汽车数据分为{optimal_k}个聚类")
print(f"2. 分类分析: 最佳模型为{best_classifier}，准确率为{classification_results[best_classifier]:.4f}")
print(f"3. 回归分析: 最佳模型为{best_regressor}，R²为{regression_results[best_regressor]:.4f}")
print(f"4. PCA分析: 解释95%方差需要{n_components}个主成分")
print("5. 关联规则挖掘: 发现了汽车特征之间的关联模式")

print("\n所有分析完成，结果已保存为图表。")