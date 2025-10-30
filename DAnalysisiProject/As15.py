import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取处理后的数据
df = pd.read_csv('car_sales_data_processed_all.csv')

# 多变量分析，探索影响汽车价格的关键因素
print("多变量分析，探索影响汽车价格的关键因素")

# 1. 相关性分析
# 选择数值型特征
numeric_features = ['Engine size', 'Year of manufacture', 'Mileage', 'Price']
correlation = df[numeric_features].corr()

# 打印相关性矩阵
print("\n特征相关性矩阵:")
print(correlation)

# 2. 特征工程 - 为分类变量创建编码
# 对分类变量进行编码
label_encoders = {}
categorical_features = ['Manufacturer', 'Model', 'Fuel type']

for feature in categorical_features:
    le = LabelEncoder()
    df[feature + '_encoded'] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# 3. 构建预测模型
# 准备特征和目标变量
X_features = ['Engine size', 'Year of manufacture', 'Mileage', 
              'Manufacturer_encoded', 'Model_encoded', 'Fuel type_encoded']
X = df[X_features]
y = df['Price']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

# 随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# 打印模型评估结果
print("\n线性回归模型评估:")
print(f"均方误差 (MSE): {lr_mse:.2f}")
print(f"决定系数 (R²): {lr_r2:.2f}")
print("\n随机森林模型评估:")
print(f"均方误差 (MSE): {rf_mse:.2f}")
print(f"决定系数 (R²): {rf_r2:.2f}")

# 4. 特征重要性分析
# 线性回归系数
lr_coef = pd.DataFrame({
    'Feature': X_features,
    'Coefficient': lr_model.coef_
})
lr_coef = lr_coef.reindex(lr_coef['Coefficient'].abs().sort_values(ascending=False).index)

# 随机森林特征重要性
rf_importance = pd.DataFrame({
    'Feature': X_features,
    'Importance': rf_model.feature_importances_
})
rf_importance = rf_importance.sort_values('Importance', ascending=False)

print("\n线性回归特征系数:")
print(lr_coef)
print("\n随机森林特征重要性:")
print(rf_importance)

# 5. 可视化分析
plt.figure(figsize=(15, 10))

# 相关性热图
plt.subplot(2, 2, 1)
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征相关性热图')

# 线性回归系数可视化
plt.subplot(2, 2, 2)
sns.barplot(x='Coefficient', y='Feature', data=lr_coef)
plt.title('线性回归模型特征系数')
plt.xlabel('系数值')
plt.ylabel('特征')

# 随机森林特征重要性可视化
plt.subplot(2, 2, 3)
sns.barplot(x='Importance', y='Feature', data=rf_importance)
plt.title('随机森林模型特征重要性')
plt.xlabel('重要性')
plt.ylabel('特征')

# 预测值与实际值对比
plt.subplot(2, 2, 4)
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('随机森林模型：预测值 vs 实际值')
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.tight_layout()

# 6. 额外分析：价格分布与各特征的关系
plt.figure(figsize=(15, 10))

# 价格与发动机大小的关系，按制造商分组
plt.subplot(2, 2, 1)
sns.scatterplot(x='Engine size', y='Price', hue='Manufacturer', data=df.sample(1000), alpha=0.7)
plt.title('价格与发动机大小的关系（按制造商分组）')
plt.xlabel('发动机大小')
plt.ylabel('价格')
plt.legend(title='制造商', loc='upper right')

# 价格与生产年份的关系，按燃料类型分组
plt.subplot(2, 2, 2)
sns.scatterplot(x='Year of manufacture', y='Price', hue='Fuel type', data=df.sample(1000), alpha=0.7)
plt.title('价格与生产年份的关系（按燃料类型分组）')
plt.xlabel('生产年份')
plt.ylabel('价格')
plt.legend(title='燃料类型', loc='upper right')

# 价格与里程的关系，按发动机大小分组
plt.subplot(2, 2, 3)
df['Engine size group'] = pd.cut(df['Engine size'], bins=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
sns.scatterplot(x='Mileage', y='Price', hue='Engine size group', data=df.sample(1000), alpha=0.7)
plt.title('价格与里程的关系（按发动机大小分组）')
plt.xlabel('里程')
plt.ylabel('价格')
plt.legend(title='发动机大小组', loc='upper right')

# 价格分布
plt.subplot(2, 2, 4)
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('汽车价格分布')
plt.xlabel('价格')
plt.ylabel('频数')
plt.tight_layout()

# 保存结果
plt.savefig('multivariate_analysis.png')
plt.close('all')

# 保存模型评估结果和特征重要性
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MSE': [lr_mse, rf_mse],
    'R2': [lr_r2, rf_r2]
})
results.to_csv('model_evaluation.csv', index=False)
rf_importance.to_csv('feature_importance.csv', index=False)

print("\n分析完成，结果已保存到 multivariate_analysis.png, model_evaluation.csv 和 feature_importance.csv")