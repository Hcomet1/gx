import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. 读取与清洗
df = pd.read_csv('car_sales_data.csv')
df.drop_duplicates(inplace=True)

# 2. 特征工程：构造"车龄"
current_year = 2025
df['Car_Age'] = current_year - df['Year of manufacture']
df.drop('Year of manufacture', axis=1, inplace=True)

# 3. 分离特征 (X) 和目标 (y)
X = df.drop('Price', axis=1)
y = df['Price']

# 4. 构建预处理管道 (Pipeline)
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# 数值处理：补均值 -> 标准化
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 类别处理：补众数 -> 独热编码
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 组合转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# 相关性热力图，目的：直观呈现车龄、里程、排量与价格之间的相关性强度
plt.figure(figsize=(8, 6))
# 计算数值列的相关系数
corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()

sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
plt.title('Feature Correlation Analysis')
plt.show()

# 执行预处理
X_processed = preprocessor.fit_transform(X)
print(f"数据预处理完成。特征矩阵形状: {X_processed.shape}")
print("包含处理：去重 -> 特征构造(车龄) -> 缺失值填充 -> 数值标准化 -> 类别独热编码")

#保存
feature_names = preprocessor.get_feature_names_out()

df_export = pd.DataFrame(X_processed, columns=feature_names)
df_export['Price'] = y.reset_index(drop=True)

output_filename = 'car_sales_preprocessed.csv'
df_export.to_csv(output_filename, index=False)

print(f"预处理数据已成功保存至: {output_filename}")
print(f"保存文件维度: {df_export.shape}")