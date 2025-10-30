import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_csv('car_sales_data.csv')

# 1. 数据基本信息
print("数据基本信息：")
print(f"数据形状: {df.shape}")
print("\n数据类型：")
print(df.dtypes)
print("\n数据前5行：")
print(df.head())

# 2. 缺失值处理
print("\n缺失值统计：")
print(df.isnull().sum())

# 检查是否存在缺失值
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

if len(missing_values) > 0:
    print("\n存在缺失值，进行处理：")
    print(missing_values)
    
    # 对数值型特征使用KNN填充
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if any(df[numeric_cols].isnull().sum() > 0):
        print("\n对数值型特征使用KNN填充：")
        # 选择数值型特征进行KNN填充
        numeric_df = df[numeric_cols]
        
        # 使用KNN填充
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
        
        # 将填充后的数值列更新到原始数据框
        for col in numeric_cols:
            df[col] = df_imputed[col]
        
        print("KNN填充后的缺失值统计：")
        print(df[numeric_cols].isnull().sum())
    
    # 对分类特征使用众数填充
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if any(df[categorical_cols].isnull().sum() > 0):
        print("\n对分类特征使用众数填充：")
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                print(f"{col} 列使用众数 '{mode_value}' 填充")
    
    print("\n填充后的缺失值统计：")
    print(df.isnull().sum())
else:
    print("\n数据中不存在缺失值，无需处理。")

# 3. 异常值检测与处理
print("\n异常值检测与处理：")

# 选择数值型列进行异常值检测
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"数值型列: {numeric_cols}")

# 创建一个图形，用于显示异常值检测结果
plt.figure(figsize=(15, 10))

# 使用Z-score方法检测异常值
z_score_outliers = {}
for i, col in enumerate(numeric_cols):
    z_scores = np.abs(stats.zscore(df[col]))
    outliers_z = np.where(z_scores > 3)[0]
    z_score_outliers[col] = outliers_z
    
    plt.subplot(len(numeric_cols), 2, 2*i+1)
    plt.boxplot(df[col])
    plt.title(f'{col} - 箱线图')
    
    plt.subplot(len(numeric_cols), 2, 2*i+2)
    plt.scatter(range(len(df)), df[col], alpha=0.5)
    plt.scatter(outliers_z, df.iloc[outliers_z][col], color='red')
    plt.title(f'{col} - Z-score异常值 (红色)')

plt.tight_layout()
plt.savefig('outliers_z_score.png')
plt.close()

# 使用IQR方法检测异常值
iqr_outliers = {}
plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_cols):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_iqr = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
    iqr_outliers[col] = outliers_iqr
    
    plt.subplot(len(numeric_cols), 2, 2*i+1)
    sns.boxplot(x=df[col])
    plt.title(f'{col} - 箱线图')
    
    plt.subplot(len(numeric_cols), 2, 2*i+2)
    plt.scatter(range(len(df)), df[col], alpha=0.5)
    plt.scatter(outliers_iqr, df.iloc[outliers_iqr][col], color='red')
    plt.title(f'{col} - IQR异常值 (红色)')

plt.tight_layout()
plt.savefig('outliers_iqr.png')
plt.close()

# 打印异常值统计
print("\nZ-score方法检测到的异常值数量：")
for col in numeric_cols:
    print(f"{col}: {len(z_score_outliers[col])}")

print("\nIQR方法检测到的异常值数量：")
for col in numeric_cols:
    print(f"{col}: {len(iqr_outliers[col])}")

# 创建一个综合处理的数据框
df_processed = df.copy()

# 处理异常值 - 使用截断法（将异常值替换为上下限值）
for col in numeric_cols:
    # 使用IQR方法确定上下限
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 将小于下限的值设为下限
    df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
    # 将大于上限的值设为上限
    df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound

print("\n异常值处理后的统计描述：")
print(df_processed[numeric_cols].describe())

# 4. 数据转换（标准化/归一化）
print("\n数据转换：")

# 标准化处理 - 将结果添加为新列
scaler = StandardScaler()
standardized_values = scaler.fit_transform(df_processed[numeric_cols])

# 将标准化结果添加为新列
for i, col in enumerate(numeric_cols):
    df_processed[f"{col}_standardized"] = standardized_values[:, i]

# 归一化处理 - 将结果添加为新列
min_max_scaler = MinMaxScaler()
normalized_values = min_max_scaler.fit_transform(df_processed[numeric_cols])

# 将归一化结果添加为新列
for i, col in enumerate(numeric_cols):
    df_processed[f"{col}_normalized"] = normalized_values[:, i]

print("\n处理后的数据统计（包含原始数据、标准化和归一化结果）：")
print(df_processed.describe())

# 可视化标准化和归一化的效果
plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_cols):
    plt.subplot(len(numeric_cols), 3, 3*i+1)
    plt.hist(df_processed[col], bins=30, alpha=0.5)
    plt.title(f'原始数据 - {col}')
    
    plt.subplot(len(numeric_cols), 3, 3*i+2)
    plt.hist(df_processed[f"{col}_standardized"], bins=30, alpha=0.5)
    plt.title(f'标准化后 - {col}')
    
    plt.subplot(len(numeric_cols), 3, 3*i+3)
    plt.hist(df_processed[f"{col}_normalized"], bins=30, alpha=0.5)
    plt.title(f'归一化后 - {col}')

plt.tight_layout()
plt.savefig('data_transformation.png')
plt.close()

# 5. 保存处理后的数据（包含所有处理结果）
df_processed.to_csv('car_sales_data_processed_all.csv', index=False)

print("\n数据预处理完成，处理后的数据已保存。")
print("- 综合处理后数据（包含原始数据、标准化和归一化结果）：car_sales_data_processed_all.csv")

# 6. 数据预处理总结
print("\n数据预处理总结：")
print("1. 缺失值处理：")
if len(missing_values) > 0:
    print("   - 数值型特征：使用KNN填充")
    print("   - 分类特征：使用众数填充")
else:
    print("   - 数据中不存在缺失值，无需处理")

print("\n2. 异常值处理：")
print("   - 使用Z-score和IQR方法检测异常值")
print("   - 采用截断法处理异常值，将异常值替换为上下限值")
print("   - 异常值检测结果已保存为图片：outliers_z_score.png 和 outliers_iqr.png")

print("\n3. 数据转换：")
print("   - 对数值型特征进行了标准化和归一化处理，并将结果添加为新列")
print("   - 转换效果已保存为图片：data_transformation.png")
print("   - 标准化公式：z = (x - μ) / σ，其中μ是均值，σ是标准差")
print("   - 归一化公式：x' = (x - min) / (max - min)，将数据缩放到[0, 1]区间")
print("   - 所有处理结果已整合到一个CSV文件中：car_sales_data_processed_all.csv")