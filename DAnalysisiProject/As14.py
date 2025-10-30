import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取处理后的数据
df = pd.read_csv('car_sales_data_processed_all.csv')

# 按发动机大小分析燃油效率和价格的相关性
print("按发动机大小分析燃油效率和价格的相关性")

# 创建发动机大小的分组
df['engine_size_group'] = pd.cut(df['Engine size'], bins=[0, 1.0, 1.5, 2.0, 2.5, 3.0], 
                                 labels=['0-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0'])

# 计算每个发动机大小组的统计特征
engine_stats = df.groupby('engine_size_group').agg({
    'Price': ['mean', 'median', 'std', 'min', 'max'],
    'Mileage': ['mean', 'median', 'std', 'min', 'max'],
    'Year of manufacture': ['mean', 'count']
}).reset_index()

# 打印统计结果
print("\n发动机大小组统计特征:")
print(engine_stats)

# 保存统计结果到CSV
engine_stats.to_csv('engine_size_stats.csv')

# 计算燃油效率指标（假设里程数与燃油效率成反比，即里程数越高，燃油效率越低）
# 这里我们使用里程/价格作为一个简单的效率指标
df['efficiency_index'] = df['Mileage'] / df['Price']

# 可视化分析
plt.figure(figsize=(15, 10))

# 1. 发动机大小与平均价格的关系
plt.subplot(2, 2, 1)
sns.barplot(x='engine_size_group', y=('Price', 'mean'), data=engine_stats)
plt.title('各发动机大小组的平均价格')
plt.xlabel('发动机大小组')
plt.ylabel('平均价格')

# 2. 发动机大小与平均里程的关系
plt.subplot(2, 2, 2)
sns.barplot(x='engine_size_group', y=('Mileage', 'mean'), data=engine_stats)
plt.title('各发动机大小组的平均里程')
plt.xlabel('发动机大小组')
plt.ylabel('平均里程')

# 3. 散点图：发动机大小与价格的关系
plt.subplot(2, 2, 3)
sns.scatterplot(x='Engine size', y='Price', data=df, alpha=0.5)
plt.title('发动机大小与价格的关系')
plt.xlabel('发动机大小')
plt.ylabel('价格')

# 4. 散点图：发动机大小与效率指标的关系
plt.subplot(2, 2, 4)
sns.scatterplot(x='Engine size', y='efficiency_index', data=df, alpha=0.5)
plt.title('发动机大小与效率指标的关系')
plt.xlabel('发动机大小')
plt.ylabel('效率指标（里程/价格）')
plt.tight_layout()

# 5. 箱线图：发动机大小组与价格分布
plt.figure(figsize=(12, 6))
sns.boxplot(x='engine_size_group', y='Price', data=df)
plt.title('各发动机大小组的价格分布')
plt.xlabel('发动机大小组')
plt.ylabel('价格')
plt.tight_layout()

# 6. 按燃料类型和发动机大小分析价格
plt.figure(figsize=(12, 6))
sns.boxplot(x='engine_size_group', y='Price', hue='Fuel type', data=df)
plt.title('各发动机大小组和燃料类型的价格分布')
plt.xlabel('发动机大小组')
plt.ylabel('价格')
plt.legend(title='燃料类型')
plt.tight_layout()

# 7. 热图：发动机大小、价格、里程、年份的相关性
plt.figure(figsize=(10, 8))
correlation_data = df[['Engine size', 'Price', 'Mileage', 'Year of manufacture']].corr()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('发动机大小、价格、里程和生产年份的相关性')
plt.tight_layout()

# 保存图表
plt.savefig('engine_size_analysis.png')
plt.close('all')

print("\n分析完成，结果已保存到 engine_size_stats.csv 和 engine_size_analysis.png")