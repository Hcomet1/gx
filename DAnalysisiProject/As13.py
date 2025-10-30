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

# 按生产年份分析汽车价格趋势和折旧率
print("按生产年份分析汽车价格趋势和折旧率")

# 按生产年份分组计算统计特征
year_stats = df.groupby('Year of manufacture').agg({
    'Price': ['mean', 'median', 'std', 'min', 'max', 'count'],
    'Mileage': ['mean', 'median']
}).reset_index()

# 打印统计结果
print("\n生产年份统计特征:")
print(year_stats)

# 保存统计结果到CSV
year_stats.to_csv('year_stats.csv')

# 计算每年的折旧率（相对于最新年份的平均价格）
# 获取最新年份和其平均价格
latest_year = year_stats['Year of manufacture'].max()
latest_year_price = year_stats.loc[year_stats['Year of manufacture'] == latest_year, ('Price', 'mean')].values[0]

# 计算每年的折旧率
year_stats['depreciation_rate'] = 1 - (year_stats[('Price', 'mean')] / latest_year_price)

# 可视化分析
plt.figure(figsize=(15, 10))

# 1. 生产年份与平均价格的关系（价格趋势）
plt.subplot(2, 2, 1)
sns.lineplot(x='Year of manufacture', y=('Price', 'mean'), data=year_stats, marker='o')
plt.title('汽车价格随生产年份的变化趋势')
plt.xlabel('生产年份')
plt.ylabel('平均价格')
plt.grid(True)

# 2. 生产年份与折旧率的关系
plt.subplot(2, 2, 2)
sns.lineplot(x='Year of manufacture', y='depreciation_rate', data=year_stats, marker='o')
plt.title('汽车折旧率随生产年份的变化')
plt.xlabel('生产年份')
plt.ylabel('折旧率（相对于最新年份）')
plt.grid(True)

# 3. 生产年份与平均里程的关系
plt.subplot(2, 2, 3)
sns.lineplot(x='Year of manufacture', y=('Mileage', 'mean'), data=year_stats, marker='o')
plt.title('汽车平均里程随生产年份的变化')
plt.xlabel('生产年份')
plt.ylabel('平均里程')
plt.grid(True)

# 4. 散点图：生产年份、里程与价格的关系
plt.subplot(2, 2, 4)
scatter = plt.scatter(df['Year of manufacture'], df['Price'], c=df['Mileage'], cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='里程')
plt.title('生产年份、里程与价格的关系')
plt.xlabel('生产年份')
plt.ylabel('价格')
plt.grid(True)
plt.tight_layout()

# 5. 箱线图：生产年份与价格分布（每5年一组）
plt.figure(figsize=(15, 8))
# 创建年份分组
df['year_group'] = pd.cut(df['Year of manufacture'], bins=range(df['Year of manufacture'].min(), df['Year of manufacture'].max()+6, 5), right=False)
sns.boxplot(x='year_group', y='Price', data=df)
plt.title('不同年份组汽车价格分布')
plt.xlabel('生产年份组')
plt.ylabel('价格')
plt.xticks(rotation=45)
plt.tight_layout()

# 6. 热图：生产年份、价格、里程的相关性
plt.figure(figsize=(10, 8))
correlation_data = df[['Year of manufacture', 'Price', 'Mileage', 'Engine size']].corr()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('生产年份、价格、里程和发动机大小的相关性')
plt.tight_layout()

# 保存图表
plt.savefig('year_analysis.png')
plt.close('all')

print("\n分析完成，结果已保存到 year_stats.csv 和 year_analysis.png")