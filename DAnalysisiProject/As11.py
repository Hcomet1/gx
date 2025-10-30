import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取处理后的数据
df = pd.read_csv('car_sales_data_processed_all.csv')

# 按制造商分析汽车价格、发动机大小和里程的统计特征
print("按制造商分析汽车价格、发动机大小和里程的统计特征")

# 按制造商分组计算统计特征
manufacturer_stats = df.groupby('Manufacturer').agg({
    'Price': ['mean', 'median', 'std', 'min', 'max'],
    'Engine size': ['mean', 'median', 'std', 'min', 'max'],
    'Mileage': ['mean', 'median', 'std', 'min', 'max']
}).reset_index()

# 打印统计结果
print("\n制造商统计特征:")
print(manufacturer_stats)

# 保存统计结果到CSV
manufacturer_stats.to_csv('manufacturer_stats.csv')

# 可视化分析
plt.figure(figsize=(15, 10))

# 1. 制造商与平均价格的关系
plt.subplot(2, 2, 1)
sns.barplot(x='Manufacturer', y=('Price', 'mean'), data=manufacturer_stats)
plt.title('各制造商汽车平均价格')
plt.xticks(rotation=45)
plt.ylabel('平均价格')

# 2. 制造商与平均发动机大小的关系
plt.subplot(2, 2, 2)
sns.barplot(x='Manufacturer', y=('Engine size', 'mean'), data=manufacturer_stats)
plt.title('各制造商汽车平均发动机大小')
plt.xticks(rotation=45)
plt.ylabel('平均发动机大小')

# 3. 制造商与平均里程的关系
plt.subplot(2, 2, 3)
sns.barplot(x='Manufacturer', y=('Mileage', 'mean'), data=manufacturer_stats)
plt.title('各制造商汽车平均里程')
plt.xticks(rotation=45)
plt.ylabel('平均里程')

# 4. 箱线图：制造商与价格分布
plt.figure(figsize=(15, 8))
sns.boxplot(x='Manufacturer', y='Price', data=df)
plt.title('各制造商汽车价格分布')
plt.xticks(rotation=45)
plt.ylabel('价格')
plt.tight_layout()

# 5. 散点图：发动机大小与价格的关系，按制造商分组
plt.figure(figsize=(15, 8))
sns.scatterplot(x='Engine size', y='Price', hue='Manufacturer', data=df)
plt.title('发动机大小与价格的关系（按制造商分组）')
plt.xlabel('发动机大小')
plt.ylabel('价格')
plt.legend(title='制造商')
plt.tight_layout()

# 保存图表
plt.savefig('manufacturer_analysis.png')
plt.close('all')

print("\n分析完成，结果已保存到 manufacturer_stats.csv 和 manufacturer_analysis.png")