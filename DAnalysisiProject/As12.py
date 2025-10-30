import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取处理后的数据
df = pd.read_csv('car_sales_data_processed_all.csv')

# 按燃料类型分析汽车性能和价格的关系
print("按燃料类型分析汽车性能和价格的关系")

# 按燃料类型分组计算统计特征
fuel_stats = df.groupby('Fuel type').agg({
    'Price': ['mean', 'median', 'std', 'min', 'max'],
    'Engine size': ['mean', 'median', 'std', 'min', 'max'],
    'Mileage': ['mean', 'median', 'std', 'min', 'max'],
    'Year of manufacture': ['mean', 'median', 'min', 'max']
}).reset_index()

# 打印统计结果
print("\n燃料类型统计特征:")
print(fuel_stats)

# 保存统计结果到CSV
fuel_stats.to_csv('fuel_type_stats.csv')

# 可视化分析
plt.figure(figsize=(15, 10))

# 1. 燃料类型与平均价格的关系
plt.subplot(2, 2, 1)
sns.barplot(x='Fuel type', y=('Price', 'mean'), data=fuel_stats)
plt.title('各燃料类型汽车平均价格')
plt.ylabel('平均价格')

# 2. 燃料类型与平均发动机大小的关系
plt.subplot(2, 2, 2)
sns.barplot(x='Fuel type', y=('Engine size', 'mean'), data=fuel_stats)
plt.title('各燃料类型汽车平均发动机大小')
plt.ylabel('平均发动机大小')

# 3. 燃料类型与平均里程的关系
plt.subplot(2, 2, 3)
sns.barplot(x='Fuel type', y=('Mileage', 'mean'), data=fuel_stats)
plt.title('各燃料类型汽车平均里程')
plt.ylabel('平均里程')

# 4. 燃料类型与平均生产年份的关系
plt.subplot(2, 2, 4)
sns.barplot(x='Fuel type', y=('Year of manufacture', 'mean'), data=fuel_stats)
plt.title('各燃料类型汽车平均生产年份')
plt.ylabel('平均生产年份')
plt.tight_layout()

# 5. 箱线图：燃料类型与价格分布
plt.figure(figsize=(12, 6))
sns.boxplot(x='Fuel type', y='Price', data=df)
plt.title('各燃料类型汽车价格分布')
plt.ylabel('价格')
plt.tight_layout()

# 6. 小提琴图：燃料类型与发动机大小分布
plt.figure(figsize=(12, 6))
sns.violinplot(x='Fuel type', y='Engine size', data=df)
plt.title('各燃料类型汽车发动机大小分布')
plt.ylabel('发动机大小')
plt.tight_layout()

# 7. 散点图：发动机大小与价格的关系，按燃料类型分组
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Engine size', y='Price', hue='Fuel type', data=df)
plt.title('发动机大小与价格的关系（按燃料类型分组）')
plt.xlabel('发动机大小')
plt.ylabel('价格')
plt.legend(title='燃料类型')
plt.tight_layout()

# 保存图表
plt.savefig('fuel_type_analysis.png')
plt.close('all')

print("\n分析完成，结果已保存到 fuel_type_stats.csv 和 fuel_type_analysis.png")