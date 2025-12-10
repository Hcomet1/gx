import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格1
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号

# 读取数据
df = pd.read_csv('car_sales_preprocessed.csv')

# 维度4：引擎大小分段价格分布
plt.figure(figsize=(10, 6))

#为了可视化，我们将标准化后的引擎大小进行简单的分箱（Binning）,连续变量离散化为：小排量、中排量、大排量
df['Engine_Group'] = pd.cut(df['num__Engine size'], bins=3, labels=['小排量', '中排量', '大排量'])

sns.violinplot(x='Engine_Group', y='Price', data=df, hue='Engine_Group', legend=False, palette='muted')

plt.title('不同排量组别的价格分布形态 (Violin Plot)')
plt.xlabel('排量组别 (Based on Standardized Size)')
plt.ylabel('价格')

plt.savefig('dim4_engine_price_violin.png')
print("维度4分析完成：图片已保存为 dim4_engine_price_violin.png")
plt.show()