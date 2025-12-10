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

# 维度3：里程 vs 价格 密度分析
plt.figure(figsize=(9, 7))

# 使用 Hexbin 图代替散点图,颜色越深代表该区域车辆越多
hb = plt.hexbin(df['num__Mileage'], df['Price'], gridsize=25, cmap='Blues', mincnt=1)

plt.colorbar(hb, label='车辆数量 (Count)')
plt.title('里程与价格密度分布 (Hexbin Plot)')
plt.xlabel('标准化里程 (Standardized Mileage)')
plt.ylabel('价格')

plt.savefig('dim3_mileage_price_density.png')
print("维度3分析完成：图片已保存为 dim3_mileage_price_density.png")
plt.show()