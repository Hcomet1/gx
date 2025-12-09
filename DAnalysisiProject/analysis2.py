import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号

# 读取数据
df = pd.read_csv('car_sales_preprocessed.csv')

# 维度2：车龄 vs 价格 线性回归分析
plt.figure(figsize=(8, 6))

# 计算相关系数用于标题展示
corr = df['num__Car_Age'].corr(df['Price'])

# 绘制带有回归线的散点图
# scatter_kws={'alpha': 0.1}: 设置透明度，解决数据点重叠问题
# line_kws={'color': 'red'}: 突出显示下降趋势线
sns.regplot(x='num__Car_Age', y='Price', data=df,
            scatter_kws={'alpha': 0.1, 's': 10},
            line_kws={'color': 'red', 'linewidth': 2})

plt.title(f'车龄与价格回归分析 (相关系数: {corr:.2f})')
plt.xlabel('标准化车龄 (Standardized Car Age)')
plt.ylabel('价格')

plt.savefig('dim2_age_price_regression.png')
print("维度2分析完成：图片已保存为 dim2_age_price_regression.png")
plt.show()