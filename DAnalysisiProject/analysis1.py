import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号

# 读取数据
df = pd.read_csv('car_sales_preprocessed.csv')

# 维度1：价格分布与异常值检测
plt.figure(figsize=(12, 5))

# 子图1：直方图 (看分布形态)
plt.subplot(1, 2, 1)
sns.histplot(df['Price'], kde=True, bins=40, color='skyblue')
plt.title('价格分布直方图 (Price Distribution)')
plt.xlabel('价格')

# 子图2：箱线图 (看异常值)
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Price'], color='lightcoral')
plt.title('价格箱线图 (Price Boxplot)')
plt.xlabel('价格')

plt.tight_layout()

plt.savefig('dim1_price_distribution.png')
print("维度1分析完成：图片已保存为 dim1_price_distribution.png")
plt.show()