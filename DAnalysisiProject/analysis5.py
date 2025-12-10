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

# --- 维度5：各品牌平均价格对比 ---
# 1. 提取品牌列并计算均值
mfr_cols = [c for c in df.columns if 'cat__Manufacturer_' in c]
brand_stats = []

for col in mfr_cols:
    brand_name = col.split('_')[-1] # 从 "cat__Manufacturer_BMW" 提取 "BMW"
    # 计算该品牌为1时的价格均值
    avg_price = df[df[col] == 1]['Price'].mean()
    brand_stats.append({'Brand': brand_name, 'Avg_Price': avg_price})

# 2. 转为DataFrame并排序
brand_df = pd.DataFrame(brand_stats).sort_values('Avg_Price', ascending=False)

# 3. 绘图
plt.figure(figsize=(10, 6))
sns.barplot(x='Brand', y='Avg_Price', data=brand_df, hue='Brand', legend=False, palette='viridis')

plt.title('各品牌平均售价对比 (Brand Premium)')
plt.xlabel('品牌')
plt.ylabel('平均价格')
plt.xticks(rotation=45) # 旋转标签防止重叠

plt.savefig('dim5_brand_premium.png')
print("维度5分析完成：图片已保存为 dim5_brand_premium.png")
plt.show()