import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def run_association(X, y, base_dir):
    """关联规则模块：将价格离散并与类别特征组成一体化one-hot，
    在多支持度/置信度下生成规则，输出Top规则与两类可视化并保存表格。"""
    print("\n--- 4. 关联规则分析: 车型特征关联 (FPGrowth) ---")
    try:
        from mlxtend.frequent_patterns import fpgrowth, association_rules
    except ImportError:
        print("未安装 mlxtend，已跳过关联规则")
        return
    #统一 one-hot
    cat_cols = [c for c in X.columns if c.startswith('cat__')]
    items = X[cat_cols].astype(bool)
    labels = ['Low','Medium','High']
    q1, q2 = np.quantile(y, [1/3, 2/3])  # 将连续价格离散为三档
    bins = [-np.inf, q1, q2, np.inf]
    y_c = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
    price_dummies = pd.get_dummies(y_c, prefix='Price_Discrete')
    onehot = pd.concat([items, price_dummies], axis=1).astype(bool)
    print(f"参与关联分析的特征数: {onehot.shape[1]}")
    #在支持度与置信度网格下生成规则，合并为总规则集
    supports = [0.02, 0.05]
    confs = [0.3, 0.5]
    all_rules = []
    itemsets_min = fpgrowth(onehot, min_support=min(supports), use_colnames=True)
    for s in supports:
        itemsets = fpgrowth(onehot, min_support=s, use_colnames=True)
        for c in confs:
            rules = association_rules(itemsets, metric='confidence', min_threshold=c)
            rules['support_threshold'] = s
            rules['confidence_threshold'] = c
            all_rules.append(rules)
    final_rules = pd.concat(all_rules, ignore_index=True)
    #按 lift 与 confidence 输出 Top 5 强规则
    top_rules = final_rules.sort_values(['lift', 'confidence'], ascending=[False, False]).head(5)
    print("\nTop 5 强关联规则 (按提升度 Lift 排序):")
    for _, row in top_rules.iterrows():
        print(f"规则: {set(row['antecedents'])} -> {set(row['consequents'])} | Lift: {row['lift']:.2f}")

    #Top20 频繁项集柱状图（assoc_itemsets_top20.png）
    plt.figure(figsize=(12, 8))
    sns.barplot(x='support', y='itemsets', data=itemsets_min.sort_values('support', ascending=False).head(20), palette='viridis')
    plt.yticks(range(20), [', '.join(map(str, s)) for s in itemsets_min.sort_values('support', ascending=False).head(20)['itemsets']])
    plt.tight_layout()
    path_itemsets = base_dir / 'assoc_itemsets_top20.png'
    plt.savefig(path_itemsets)
    plt.close()
    print(f"图像: {path_itemsets} - 频繁项集Top20柱状")
    #规则支持度 - 置信度散点图（analysis_association_rules.png）
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='support', y='confidence', size='lift', data=final_rules, hue='lift', palette='viridis', sizes=(20, 200))
    plt.tight_layout()
    path_rules = base_dir / 'analysis_association_rules.png'
    plt.savefig(path_rules)
    plt.close()
    print(f"图像: {path_rules} - 规则支持度-置信度散点")

    itemsets_save = itemsets_min.copy()  # 保存频繁项集与所有规则到CSV
    itemsets_save['itemsets'] = itemsets_save['itemsets'].apply(lambda x: ','.join(sorted(list(x))))
    itemsets_save.to_csv(base_dir / 'fpgrowth_itemsets.csv', index=False, encoding='utf-8')
    rules_save = final_rules.copy()
    rules_save['antecedents'] = rules_save['antecedents'].apply(lambda x: ','.join(sorted(list(x))))
    rules_save['consequents'] = rules_save['consequents'].apply(lambda x: ','.join(sorted(list(x))))
    rules_save.to_csv(base_dir / 'fpgrowth_rules_all.csv', index=False, encoding='utf-8')
    price_rules = rules_save[rules_save['consequents'].str.contains('Price_Discrete_', na=False)]
    price_rules.to_csv(base_dir / 'fpgrowth_rules_target_price.csv', index=False, encoding='utf-8')
