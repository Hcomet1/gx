import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def run_classification(df, base_dir):
    """分类模块：将价格按训练集分位数离散为三档，
    对候选模型与加权软投票进行5折分层CV评估，选择最优并输出报告与混淆矩阵。"""
    print("\n--- 2. 分类分析: 价格档次预测 ---")
    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    labels_order = ['Low', 'Medium', 'High']
    q1, q2 = np.quantile(y_train, [1/3, 2/3])  # 用训练集分位数防止泄露
    bins = [-np.inf, q1, q2, np.inf]
    y_train_c = pd.cut(y_train, bins=bins, labels=labels_order, include_lowest=True)
    y_test_c = pd.cut(y_test, bins=bins, labels=labels_order, include_lowest=True)

    #定义候选分类模型（逻辑回归、RF、GB），使用分层 5 折交叉验证计算准确率；
    models = {
        'lr': LogisticRegression(C=0.5, max_iter=1000, multi_class='multinomial', solver='lbfgs'),
        'rf': RandomForestClassifier(n_estimators=600, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1),
        'gb': GradientBoostingClassifier(random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_stats = {}
    for name, mdl in models.items():  # 基学习器CV准确率
        scores = cross_val_score(mdl, X_train, y_train_c, scoring='accuracy', cv=cv, n_jobs=-1)
        base_stats[name] = float(np.mean(scores))
    #选择最优模型
    raw = np.array([max(0.0, base_stats[n]) for n in ['lr','rf','gb']])  # 弱模型自动降权
    best = float(np.max(raw))
    mask = raw >= (best - 0.01)
    weights = raw * mask
    weights = (weights / weights.sum()) if weights.sum() > 0 else np.array([1/3,1/3,1/3])
    ens = VotingClassifier([
        ('lr', models['lr']), ('rf', models['rf']), ('gb', models['gb'])
    ], voting='soft', weights=list(weights))
    cv_stats = {k: (base_stats[k], 0.0) for k in base_stats}  # 仅记录均值
    ens_scores = cross_val_score(ens, X_train, y_train_c, scoring='accuracy', cv=cv, n_jobs=-1)
    cv_stats['ens'] = (float(np.mean(ens_scores)), float(np.std(ens_scores)))

    best_name = max(cv_stats.keys(), key=lambda n: cv_stats[n][0])
    best_model = ens if best_name == 'ens' else models[best_name]
    best_model.fit(X_train, y_train_c)
    y_pred = best_model.predict(X_test)

    print("CV准确率 (mean±std):")
    for n, (m, s) in cv_stats.items():
        print(f"  {n}: {m:.4f}±{s:.4f}")
    print(f"集成权重: lr={weights[0]:.2f}, rf={weights[1]:.2f}, gb={weights[2]:.2f}")
    print(f"最终采用: {best_name}")

    #输出分类报告,绘制混淆矩阵热图
    print(classification_report(y_test_c, y_pred, labels=labels_order, target_names=labels_order))
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test_c, y_pred, labels=labels_order)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_order, yticklabels=labels_order)
    plt.title('分类分析：混淆矩阵')
    plt.tight_layout()
    path = base_dir / 'analysis_classification.png'
    plt.savefig(path)
    plt.close()
    acc = accuracy_score(y_test_c, y_pred)
    f1m = f1_score(y_test_c, y_pred, average='macro')
    print(f"图像: {path} - 分类混淆矩阵 | 评估: Acc={acc:.3f}, MacroF1={f1m:.3f}")
