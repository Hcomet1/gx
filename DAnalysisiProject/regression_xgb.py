import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor

def run_regression(X_train, y_train, X_test, y_test, base_dir):
    """回归模块：使用候选模型与加权集成，基于5折CV的R2选择最优模型，
    测试集上报告R2/RMSE/MAE/MAPE并保存散点图。"""
    # 候选模型：强化非线性与稳健性
    base = {
        'xgb': XGBRegressor(
            n_estimators=220, learning_rate=0.01,
            max_depth=3, min_child_weight=8, gamma=0.3,
            subsample=0.65, colsample_bytree=0.7,
            reg_lambda=15.0, reg_alpha=0.5,
            tree_method='hist', random_state=42, n_jobs=-1
        ),
        'rf': RandomForestRegressor(n_estimators=600, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1),
        'gb': GradientBoostingRegressor(random_state=42),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_stats = {}
    for name, mdl in base.items():
        # 交叉验证R2均值/方差
        scores = cross_val_score(mdl, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
        cv_stats[name] = (float(np.mean(scores)), float(np.std(scores)))
    # 基于 CV R2 的权重归一化赋值
    weights_raw = np.array([max(0.0, cv_stats[n][0]) for n in ['xgb', 'rf', 'gb']])
    best_cv = float(np.max(weights_raw))
    mask = weights_raw >= (best_cv - 0.01)  # 弱于最佳超过阈值的模型降权到0
    weights = weights_raw * mask
    weights = (weights / weights.sum()) if weights.sum() > 0 else np.array([1/3, 1/3, 1/3])
    ens = VotingRegressor(
        [('xgb', base['xgb']), ('rf', base['rf']), ('gb', base['gb'])],
        weights=list(weights)
    )
    ens_scores = cross_val_score(ens, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
    cv_stats['ens'] = (float(np.mean(ens_scores)), float(np.std(ens_scores)))

    # 选择 CV R2 最优模型用于最终报告
    best_name = max(cv_stats.keys(), key=lambda n: cv_stats[n][0])
    best_model = ens if best_name == 'ens' else base[best_name]
    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)
    # 测试集多指标评估
    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
    mae = float(mean_absolute_error(y_test, pred))
    mape = float(np.mean(np.abs((y_test - pred) / np.maximum(1e-8, np.abs(y_test)))))

    # 打印评估与取舍
    print("CV评估 (R2 mean±std):")
    for n, (m, s) in cv_stats.items():
        print(f"  {n}: {m:.4f}±{s:.4f}")
    print(f"集成权重: xgb={weights[0]:.2f}, rf={weights[1]:.2f}, gb={weights[2]:.2f}")
    print(f"测试集: R2={r2:.4f}, RMSE={rmse:.0f}, MAE={mae:.0f}, MAPE={mape:.2%}")
    print(f"最终采用: {best_name}")

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('回归分析：真实值 vs 预测值')
    plt.xlabel('真实价格')
    plt.ylabel('预测价格')
    plt.tight_layout()
    path = base_dir / 'analysis_regression.png'
    plt.savefig(path)
    plt.close()
    level = '拟合优' if r2 >= 0.90 else ('中等' if r2 >= 0.75 else '较弱')
    print(f"图像: {path} - 回归真实vs预测散点 | 评估: R2={r2:.3f}, RMSE={rmse:.0f} ({level})")
