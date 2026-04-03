#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost 生存分析模型使用示例
展示如何使用新添加到 model.py 中的 XGBoost 模型
"""

import numpy as np
import pandas as pd
from model import XGBoostSurvivalModel, XGBoostModelConfig


def example_1_basic_usage():
    """示例1: 基本使用方法"""
    print("=" * 60)
    print("示例1: XGBoost 生存分析模型基本使用")
    print("=" * 60)
    
    # 1. 创建模型实例
    model = XGBoostSurvivalModel(
        xgb_params={
            'max_depth': 5,
            'learning_rate': 0.01,
            'num_round': 500
        },
        num_time_bins=10
    )
    
    # 2. 准备训练数据（示例数据）
    n_samples = 100
    n_features = 5
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    duration_train = np.random.exponential(scale=100, size=n_samples)
    event_train = np.random.binomial(1, 0.7, size=n_samples)
    
    # 3. 训练模型
    model.fit(X_train, duration_train, event_train)
    
    # 4. 预测
    X_test = pd.DataFrame(
        np.random.randn(20, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    surv_probs, time_points = model.predict_survival(X_test)
    
    print(f"\n预测结果:")
    print(f"  - 样本数: {surv_probs.shape[0]}")
    print(f"  - 时间点数: {surv_probs.shape[1]}")
    print(f"  - 时间点: {time_points}")
    
    # 5. 评估模型
    duration_test = np.random.exponential(scale=100, size=20)
    event_test = np.random.binomial(1, 0.7, size=20)
    c_index = model.calculate_concordance_index(X_test, duration_test, event_test)
    print(f"\nC-index: {c_index:.4f}")
    
    # 6. 获取特征重要性
    importance = model.get_feature_importance()
    print(f"\n特征重要性:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.2f}")
    
    print()


def example_2_with_validation():
    """示例2: 使用验证集训练"""
    print("=" * 60)
    print("示例2: 使用验证集训练 XGBoost 模型")
    print("=" * 60)
    
    # 准备数据
    n_train = 100
    n_val = 30
    n_features = 5
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    duration_train = np.random.exponential(scale=100, size=n_train)
    event_train = np.random.binomial(1, 0.7, size=n_train)
    
    X_val = pd.DataFrame(
        np.random.randn(n_val, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    duration_val = np.random.exponential(scale=100, size=n_val)
    event_val = np.random.binomial(1, 0.7, size=n_val)
    
    # 创建并训练模型（带验证集）
    model = XGBoostSurvivalModel(num_time_bins=10)
    model.fit(
        X_train, duration_train, event_train,
        X_val=X_val, duration_val=duration_val, event_val=event_val,
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    # 保存模型
    model.save_model('/tmp/xgboost_survival_model.json')
    
    # 加载模型
    new_model = XGBoostSurvivalModel(num_time_bins=10)
    new_model.load_model('/tmp/xgboost_survival_model.json')
    
    print()


def example_3_using_config():
    """示例3: 使用配置类创建模型"""
    print("=" * 60)
    print("示例3: 使用 XGBoostModelConfig 创建模型")
    print("=" * 60)
    
    # 创建配置
    config = XGBoostModelConfig(
        xgb_params={
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'num_round': 300
        },
        num_time_bins=15
    )
    
    print(f"配置: {config}")
    
    # 从配置创建模型
    model = config.create_model()
    
    # 准备数据
    n_samples = 100
    n_features = 5
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    duration_train = np.random.exponential(scale=100, size=n_samples)
    event_train = np.random.binomial(1, 0.7, size=n_samples)
    
    # 训练
    model.fit(X_train, duration_train, event_train)
    
    print(f"模型训练完成！")
    print()


def example_4_custom_time_bins():
    """示例4: 使用自定义时间区间"""
    print("=" * 60)
    print("示例4: 使用自定义时间区间")
    print("=" * 60)
    
    # 自定义时间区间（例如：0, 30, 90, 180, 365, 730, 1095天）
    custom_time_bins = np.array([0, 30, 90, 180, 365, 730, 1095])
    
    # 创建模型
    model = XGBoostSurvivalModel(
        time_bins=custom_time_bins,
        xgb_params={'num_round': 300}
    )
    
    # 准备数据
    n_samples = 100
    n_features = 5
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    duration_train = np.random.exponential(scale=365, size=n_samples)  # 平均1年
    event_train = np.random.binomial(1, 0.6, size=n_samples)
    
    # 训练
    model.fit(X_train, duration_train, event_train)
    
    print(f"使用的时间区间: {model.time_bins}")
    print()


if __name__ == '__main__':
    # 运行所有示例
    example_1_basic_usage()
    example_2_with_validation()
    example_3_using_config()
    example_4_custom_time_bins()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

