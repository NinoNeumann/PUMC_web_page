#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生存分析数据集处理类
整合了数据加载、预处理、特征工程、数据分割等流程
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import os
import warnings
warnings.filterwarnings('ignore')


class DataConfig:
    """
    数据配置类，用于控制数据处理流程
    """
    def __init__(self, 
                 data_path=None,
                 test_path=None,
                 prediction_target='survival',
                 manual_numeric_features=None,
                 manual_categorical_features=None,
                 test_size=0.0,
                 val_size=0.0,
                 random_state=42,
                 check_numerical_stability=True,
                 max_value_threshold=100,
                 scaling_strategy='standard'):
        """
        初始化数据配置
        
        Args:
            data_path: 数据文件路径（支持 .xlsx 和 .csv）
            prediction_target: 预测目标，可选 'survival', 'event_only', 'time_only'
            manual_numeric_features: 手动指定的数值特征列表（列名），如果为None则自动识别
            manual_categorical_features: 手动指定的分类特征列表（列名），如果为None则自动识别
            test_size: 测试集比例，默认0.0。注意：如果提供了test_path，此参数将被忽略，设为0。
            val_size: 验证集比例，默认0.0
            random_state: 随机种子，默认42
            check_numerical_stability: 是否进行数值稳定性检查（v2版本特有），默认True
            max_value_threshold: 数值范围阈值，超过此值会进行额外缩放，默认100
            scaling_strategy: 数值特征缩放策略，'standard' / 'minmax' / 'robust'
        """
        self.data_path = data_path
        self.test_path = test_path
        self.prediction_target = prediction_target
        self.manual_numeric_features = manual_numeric_features
        self.manual_categorical_features = manual_categorical_features
        
        # 如果提供了独立测试集，则不需要从训练集中划分测试集
        if self.test_path is not None:
            self.test_size = 0.0
        else:
            self.test_size = test_size
            
        self.val_size = val_size
        self.random_state = random_state
        self.check_numerical_stability = check_numerical_stability
        self.max_value_threshold = max_value_threshold
        self.scaling_strategy = scaling_strategy


class SurvivalDataset:
    """
    生存分析数据集类
    整合了完整的数据处理流程
    """
    
    def __init__(self, config, verbose=True):
        """
        初始化数据集
        
        Args:
            config: DataConfig对象，包含数据处理配置
            verbose: 是否打印详细信息，默认True
        """
        self.config = config
        self.verbose = verbose
        
        # 数据存储
        self.df = None
        self.df_encoded = None
        self.encoders = {}
        self.train_val_indices = None # 用于标记训练/验证集样本索引 (当指定test_path时使用)
        self.test_indices = None      # 用于标记测试集样本索引 (当指定test_path时使用)
        
        # 特征信息
        self.feature_cols = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.x_mapper = None
        
        # 分割后的数据
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.duration_train = None
        self.duration_val = None
        self.duration_test = None
        self.event_train = None
        self.event_val = None
        self.event_test = None
        
        # 转换后的特征
        self.x_train = None
        self.x_val = None
        self.x_test = None
    
    def load_data(self):
        """
        加载数据
        """
        if self.verbose:
            print("=" * 60)
            print("1. 加载数据")
            print("=" * 60)
        
        if self.config.data_path is None:
            raise ValueError("data_path 未指定")
        
        # 读取数据 - 支持 Excel 和 CSV
        def read_file(path):
            if path.endswith('.csv'):
                return pd.read_csv(path)
            else:
                return pd.read_excel(path)

        self.df = read_file(self.config.data_path)
        
        if self.config.test_path:
            if self.verbose:
                print(f"训练数据文件: {self.config.data_path}")
                print(f"测试数据文件: {self.config.test_path}")
            
            df_test = read_file(self.config.test_path)
            
            # 记录索引范围
            train_len = len(self.df)
            test_len = len(df_test)
            self.train_val_indices = range(0, train_len)
            self.test_indices = range(train_len, train_len + test_len)
            
            # 合并数据以统一处理编码
            self.df = pd.concat([self.df, df_test], ignore_index=True)
            
            if self.verbose:
                print(f"合并后数据形状: {self.df.shape} (Train: {train_len}, Test: {test_len})")
        else:
            if self.verbose:
                print(f"数据文件: {self.config.data_path}")
                print(f"数据形状: {self.df.shape}")

        if self.verbose:
            print(f"\n数据列: {self.df.columns.tolist()}")
            print(f"\n前5行数据:")
            print(self.df.head())
            
            # 检查目标变量
            if 'death' in self.df.columns:
                print(f"\n事件分布:")
                print(self.df['death'].value_counts())
            if 'survday' in self.df.columns:
                print(f"\n生存时间统计:")
                print(self.df['survday'].describe())
        
        return self.df
    
    def encode_categorical_features(self):
        """
        编码分类特征
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("2. 编码分类特征")
            print("=" * 60)
        
        if self.df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        self.df_encoded = self.df.copy()
        
        # 识别分类列（object类型）
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        if self.verbose:
            print(f"分类列: {categorical_cols}")
        
        # 编码器字典，用于保存编码信息
        self.encoders = {}
        
        for col in categorical_cols:
            if self.verbose:
                print(f"\n编码 '{col}':")
                print(f"  唯一值: {self.df[col].unique()}")
            
            le = LabelEncoder()
            self.df_encoded[col] = le.fit_transform(self.df[col])
            self.encoders[col] = le
            
            if self.verbose:
                print(f"  编码后: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return self.df_encoded, self.encoders
    
    def prepare_features(self):
        """
        准备特征和标签
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("3. 准备特征和标签")
            print("=" * 60)
        
        if self.df_encoded is None:
            raise ValueError("请先调用 encode_categorical_features() 编码分类特征")
        
        # 特征列（除去 death 和 survday）
        self.feature_cols = [col for col in self.df_encoded.columns 
                            if col not in ['death', 'survday']]
        
        if self.verbose:
            print(f"特征列 ({len(self.feature_cols)}): {self.feature_cols}")
        
        # 分离数值型和分类型特征
        X = self.df_encoded[self.feature_cols]
        
        # 判断使用哪种模式
        if self.config.manual_numeric_features is not None:
            # 模式1: 手动指定数值特征（优先级最高）
            if self.verbose:
                print(f"\n使用手动指定的数值特征（其余为分类特征）")
            self.numeric_cols = [col for col in self.config.manual_numeric_features 
                               if col in self.feature_cols]
            self.categorical_cols = [col for col in self.feature_cols 
                                   if col not in self.numeric_cols]
        elif self.config.manual_categorical_features is not None:
            # 模式2: 手动指定分类特征
            if self.verbose:
                print(f"\n使用手动指定的分类特征（其余为数值特征）")
            self.categorical_cols = [col for col in self.config.manual_categorical_features 
                                   if col in self.feature_cols]
            self.numeric_cols = [col for col in self.feature_cols 
                               if col not in self.categorical_cols]
        else:
            # 模式3: 自动识别模式（基于数据类型）
            if self.verbose:
                print(f"\n自动识别特征类型（基于数据类型）")
            self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.categorical_cols = [col for col in self.feature_cols 
                                   if col not in self.numeric_cols]
        
        if self.verbose:
            print(f"\n数值型特征 ({len(self.numeric_cols)}): {self.numeric_cols}")
            print(f"分类型特征 ({len(self.categorical_cols)}): {self.categorical_cols}")
        
        return self.feature_cols, self.numeric_cols, self.categorical_cols
    
    def create_data_mapper(self):
        """
        创建数据转换器
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("4. 创建数据标准化映射器")
            print("=" * 60)
        
        if self.numeric_cols is None or self.categorical_cols is None:
            raise ValueError("请先调用 prepare_features() 准备特征")
        
        # 使用 ColumnTransformer 替代 DataFrameMapper
        transformers = []
        
        # 数值特征缩放
        if len(self.numeric_cols) > 0:
            strategy = (self.config.scaling_strategy or 'standard').lower()
            if strategy == 'standard':
                num_scaler = StandardScaler()
            elif strategy == 'minmax':
                num_scaler = MinMaxScaler()
            elif strategy == 'robust':
                num_scaler = RobustScaler()
            else:
                raise ValueError(f"未知的缩放策略: {strategy}")
            
            transformers.append(('num', num_scaler, self.numeric_cols))
        
        # 分类特征保持标签编码后的取值，不再做 One-Hot
        if len(self.categorical_cols) > 0:
            transformers.append((
                'cat',
                'passthrough',
                self.categorical_cols
            ))
        
        self.x_mapper = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        if self.verbose:
            print(f"将使用 {self.config.scaling_strategy} 缩放 {len(self.numeric_cols)} 个数值特征")
            print(f"保留 {len(self.categorical_cols)} 个分类特征的整数标签编码")
        
        return self.x_mapper
    
    def split_data(self):
        """
        分割数据集
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("5. 分割数据集")
            print("=" * 60)
            print(f"预测目标: {self.config.prediction_target}")
        
        if self.feature_cols is None:
            raise ValueError("请先调用 prepare_features() 准备特征")
        
        # 提取特征和目标
        X = self.df_encoded[self.feature_cols]
        duration = self.df_encoded['survday'].values
        event = self.df_encoded['death'].values
        
        # 如果指定了独立的测试集
        if self.test_indices is not None:
            if self.verbose:
                print("使用指定的独立测试集...")
            
            # 获取测试集
            X_test = X.iloc[self.test_indices]
            duration_test = duration[self.test_indices]
            event_test = event[self.test_indices]
            
            # 获取训练/验证全集
            X_train_full = X.iloc[self.train_val_indices]
            duration_train_full = duration[self.train_val_indices]
            event_train_full = event[self.train_val_indices]
            
            # 根据预测目标决定分层变量
            if self.config.prediction_target == 'time_only':
                stratify_var = event_train_full
            else:
                stratify_var = event_train_full

            # 从训练全集中分离验证集
            if self.config.val_size > 0:
                # 注意：这里val_size是相对于train_full的比例
                X_train, X_val, duration_train, duration_val, event_train, event_val = train_test_split(
                    X_train_full, duration_train_full, event_train_full,
                    test_size=self.config.val_size,
                    random_state=self.config.random_state,
                    stratify=stratify_var if self.config.prediction_target != 'time_only' else None
                )
            else:
                if self.verbose:
                    print("不进行验证集分割 (val_size=0)")
                X_train = X_train_full
                duration_train = duration_train_full
                event_train = event_train_full
                
                # 创建空的验证集
                X_val = X.iloc[:0]
                duration_val = np.array([])
                event_val = np.array([])
            
        else:
            # 原有的分割逻辑
            if self.verbose:
                print("使用随机分割生成训练/验证/测试集...")
                
            # 根据预测目标决定分层变量
            if self.config.prediction_target == 'time_only':
                # 回归任务，不使用分层（或基于事件分层）
                stratify_var = event
            else:
                # 分类或生存分析，基于事件分层
                stratify_var = event
            
            # 首先分离出测试集
            if self.config.test_size > 0:
                X_temp, X_test, duration_temp, duration_test, event_temp, event_test = train_test_split(
                    X, duration, event, 
                    test_size=self.config.test_size, 
                    random_state=self.config.random_state,
                    stratify=stratify_var
                )
            else:
                if self.verbose:
                    print("不进行测试集分割 (test_size=0)")
                X_temp = X
                duration_temp = duration
                event_temp = event
                
                # 创建空的测试集
                X_test = X.iloc[:0]
                duration_test = np.array([])
                event_test = np.array([])
            
            # 然后从剩余数据中分离验证集
            # 计算剩余数据的比例
            remaining_ratio = 1.0 - self.config.test_size
            
            if self.config.val_size > 0 and remaining_ratio > 0:
                val_ratio = self.config.val_size / remaining_ratio
                # 限制 val_ratio 在 (0, 1) 之间
                val_ratio = min(max(val_ratio, 0.0), 1.0)
                
                if 0 < val_ratio < 1:
                    X_train, X_val, duration_train, duration_val, event_train, event_val = train_test_split(
                        X_temp, duration_temp, event_temp,
                        test_size=val_ratio,
                        random_state=self.config.random_state,
                        stratify=event_temp if self.config.prediction_target != 'time_only' else None
                    )
                else:
                    # 如果比例异常（如全部分给验证集），默认不分割或全部分配
                    if val_ratio >= 1.0:
                         if self.verbose:
                            print("警告: 验证集比例占满剩余数据，训练集将为空")
                         X_val = X_temp
                         duration_val = duration_temp
                         event_val = event_temp
                         X_train = X.iloc[:0]
                         duration_train = np.array([])
                         event_train = np.array([])
                    else:
                         X_train = X_temp
                         duration_train = duration_temp
                         event_train = event_temp
                         X_val = X.iloc[:0]
                         duration_val = np.array([])
                         event_val = np.array([])
            else:
                if self.verbose and self.config.val_size <= 0:
                    print("不进行验证集分割 (val_size=0)")
                X_train = X_temp
                duration_train = duration_temp
                event_train = event_temp
                
                # 创建空的验证集
                X_val = X.iloc[:0]
                duration_val = np.array([])
                event_val = np.array([])
        
        # 保存分割后的数据
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.duration_train = duration_train
        self.duration_val = duration_val
        self.duration_test = duration_test
        self.event_train = event_train
        self.event_val = event_val
        self.event_test = event_test
        
        if self.verbose:
            print(f"训练集: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(self.df_encoded)*100:.1f}%)")
            if self.config.prediction_target != 'time_only':
                print(f"  事件数: {event_train.sum()}, 删失数: {(1-event_train).sum()}")
            else:
                print(f"  生存时间均值: {duration_train.mean():.1f} 天")
            
            print(f"验证集: {X_val.shape[0]} 样本 ({X_val.shape[0]/len(self.df_encoded)*100:.1f}%)")
            if self.config.prediction_target != 'time_only':
                print(f"  事件数: {event_val.sum()}, 删失数: {(1-event_val).sum()}")
            else:
                print(f"  生存时间均值: {duration_val.mean():.1f} 天")
            
            print(f"测试集: {X_test.shape[0]} 样本 ({X_test.shape[0]/len(self.df_encoded)*100:.1f}%)")
            if self.config.prediction_target != 'time_only':
                print(f"  事件数: {event_test.sum()}, 删失数: {(1-event_test).sum()}")
            else:
                print(f"  生存时间均值: {duration_test.mean():.1f} 天")
        
        return (self.X_train, self.X_val, self.X_test, 
                self.duration_train, self.duration_val, self.duration_test,
                self.event_train, self.event_val, self.event_test)
    
    def transform_features(self):
        """
        转换特征
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("6. 特征转换和标准化")
            print("=" * 60)
        
        if self.x_mapper is None:
            raise ValueError("请先调用 create_data_mapper() 创建数据映射器")
        if self.X_train is None:
            raise ValueError("请先调用 split_data() 分割数据")
        
        # 拟合并转换训练集
        self.x_train = self.x_mapper.fit_transform(self.X_train).astype('float32')
        # 转换验证集和测试集
        if self.X_val.shape[0] > 0:
            self.x_val = self.x_mapper.transform(self.X_val).astype('float32')
        else:
            self.x_val = np.empty((0, self.x_train.shape[1]), dtype='float32')
            
        if self.X_test.shape[0] > 0:
            self.x_test = self.x_mapper.transform(self.X_test).astype('float32')
        else:
            self.x_test = np.empty((0, self.x_train.shape[1]), dtype='float32')
        
        # 数值稳定性检查（v2版本特有）
        if self.config.check_numerical_stability:
            if self.verbose:
                print("\n数值稳定性检查:")
            
            # 检查NaN
            if np.isnan(self.x_train).any():
                nan_count = np.isnan(self.x_train).sum()
                if self.verbose:
                    print(f"⚠️ 警告: 训练集包含 {nan_count} 个 NaN 值")
                self.x_train = np.nan_to_num(self.x_train, nan=0.0)
            
            if self.x_val.shape[0] > 0 and np.isnan(self.x_val).any():
                nan_count = np.isnan(self.x_val).sum()
                if self.verbose:
                    print(f"⚠️ 警告: 验证集包含 {nan_count} 个 NaN 值")
                self.x_val = np.nan_to_num(self.x_val, nan=0.0)
            
            if self.x_test.shape[0] > 0 and np.isnan(self.x_test).any():
                nan_count = np.isnan(self.x_test).sum()
                if self.verbose:
                    print(f"⚠️ 警告: 测试集包含 {nan_count} 个 NaN 值")
                self.x_test = np.nan_to_num(self.x_test, nan=0.0)
            
            # 检查无穷大
            if np.isinf(self.x_train).any():
                inf_count = np.isinf(self.x_train).sum()
                if self.verbose:
                    print(f"⚠️ 警告: 训练集包含 {inf_count} 个无穷大值")
                self.x_train = np.nan_to_num(self.x_train, posinf=0.0, neginf=0.0)
            
            # 检查数值范围
            train_max = np.abs(self.x_train).max()
            train_mean = np.abs(self.x_train).mean()
            if self.verbose:
                print(f"训练集数值范围: max={train_max:.4f}, mean={train_mean:.4f}")
            
            # 如果数值过大，额外标准化
            if train_max > self.config.max_value_threshold:
                if self.verbose:
                    print(f"⚠️ 警告: 数值范围过大，进行额外缩放")
                scale_factor = train_max / 10
                self.x_train = self.x_train / scale_factor
                if self.x_val.shape[0] > 0:
                    self.x_val = self.x_val / scale_factor
                if self.x_test.shape[0] > 0:
                    self.x_test = self.x_test / scale_factor
                if self.verbose:
                    print(f"缩放因子: {scale_factor:.4f}")
            
            if self.verbose:
                print(f"✓ 数值检查完成")
        
        if self.verbose:
            print(f"\n转换后的特征维度: {self.x_train.shape[1]}")
            print(f"训练集特征形状: {self.x_train.shape}")
            print(f"验证集特征形状: {self.x_val.shape}")
            print(f"测试集特征形状: {self.x_test.shape}")
        
        return self.x_train, self.x_val, self.x_test
    
    def process_all(self):
        """
        执行完整的数据处理流程
        """
        self.load_data()
        self.encode_categorical_features()
        self.prepare_features()
        self.create_data_mapper()
        self.split_data()
        self.transform_features()
        
        return {
            'x_train': self.x_train,
            'x_val': self.x_val,
            'x_test': self.x_test,
            'duration_train': self.duration_train,
            'duration_val': self.duration_val,
            'duration_test': self.duration_test,
            'event_train': self.event_train,
            'event_val': self.event_val,
            'event_test': self.event_test,
            'encoders': self.encoders,
            'x_mapper': self.x_mapper,
            'feature_cols': self.feature_cols,
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols
        }
    
    def get_train_data(self):
        """
        获取训练数据（用于模型训练）
        
        Returns:
            tuple: (x_train, y_train) 或 (x_train, (duration_train, event_train))
        """
        if self.x_train is None:
            raise ValueError("请先调用 process_all() 或 transform_features() 处理数据")
        
        if self.config.prediction_target == 'survival':
            y_train = (self.duration_train, self.event_train)
        elif self.config.prediction_target == 'event_only':
            y_train = self.event_train.astype('float32').reshape(-1, 1)
        elif self.config.prediction_target == 'time_only':
            y_train = self.duration_train.astype('float32').reshape(-1, 1)
        else:
            raise ValueError(f"未知的预测目标: {self.config.prediction_target}")
        
        return self.x_train, y_train
    
    def get_val_data(self):
        """
        获取验证数据
        
        Returns:
            tuple: (x_val, y_val) 或 (x_val, (duration_val, event_val))
        """
        if self.x_val is None:
            raise ValueError("请先调用 process_all() 或 transform_features() 处理数据")
        
        if self.config.prediction_target == 'survival':
            y_val = (self.duration_val, self.event_val)
        elif self.config.prediction_target == 'event_only':
            y_val = self.event_val.astype('float32').reshape(-1, 1)
        elif self.config.prediction_target == 'time_only':
            y_val = self.duration_val.astype('float32').reshape(-1, 1)
        else:
            raise ValueError(f"未知的预测目标: {self.config.prediction_target}")
        
        return self.x_val, y_val
    
    def get_test_data(self):
        """
        获取测试数据
        
        Returns:
            tuple: (x_test, duration_test, event_test)
        """
        if self.x_test is None:
            raise ValueError("请先调用 process_all() 或 transform_features() 处理数据")
        
        return self.x_test, self.duration_test, self.event_test


# ========== 便捷函数 ==========
def create_dataset(data_path, test_path=None, prediction_target='survival', 
                  manual_numeric_features=None, manual_categorical_features=None,
                  test_size=0.2, val_size=0.1, random_state=42,
                  check_numerical_stability=True, verbose=True,
                  scaling_strategy='standard'):
    """
    便捷函数：创建并处理数据集
    
    Args:
        data_path: 数据文件路径
        test_path: 测试数据文件路径 (可选)
        prediction_target: 预测目标，可选 'survival', 'event_only', 'time_only'
        manual_numeric_features: 手动指定的数值特征列表
        manual_categorical_features: 手动指定的分类特征列表
        test_size: 测试集比例 (仅当test_path为None时有效)
        val_size: 验证集比例
        random_state: 随机种子
        check_numerical_stability: 是否进行数值稳定性检查
        verbose: 是否打印详细信息
        scaling_strategy: 数值特征缩放策略
    
    Returns:
        dataset: SurvivalDataset对象
    """
    config = DataConfig(
        data_path=data_path,
        test_path=test_path,
        prediction_target=prediction_target,
        manual_numeric_features=manual_numeric_features,
        manual_categorical_features=manual_categorical_features,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        check_numerical_stability=check_numerical_stability,
        scaling_strategy=scaling_strategy
    )
    
    dataset = SurvivalDataset(config, verbose=verbose)
    dataset.process_all()
    
    return dataset


# ========== 示例用法 ==========
if __name__ == '__main__':
    # 示例1: 使用便捷函数
    print("=" * 60)
    print("示例1: 使用便捷函数创建数据集")
    print("=" * 60)
    
    # 注意：需要提供实际的数据路径
    # dataset = create_dataset(
    #     data_path='path/to/your/data.xlsx',
    #     prediction_target='survival',
    #     manual_numeric_features=['年龄', '发病年龄', 'Hb', 'Cr', 'WBC', 'PLT', 'CRP', 'ESR'],
    #     verbose=True
    # )
    # 
    # x_train, y_train = dataset.get_train_data()
    # x_val, y_val = dataset.get_val_data()
    # x_test, duration_test, event_test = dataset.get_test_data()
    # 
    # print(f"\n训练集形状: {x_train.shape}")
    # print(f"验证集形状: {x_val.shape}")
    # print(f"测试集形状: {x_test.shape}")
    
    # 示例2: 使用配置类和数据集类
    print("\n" + "=" * 60)
    print("示例2: 使用配置类和数据集类")
    print("=" * 60)
    
    config = DataConfig(
        data_path=None,  # 需要提供实际路径
        prediction_target='survival',
        manual_numeric_features=['年龄', '发病年龄', 'Hb', 'Cr', 'WBC', 'PLT', 'CRP', 'ESR'],
        test_size=0.2,
        val_size=0.1,
        check_numerical_stability=True
    )
    
    print(f"数据配置: {config.prediction_target}")
    print(f"测试集比例: {config.test_size}")
    print(f"验证集比例: {config.val_size}")
    print(f"数值稳定性检查: {config.check_numerical_stability}")

