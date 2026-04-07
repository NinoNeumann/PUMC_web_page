"""
与 PUMC_11_17/train/train_3_9_5cross_val.py 中 save_survival_predictions 一致的时间网格与插值。
"""
import numpy as np


def survival_grid_from_surv_df(surv_df, duration=None):
    """
    将 predict_surv_df 的输出转为与训练 CSV 相同的半年网格上的生存概率。

    Args:
        surv_df: DataFrame，index=时间，columns=样本（可为单列）。
        duration: 可选，与训练脚本中 duration 数组；若为 None 则仅用 surv_df.index.max()。

    Returns:
        (surv_grid, time_grid): surv_grid 为 DataFrame，行=样本，列=time_grid 各时刻的生存概率。
    """
    if duration is not None:
        max_time = max(float(np.asarray(duration).max()), float(surv_df.index.max()))
    else:
        max_time = float(surv_df.index.max())
    grid_step = 182.5
    time_grid = np.arange(0, max_time + grid_step, grid_step).astype(int)

    surv_df = surv_df.copy()
    if 0 not in surv_df.index:
        surv_df.loc[0] = 1.0
    surv_df = surv_df.sort_index()
    combined_index = surv_df.index.union(time_grid).unique().sort_values()
    surv_df_extended = surv_df.reindex(combined_index).ffill()
    surv_grid = surv_df_extended.loc[time_grid].T
    return surv_grid, time_grid


def survival_at_days(surv_df, duration=None, days=(365, 730)):
    """
    在给定日历天（如 1 年、2 年）上取生存概率，规则与训练网格一致：
    先构造半年网格，再对每个目标天使用「不超过该天的最大网格列」对应的概率（右连续阶梯）。
    """
    surv_grid, time_grid = survival_grid_from_surv_df(surv_df, duration=duration)
    cols = np.asarray(surv_grid.columns)
    out = {}
    for d in days:
        eligible = cols[cols <= d]
        if len(eligible) == 0:
            out[d] = None
            continue
        t_pick = int(eligible.max())
        val = float(surv_grid[t_pick].iloc[0])
        out[d] = min(max(val, 0.0), 1.0)
    return out
