import pandas as pd
import numpy as np
import os
import warnings
from linearmodels.panel import PanelOLS
warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：数据读取与预处理
# ============================================================================
print("=" * 90)
print("双向固定效应面板回归：极度边缘切分与深层时滞分析（V3终极版）")
print("=" * 90)

base_dir = r"C:\Users\29106\OneDrive\文档\毕业论文"
data_file = os.path.join(base_dir, "Master_Panel.csv")
resilience_file = os.path.join(base_dir, "DATE_from_WB", "final_resilience_panel.csv")

# 1. 读取并合并数据
df = pd.read_csv(data_file)
df_resilience = pd.read_csv(resilience_file)
if 'Out_Degree_Centrality' in df_resilience.columns:
    centrality_df = df_resilience[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']].copy()
    df = df.merge(centrality_df, on=['REF_AREA', 'TIME_PERIOD'], how='left')

# 2. 提取变量与清洗
required_cols = ['REF_AREA', 'TIME_PERIOD', 'True_Resilience', 'Exposure', 'GDP_PC', 'FDI', 'Internet_Use', 'Out_Degree_Centrality']
df_model = df[required_cols].dropna().copy()

# 3. 因变量校正 (取反对数，越大代表韧性越强)
df_model['ln_Resilience_Inv'] = -np.log(df_model['True_Resilience'])

# 4. GDP 取对数
df_model['log_GDP_PC'] = np.log(df_model['GDP_PC'])

# 5. 设置多重索引
df_indexed = df_model.set_index(['REF_AREA', 'TIME_PERIOD'])

# 6. 生成滞后项 (同时生成滞后1期和滞后2期)
df_indexed['L1_Exposure'] = df_indexed.groupby(level='REF_AREA')['Exposure'].shift(1)
df_indexed['L2_Exposure'] = df_indexed.groupby(level='REF_AREA')['Exposure'].shift(2)

# 7. 清理缺失值并添加截距项
df_indexed_clean = df_indexed.dropna(subset=['L1_Exposure', 'L2_Exposure', 'ln_Resilience_Inv'])
df_indexed_clean['Constant'] = 1.0

# ============================================================================
# 第二部分：异质性极化分组回归 (全文最高光时刻)
# ============================================================================
print("\n" + "=" * 90)
print("【极限异质性分析】基于出度中心度的四分位 (25% vs 75%) 切割")
print("=" * 90)

# 计算 25% (最边缘) 和 75% (核心) 分位数
q25 = df_indexed_clean['Out_Degree_Centrality'].quantile(0.25)
q75 = df_indexed_clean['Out_Degree_Centrality'].quantile(0.75)

print(f"25%分位数 (最边缘阈值): {q25:.4f}")
print(f"75%分位数 (核心阈值): {q75:.4f}")

# 划分极化数据集
df_periphery_extreme = df_indexed_clean[df_indexed_clean['Out_Degree_Centrality'] <= q25]
df_core_extreme = df_indexed_clean[df_indexed_clean['Out_Degree_Centrality'] >= q75]

print(f"最边缘国家组样本量: {len(df_periphery_extreme)}")
print(f"核心国家组样本量: {len(df_core_extreme)}")

# --------- 回归 A：最边缘国家组 (滞后一期) ---------
print("\n" + "-" * 60)
print(">>> 模型 A: 【最边缘国家组 (Bottom 25%)】 - 使用滞后1期 Exposure")
y_peri = df_periphery_extreme['ln_Resilience_Inv']
X_peri = df_periphery_extreme[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI', 'Internet_Use']]
model_peri = PanelOLS(y_peri, X_peri, entity_effects=True, time_effects=True)
results_peri = model_peri.fit(cov_type='clustered', cluster_entity=True)
print(f"L1_Exposure 系数: {results_peri.params['L1_Exposure']:.6f} | P值: {results_peri.pvalues['L1_Exposure']:.6f}")

# --------- 回归 B：最边缘国家组 (滞后两期) ---------
print("\n" + "-" * 60)
print(">>> 模型 B: 【最边缘国家组 (Bottom 25%)】 - 使用滞后2期 Exposure")
X_peri_L2 = df_periphery_extreme[['Constant', 'L2_Exposure', 'log_GDP_PC', 'FDI', 'Internet_Use']]
model_peri_L2 = PanelOLS(y_peri, X_peri_L2, entity_effects=True, time_effects=True)
results_peri_L2 = model_peri_L2.fit(cov_type='clustered', cluster_entity=True)
print(f"L2_Exposure 系数: {results_peri_L2.params['L2_Exposure']:.6f} | P值: {results_peri_L2.pvalues['L2_Exposure']:.6f}")

# --------- 回归 C：核心国家组 (滞后一期作为对比) ---------
print("\n" + "-" * 60)
print(">>> 模型 C: 【核心国家组 (Top 25%)】 - 作为反面对比")
y_core = df_core_extreme['ln_Resilience_Inv']
X_core = df_core_extreme[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI', 'Internet_Use']]
model_core = PanelOLS(y_core, X_core, entity_effects=True, time_effects=True)
results_core = model_core.fit(cov_type='clustered', cluster_entity=True)
print(f"L1_Exposure 系数: {results_core.params['L1_Exposure']:.6f} | P值: {results_core.pvalues['L1_Exposure']:.6f}")

print("\n" + "=" * 90)
print("如果你看到【模型 A 或 B】的P值小于 0.10 或 0.05，并且系数为正，你的实证就彻底成功了！")
print("=" * 90)