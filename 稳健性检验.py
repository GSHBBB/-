import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from linearmodels.panel import PanelOLS

warnings.filterwarnings('ignore')

# 设置中文字体
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 0. 数据读取与基础清洗
# ============================================================================
print("加载数据中...")
base_dir = r"C:\Users\29106\OneDrive\文档\毕业论文"
df_model = pd.read_csv(os.path.join(base_dir, "Master_Panel.csv"))
df_resilience = pd.read_csv(os.path.join(base_dir, "DATE_from_WB", "final_resilience_panel.csv"))

if 'Out_Degree_Centrality' not in df_model.columns:
    df_model = df_model.merge(df_resilience[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']], on=['REF_AREA', 'TIME_PERIOD'], how='left')

cols = ['REF_AREA', 'TIME_PERIOD', 'True_Resilience', 'Exposure', 'GDP_PC', 'FDI', 'Out_Degree_Centrality']
df = df_model[cols].dropna().copy()
df['ln_Resilience_Inv'] = -np.log(df['True_Resilience'])
df['log_GDP_PC'] = np.log(df['GDP_PC'])

df_indexed = df.set_index(['REF_AREA', 'TIME_PERIOD'])

# ============================================================================
# 第一部分：连续 DID 平行趋势检验 (Lead-Lag Model)
# ============================================================================
print("\n执行连续DID平行趋势检验 (前置与滞后项模型)...")

# 生成前置项(未来)和滞后项(过去)
df_trend = df_indexed.copy()
df_trend['F2_Exposure'] = df_trend.groupby(level='REF_AREA')['Exposure'].shift(-2) # 未来2年
df_trend['F1_Exposure'] = df_trend.groupby(level='REF_AREA')['Exposure'].shift(-1) # 未来1年
df_trend['Current_Exposure'] = df_trend['Exposure']                                # 当年
df_trend['L1_Exposure'] = df_trend.groupby(level='REF_AREA')['Exposure'].shift(1)  # 滞后1年
df_trend['L2_Exposure'] = df_trend.groupby(level='REF_AREA')['Exposure'].shift(2)  # 滞后2年

df_trend = df_trend.dropna(subset=['F2_Exposure', 'F1_Exposure', 'L1_Exposure', 'L2_Exposure', 'ln_Resilience_Inv', 'Out_Degree_Centrality'])
df_trend['Constant'] = 1.0

# 提取边缘组进行检验
q33_trend = df_trend['Out_Degree_Centrality'].quantile(0.33)
df_trend_peri = df_trend[df_trend['Out_Degree_Centrality'] <= q33_trend]

# 运行包含所有期数的回归
X_trend = df_trend_peri[['Constant', 'F2_Exposure', 'F1_Exposure', 'Current_Exposure', 'L1_Exposure', 'L2_Exposure', 'log_GDP_PC', 'FDI']]
y_trend = df_trend_peri['ln_Resilience_Inv']
model_trend = PanelOLS(y_trend, X_trend, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)

# 提取系数和置信区间绘图
terms = ['F2_Exposure', 'F1_Exposure', 'Current_Exposure', 'L1_Exposure', 'L2_Exposure']
labels = ['Pre-2 (F2)', 'Pre-1 (F1)', 'Current (0)', 'Post-1 (L1)', 'Post-2 (L2)']
coefs = [model_trend.params[term] for term in terms]
errs = [1.96 * model_trend.std_errors[term] for term in terms]

plt.figure(figsize=(9, 5))
plt.errorbar(labels, coefs, yerr=errs, fmt='o-', color='darkblue', capsize=5, markersize=8, linewidth=2)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.title('图 1：连续 DID 平行趋势与动态效应 (Lead-Lag Model)', fontsize=15)
plt.xlabel('相对期数 (Pre代表政策发生前，Post代表发生后)', fontsize=12)
plt.ylabel('估计系数', fontsize=12)
plt.text(0.5, max(coefs)*0.9, '如果Pre不显著(跨越0线)，Post显著大于0，即通过检验！', color='green', fontsize=11)
plt.tight_layout()
plt.show()

# ============================================================================
# 第二部分：截面洗牌安慰剂检验 (极度控制方差版)
# ============================================================================
print("\n执行 500 次截面洗牌安慰剂检验 (防方差爆炸版)...")

df_pla = df_indexed.copy()
df_pla['L1_Exposure'] = df_pla.groupby(level='REF_AREA')['Exposure'].shift(1)
df_pla = df_pla.dropna(subset=['L1_Exposure', 'ln_Resilience_Inv', 'Out_Degree_Centrality'])
df_pla['Constant'] = 1.0

# 提取真实边缘组并计算真实系数
q33 = df_pla['Out_Degree_Centrality'].quantile(0.33)
df_peri = df_pla[df_pla['Out_Degree_Centrality'] <= q33]
res_true = PanelOLS(df_peri['ln_Resilience_Inv'], df_peri[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']], entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
true_coef = res_true.params['L1_Exposure']

placebo_coefs = []

for i in range(500):
    # 【神级修正】：在每一个年份的截面上独立洗牌，完美保持每年X的方差不变！
    df_pla_fake = df_pla.copy()
    df_pla_fake['L1_Exposure_fake'] = df_pla_fake.groupby('TIME_PERIOD')['L1_Exposure'].transform(np.random.permutation)
    
    # 切分边缘组
    df_fake_peri = df_pla_fake[df_pla_fake['Out_Degree_Centrality'] <= q33]
    
    X_fake = df_fake_peri[['Constant', 'L1_Exposure_fake', 'log_GDP_PC', 'FDI']]
    y_fake = df_fake_peri['ln_Resilience_Inv']
    
    try:
        res_fake = PanelOLS(y_fake, X_fake, entity_effects=True, time_effects=True).fit()
        placebo_coefs.append(res_fake.params['L1_Exposure_fake'])
    except:
        continue
        
    if (i+1) % 100 == 0:
        print(f"已完成 {i+1}/500 次模拟...")

plt.figure(figsize=(9, 5))
sns.kdeplot(placebo_coefs, fill=True, color='skyblue', edgecolor='royalblue', linewidth=2)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='真实值为 0')
plt.axvline(x=true_coef, color='red', linestyle='-', linewidth=2, label=f'真实系数 ({true_coef:.4f})')
plt.title('图 2：安慰剂检验 (年度截面置换控制方差法)', fontsize=15)
plt.xlabel('模拟回归估计系数', fontsize=12)
plt.ylabel('概率密度', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

print("\n稳健性检验运行完毕！")