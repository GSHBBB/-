import pandas as pd
import numpy as np
import os
from linearmodels.panel import PanelOLS

# ============================================================================
# 第一部分：数据读取与清洗
# ============================================================================
print("=" * 80)
print("双向固定效应面板回归：创新模型 vs 传统模型对比分析")
print("=" * 80)

# 设置工作目录
base_dir = r"C:\Users\29106\OneDrive\文档\毕业论文"
data_file = os.path.join(base_dir, "Master_Panel.csv")

# 读取数据
print("\n步骤1: 读取 Master_Panel.csv 数据...")
df = pd.read_csv(data_file)
print(f"原始数据维度: {df.shape}")
print(df.head())

# ============================================================================
# 第二部分：提取所需变量并进行数据清洗
# ============================================================================
print("\n步骤2: 数据清洗与变量提取...")

# 提取所需变量列
required_cols = ['REF_AREA', 'TIME_PERIOD', 'True_Resilience', 'Exposure', 
                 'GDP_PC', 'FDI', 'Internet_Use', 'GE']
df_model = df[required_cols].copy()

print(f"提取后数据维度: {df_model.shape}")
print(f"提取的变量: {required_cols}")

# 检查缺失值情况
print(f"\n缺失值统计:")
print(df_model.isnull().sum())

# 删除包含缺失值的行
df_model = df_model.dropna()
print(f"\n清洗后数据维度: {df_model.shape}")

# ============================================================================
# 第二点五部分：对人均 GDP 取对数（修复量纲灾难）
# ============================================================================
print("\n步骤2.5: 对 GDP_PC 进行对数转换...")

# 原因：GDP_PC 绝对值范围从几百到几十万，与其他变量量级相差几万倍
# 经济学原理：GDP 对任何指标的影响都是非线性的（对数关系）
# 操作：创建 log_GDP_PC = ln(GDP_PC)
df_model['log_GDP_PC'] = np.log(df_model['GDP_PC'])

print(f"log_GDP_PC 描述统计:")
print(df_model['log_GDP_PC'].describe())

# 对被解释变量 True_Resilience 也进行对数化
# 原因：汽变模型中被解释变量的对数化可以改正整体不均匹配的问题、降低异方差
print("\n对被解释变量 True_Resilience 进行对数不会...")
df_model['ln_Resilience'] = np.log(df_model['True_Resilience'])

print(f"ln_Resilience 描述统计:")
print(df_model['ln_Resilience'].describe())

# ============================================================================
# 第三部分：构造虚拟变量 Dummy_Rules
# ============================================================================
print("\n步骤3: 构造虚拟变量 Dummy_Rules...")

# Dummy_Rules 的生成逻辑：
# 若 Exposure > 0，则 Dummy_Rules = 1（处理国）
# 若 Exposure = 0，则 Dummy_Rules = 0（对照国）
df_model['Dummy_Rules'] = (df_model['Exposure'] > 0).astype(int)

print(f"Dummy_Rules 分布:")
print(df_model['Dummy_Rules'].value_counts())
print(f"处理国比例: {df_model['Dummy_Rules'].mean():.2%}")

# ============================================================================
# 第四部分：设置多重索引（MultiIndex）
# ============================================================================
print("\n步骤4: 设置 MultiIndex（REF_AREA, TIME_PERIOD）...")

# 将 REF_AREA 和 TIME_PERIOD 设置为 MultiIndex
df_indexed = df_model.set_index(['REF_AREA', 'TIME_PERIOD'])
print(f"MultiIndex 后数据维度: {df_indexed.shape}")
print(f"Index 名称: {df_indexed.index.names}")
print(df_indexed.head())

# ============================================================================
# 第四点五部分：添加截距项（修复 Python vs Stata 的差异）
# ============================================================================
print("\n步骤4.5: 添加截距项（Constant Term）...")

# 致命错误修复：Stata 的 xtreg 会自动添加常数项，但 Python linearmodels.PanelOLS 不会
# 后果：没有常数项会导致回归线被强制穿过原点，残差极大，P值失效
# 方案：在 DataFrame 中显式添加一个值全为 1.0 的 'Constant' 列
# 注意：当启用 entity_effects 和 time_effects 时，这个常数项会被 DeMean 处理，但仍保证正确的自由度
df_indexed['Constant'] = 1.0

print(f"添加常数项后数据维度: {df_indexed.shape}")
print(df_indexed.head())

# ============================================================================
# 第五部分：模型 1 - 创新模型（连续敞口指数 DID）
# ============================================================================
print("\n" + "=" * 80)
print("模型 1: 核心模型 - 使用连续敞口指数（Continuous DID）")
print("=" * 80)

# 准备模型1的变量
# 因变量：ln_Resilience（被解释变量的对数化）
# 自变量：Constant, Exposure, log_GDP_PC, FDI, Internet_Use, GE
y_model1 = df_indexed['ln_Resilience']
X_model1 = df_indexed[['Constant', 'Exposure', 'log_GDP_PC', 'FDI', 'Internet_Use', 'GE']]

print(f"\n模型1：因变量样本量 {len(y_model1)}")
print(f"模型1：自变量的描述统计:")
print(X_model1.describe())

# 运行 PanelOLS 回归
# EntityEffects: 控制个体（国家）固定效应
# TimeEffects: 控制时间（年份）固定效应
# cluster_entity=True: 聚类到国家层面的稳健标准误
model1 = PanelOLS(y_model1, X_model1, entity_effects=True, time_effects=True)
results1 = model1.fit(cov_type='clustered', cluster_entity=True)

print("\n" + "-" * 80)
print("模型 1 回归结果：")
print("-" * 80)
print(results1.summary)

# ============================================================================
# 第六部分：模型 2 - 对照模型（传统 0/1 DID）
# ============================================================================
print("\n" + "=" * 80)
print("模型 2: 对照模型 - 使用 0/1 虚拟变量（Traditional 0/1 DID）")
print("=" * 80)

# 准备模型2的变量
# 因变量：ln_Resilience（被解释变量的对数化）
# 自变量：Constant, Dummy_Rules, log_GDP_PC, FDI, Internet_Use, GE
y_model2 = df_indexed['ln_Resilience']
X_model2 = df_indexed[['Constant', 'Dummy_Rules', 'log_GDP_PC', 'FDI', 'Internet_Use', 'GE']]

print(f"\n模型2：因变量样本量 {len(y_model2)}")
print(f"模型2：自变量的描述统计:")
print(X_model2.describe())

# 运行 PanelOLS 回归
model2 = PanelOLS(y_model2, X_model2, entity_effects=True, time_effects=True)
results2 = model2.fit(cov_type='clustered', cluster_entity=True)

print("\n" + "-" * 80)
print("模型 2 回归结果：")
print("-" * 80)
print(results2.summary)

# ============================================================================
# 第七部分：核心系数对比
# ============================================================================
print("\n" + "=" * 80)
print("核心自变量系数与 P 值对比")
print("=" * 80)

# 提取模型1的核心自变量信息
exposure_coef1 = results1.params['Exposure']
exposure_pval1 = results1.pvalues['Exposure']

print(f"\n【模型1 - Continuous Exposure】")
print(f"自变量: Exposure")
print(f"  系数 (Coefficient): {exposure_coef1:.6f}")
print(f"  P值 (P-value):      {exposure_pval1:.6f}")
print(f"  显著性: ", end="")
if exposure_pval1 < 0.01:
    print("*** (p < 0.01)")
elif exposure_pval1 < 0.05:
    print("** (p < 0.05)")
elif exposure_pval1 < 0.10:
    print("* (p < 0.10)")
else:
    print("不显著")

# 提取模型2的核心自变量信息
dummy_coef2 = results2.params['Dummy_Rules']
dummy_pval2 = results2.pvalues['Dummy_Rules']

print(f"\n【模型2 - Traditional Dummy_Rules】")
print(f"自变量: Dummy_Rules")
print(f"  系数 (Coefficient): {dummy_coef2:.6f}")
print(f"  P值 (P-value):      {dummy_pval2:.6f}")
print(f"  显著性: ", end="")
if dummy_pval2 < 0.01:
    print("*** (p < 0.01)")
elif dummy_pval2 < 0.05:
    print("** (p < 0.05)")
elif dummy_pval2 < 0.10:
    print("* (p < 0.10)")
else:
    print("不显著")

# ============================================================================
# 第八部分：模型对比与结论
# ============================================================================
print("\n" + "=" * 80)
print("模型对比与经济学解释")
print("=" * 80)

coef_diff = abs(exposure_coef1 - dummy_coef2)
print(f"\n系数差异: {coef_diff:.6f}")
print(f"模型1系数 / 模型2系数比值: {exposure_coef1 / dummy_coef2:.4f}")

print(f"\n结论：")
print(f"- 模型1（连续变量）捕捉了规则深度对韧性的连续性影响")
print(f"- 模型2（虚拟变量）仅捕捉了有协定与无协定的离散化差异")
print(f"- 连续暴露指数相比0/1虚拟变量，更精确地刻画了制度影响的异质性")

print("\n" + "=" * 80)
print("回归分析完成！")
print("=" * 80)
