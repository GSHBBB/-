import pandas as pd
import numpy as np

# ============================================================================
# 描述性统计脚本：基于最终确定的变量组合
# ============================================================================

# 设置工作目录（根据实际路径调整）
base_dir = r"C:\Users\29106\OneDrive\文档\毕业论文"

# 1. 读取主面板数据
print("读取主面板数据...")
df = pd.read_csv(f"{base_dir}/Master_Panel.csv")

# 2. 读取并合并节点中心性数据
print("读取并合并节点中心性数据...")
centrality_df = pd.read_csv(f"{base_dir}/date_newly_established/node_centrality_panel.csv")
df = df.merge(centrality_df, on=['REF_AREA', 'TIME_PERIOD'], how='left')

# 3. 计算新变量
print("计算新变量...")
df['ln_Resilience_Inv'] = -np.log(df['True_Resilience'])  # 网络韧性对数（负对数）
df['log_GDP_PC'] = np.log(df['GDP_PC'])  # 人均GDP对数

# 4. 生成滞后项
print("生成滞后项...")
df = df.sort_values(['REF_AREA', 'TIME_PERIOD'])
df['L1_Exposure'] = df.groupby('REF_AREA')['Exposure'].shift(1)  # 敞口指数滞后1期

# 5. 选择最终变量组合（剔除Internet_Use）
variables = [
    'ln_Resilience_Inv',    # 被解释变量：网络韧性对数
    'Exposure',             # 核心自变量：敞口指数
    'L1_Exposure',          # 敞口指数滞后项
    'log_GDP_PC',           # 控制变量：人均GDP对数
    'FDI',                  # 控制变量：FDI
    'RQ',                   # 机制变量：监管质量
    'GE',                   # 机制变量：政府效能
    'Out_Degree_Centrality' # 异质性变量：出度中心性
]

# 6. 数据清理：删除缺失值
print("数据清理...")
df_clean = df[variables].dropna()
print(f"最终样本量：{len(df_clean)} 观测值")

# 7. 执行描述性统计
print("\n执行描述性统计...")
desc = df_clean.describe()

# 8. 提取关键统计量
result = desc.loc[['count', 'mean', 'std', 'min', 'max']]

# 9. 格式化输出
print("\n" + "="*80)
print("描述性统计结果")
print("="*80)

# 定义变量中文名称映射
var_names = {
    'ln_Resilience_Inv': '网络韧性对数',
    'Exposure': '敞口指数',
    'L1_Exposure': '敞口指数滞后项',
    'log_GDP_PC': '人均GDP对数',
    'FDI': 'FDI',
    'RQ': '监管质量',
    'GE': '政府效能',
    'Out_Degree_Centrality': '出度中心性'
}

# 打印表格
print(f"{'变量':<12} {'观测值':<8} {'均值':<10} {'标准差':<8} {'最小值':<10} {'最大值':<10}")
print("-" * 70)

for var in variables:
    if var in result.columns:
        count = int(result.loc['count', var])
        mean = f"{result.loc['mean', var]:.3f}"
        std = f"{result.loc['std', var]:.3f}"
        min_val = f"{result.loc['min', var]:.3f}"
        max_val = f"{result.loc['max', var]:.3f}"
        print(f"{var_names[var]:<12} {count:<8} {mean:<10} {std:<8} {min_val:<10} {max_val:<10}")

print("\n" + "="*80)

# 可选：保存结果到CSV文件
output_file = f"{base_dir}/descriptive_stats.csv"
result.to_csv(output_file, encoding='utf-8-sig')
print(f"描述性统计结果已保存到：{output_file}")

print("\n脚本执行完成！")