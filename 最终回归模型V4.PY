import pandas as pd
import numpy as np
import os
import warnings
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：数据读取与预处理
# ============================================================================
print("=" * 90)
print("双向固定效应面板回归：样本释放与交乘项检验（V4 终极版）")
print("=" * 90)

base_dir = r"C:\Users\29106\OneDrive\文档\毕业论文"
data_file = os.path.join(base_dir, "Master_Panel.csv")
resilience_file = os.path.join(base_dir, "DATE_from_WB", "final_resilience_panel.csv")

df = pd.read_csv(data_file)
df_resilience = pd.read_csv(resilience_file)
if 'Out_Degree_Centrality' in df_resilience.columns:
    centrality_df = df_resilience[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']].copy()
    df = df.merge(centrality_df, on=['REF_AREA', 'TIME_PERIOD'], how='left')

# 【关键修改 1】：坚决剔除 Internet_Use，释放被误删的边缘穷国样本！
required_cols = ['REF_AREA', 'TIME_PERIOD', 'True_Resilience', 'Exposure', 'GDP_PC', 'FDI', 'Out_Degree_Centrality']
df_model = df[required_cols].dropna().copy()

# 因变量与对数化
df_model['ln_Resilience_Inv'] = -np.log(df_model['True_Resilience'])
df_model['log_GDP_PC'] = np.log(df_model['GDP_PC'])

df_indexed = df_model.set_index(['REF_AREA', 'TIME_PERIOD'])

# 生成滞后1期
df_indexed['L1_Exposure'] = df_indexed.groupby(level='REF_AREA')['Exposure'].shift(1)
df_indexed['L1_Centrality'] = df_indexed.groupby(level='REF_AREA')['Out_Degree_Centrality'].shift(1)

df_indexed_clean = df_indexed.dropna(subset=['L1_Exposure', 'L1_Centrality', 'ln_Resilience_Inv'])
df_indexed_clean['Constant'] = 1.0

# 【关键修改 2】：构造核心交乘项 (Exposure * Centrality)
df_indexed_clean['Interaction'] = df_indexed_clean['L1_Exposure'] * df_indexed_clean['L1_Centrality']

# ============================================================================
# 第二部分：全样本交乘项回归（用最严谨的方式证明异质性）
# ============================================================================
print("\n" + "-" * 80)
print(">>> 模型 1: 全样本交互项模型 (验证边际效用递减)")
y_inter = df_indexed_clean['ln_Resilience_Inv']
X_inter = df_indexed_clean[['Constant', 'L1_Exposure', 'Interaction', 'log_GDP_PC', 'FDI']]

model_inter = PanelOLS(y_inter, X_inter, entity_effects=True, time_effects=True)
results_inter = model_inter.fit(cov_type='clustered', cluster_entity=True)
print(results_inter.summary)

print("\n【交乘项经济学含义解析】")
print("如果 L1_Exposure 系数为正(***)，Interaction 系数为负(***)：")
print("这证明：在网络最边缘(Centrality接近0)，规则对韧性有极强的正向拉动（雪中送炭）；")
print("随着网络中心度升高，正向作用被 Interaction 的负值抵消，直至在核心国家转为负效应（竞争内耗）！")

# ============================================================================
# 第三部分：重新运行 33% 宽口径分组回归 (验证机制)
# ============================================================================
print("\n" + "=" * 90)
print("【放宽口径的三分位分组检验】")
q33 = df_indexed_clean['L1_Centrality'].quantile(0.33)
q66 = df_indexed_clean['L1_Centrality'].quantile(0.66)

df_peri = df_indexed_clean[df_indexed_clean['L1_Centrality'] <= q33]
df_core = df_indexed_clean[df_indexed_clean['L1_Centrality'] >= q66]

print(f"释放样本后，边缘组(Bottom 33%)样本量: {len(df_peri)}")

print("\n>>> 模型 2: 真正的边缘组回归 (剔除缺失值干扰后)")
y_p = df_peri['ln_Resilience_Inv']
X_p = df_peri[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_p = PanelOLS(y_p, X_p, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(f"L1_Exposure 系数: {res_p.params['L1_Exposure']:.6f} | P值: {res_p.pvalues['L1_Exposure']:.6f}")

print("\n>>> 模型 3: 真正的核心组回归")
y_c = df_core['ln_Resilience_Inv']
X_c = df_core[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_c = PanelOLS(y_c, X_c, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(f"L1_Exposure 系数: {res_c.params['L1_Exposure']:.6f} | P值: {res_c.pvalues['L1_Exposure']:.6f}")


# ============================================================================
# 第四部分：将所有回归结果导出为 Excel
# ============================================================================
def results_to_dataframe(res):
    try:
        params = res.params
    except Exception:
        return None, {}

    rows = []
    for param in params.index:
        coef = params[param]

        se = np.nan
        for attr in ('std_errors', 'std_err', 'bse'):
            try:
                se = getattr(res, attr)[param]
                break
            except Exception:
                se = np.nan

        tstat = np.nan
        for attr in ('tstats', 'tvalues'):
            try:
                tstat = getattr(res, attr)[param]
                break
            except Exception:
                tstat = np.nan

        pval = np.nan
        for attr in ('pvalues', 'pvals'):
            try:
                pval = getattr(res, attr)[param]
                break
            except Exception:
                pval = np.nan

        try:
            ci = res.conf_int()
            ci_lower = ci.loc[param][0]
            ci_upper = ci.loc[param][1]
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan

        rows.append({
            'coef': coef,
            'std_err': se,
            't': tstat,
            'pval': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        })

    df_coef = pd.DataFrame(rows, index=params.index)
    meta = {
        'nobs': getattr(res, 'nobs', np.nan),
        'rsquared': getattr(res, 'rsquared', np.nan),
        'params_count': len(params),
    }
    return df_coef, meta


def export_all_results_to_excel(path):
    writer = pd.ExcelWriter(path, engine='openpyxl')
    try:
        models = [
            ("全样本交互项", results_inter),
            ("边缘组", res_p),
            ("核心组", res_c),
        ]

        for name, res in models:
            df_coef, meta = results_to_dataframe(res)
            if df_coef is None:
                continue
            sheet_name = name[:31]
            df_coef.to_excel(writer, sheet_name=sheet_name)
            meta_df = pd.DataFrame.from_dict(meta, orient='index', columns=['value'])
            meta_df.to_excel(writer, sheet_name=sheet_name + '_meta')
        # 对于 openpyxl 的 ExcelWriter，不需要显式调用 save()
    finally:
        writer.close()


output_path = os.path.join(base_dir, 'regression_results.xlsx')
export_all_results_to_excel(output_path)
print(f"\n已将回归结果导出到: {output_path}")