"""
Network Resilience Calculator - 重构版本
=========================================
对网络中心度面板数据进行HP滤波，计算网络韧性的被解释变量(Inexpress)

输入文件: node_centrality_panel.csv
输出文件: final_resilience_panel.csv

核心处理流程：
  1. 加载node_centrality_panel.csv
  2. 按国家分组，进行HP滤波(lambda=6.25，年度数据标准参数)
  3. 计算波动率：volatility = |cycle|
  4. 关键保护机制：safe_trend = np.clip(trend, 0.01, None)
     防止极小值或负值导致比值爆炸
  5. 计算Express = volatility / safe_trend
  6. 最终被解释变量：Inexpress = log(Express + 1e-6)
  7. 过滤地区数据年份 < 4年的条目
  8. 输出最终面板数据并进行极值验证
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
import os
import sys
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
# 抑制不必要的警告
warnings.filterwarnings('ignore')

# 中文字体设置函数（健壮版本）
def setup_chinese_font():
    """设置可用的中文字体，自动检测系统环境"""
    # 尝试的中文字体列表（优先级从高到低）
    chinese_fonts = [
        'SimHei',           # Windows 内置
        'Microsoft YaHei',  # Windows 内置
        'STSong',          # Mac 内置（宋体）
        'Heiti SC',        # Mac 内置（黑体）
        'SimSun',          # Windows 备选（仿宋）
        'WenQuanYi Zen Hei',  # Linux 开源
    ]
    
    # 获取系统中已安装的字体名称
    try:
        available_font_names = {font.name for font in fm.fontManager.ttflist}
    except:
        # 如果上述方法失败，使用备选方法
        try:
            available_font_names = set(fm.findSystemFonts(fontpaths=None))
        except:
            available_font_names = set()
    
    # 从列表中选择第一个可用的字体
    selected_font = None
    for font in chinese_fonts:
        try:
            fm.findfont(font)
            selected_font = font
            break
        except:
            continue
    
    # 如果都不可用，使用通用备选
    if selected_font is None:
        selected_font = 'DejaVu Sans'
    
    plt.rcParams['font.sans-serif'] = [selected_font, 'Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    
    return selected_font

# 执行字体设置
try:
    _ = setup_chinese_font()
except Exception as e:
    # 如果字体设置失败，使用最小化配置
    plt.rcParams['axes.unicode_minus'] = False
    pass


# ====== 配置 ======
INPUT_FILE = r"date_newly_established\node_centrality_panel.csv"
OUTPUT_FILE = r"date_newly_established\final_resilience_panel.csv"
LAMBDA = 6.25  # HP滤波平滑参数（年度数据的标准值）
MIN_YEARS = 5   # 最少年份要求（用于有效滤波）
TREND_MIN = 0.001  # 分母保护下界


def load_and_validate_data(filepath):
    """
    加载并验证输入数据。
    
    参数:
        filepath (str): 输入CSV文件路径
    
    返回:
        pd.DataFrame: 经过验证的数据框
    """
    print(f"\n========== 【步骤1】加载输入数据 ==========")
    print(f"✓ 读取文件: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ 错误：文件不存在 {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误：读取文件失败 - {e}")
        sys.exit(1)
    
    print(f"  原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"  列名: {list(df.columns)}")
    
    # 验证必要列存在
    required_cols = ['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 错误：缺少必要列 {missing_cols}")
        sys.exit(1)
    
    # 删除缺失值
    df = df.dropna(subset=['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality'])
    print(f"  删除缺失值后: {df.shape[0]} 行")
    
    # 排序
    df = df.sort_values(['REF_AREA', 'TIME_PERIOD']).reset_index(drop=True)
    print(f"  ├─ 时间范围: {df['TIME_PERIOD'].min()} - {df['TIME_PERIOD'].max()}")
    print(f"  └─ 国家总数: {df['REF_AREA'].nunique()}")
    
    return df


def apply_hp_filter_to_country(sub_df, country, lamb=6.25):
    """
    对单个国家的时间序列应用HP滤波。
    
    参数:
        sub_df (pd.DataFrame): 该国家的子数据框
        country (str): 国家代码
        lamb (float): HP滤波参数
    
    返回:
        pd.DataFrame: 包含滤波结果的数据框，若失败返回None
    """
    n_years = len(sub_df)
    
    if n_years < MIN_YEARS:
        return None
    
    try:
        # 获取时间序列
        series = sub_df['Out_Degree_Centrality'].values
        
        # 执行HP滤波（注意：正式API返回顺序是(cycle, trend)）
        cycle, trend = hpfilter(series, lamb=lamb)
        
        # 计算波动率（绝对波动项）
        volatility = np.abs(cycle)
        
        # 【关键】分母保护：使用clip防止极小值导致的数值爆炸
        safe_trend = np.clip(trend, a_min=TREND_MIN, a_max=None)
        
        # 计算比值（网络表达能力Express）
        express = volatility / safe_trend
        
        # 取对数得到不导米韧性指标Inexpress
        inexpress = np.log(express + 1e-6)
        
        # 组装结果数据框
        result = sub_df[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']].copy()
        result['Trend'] = trend
        result['Volatility'] = volatility
        result['Safe_Trend'] = safe_trend
        result['Express'] = express
        result['Inexpress'] = inexpress
        
        return result
        
    except Exception as e:
        # 单个国家失败不影响整体，仅记录警告
        print(f"  ⚠ 国家 {country} 滤波失败: {e}")
        return None


def main():
    """主执行流程。"""
    
    print("\n" + "="*75)
    print("" * 20 + "【网络韧性HP滤波计算】")
    print("生成双重差分回归的被解释变量 Inexpress")
    print("="*75)
    
    # 【步骤1】加载并验证数据
    df = load_and_validate_data(INPUT_FILE)
    
    # 【步骤2】按国家分组，执行HP滤波
    print(f"\n========== 【步骤2】按国家进行HP滤波处理 ==========")
    print(f"HP参数 lambda={LAMBDA}, 最低年份数={MIN_YEARS}, 分母保护下界={TREND_MIN}")
    
    all_results = []
    grouped = df.groupby('REF_AREA')
    total_countries = len(grouped)
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for i, (country, sub_df) in enumerate(grouped, 1):
        n_years = len(sub_df)
        
        # 检查年份数是否充足
        if n_years < MIN_YEARS:
            print(f"  [{i}/{total_countries}] {country}: ⊘ 跳过 ({n_years} 年 < {MIN_YEARS} 年)")
            skip_count += 1
            continue
        
        # 执行HP滤波
        result = apply_hp_filter_to_country(sub_df, country, lamb=LAMBDA)
        
        if result is not None:
            all_results.append(result)
            success_count += 1
            print(f"  [{i}/{total_countries}] {country}: ✓ 成功 ({n_years} 年)")
        else:
            fail_count += 1
    
    print(f"\n滤波统计：成功={success_count}, 跳过={skip_count}, 失败={fail_count}")
    
    if not all_results:
        print("\n❌ 错误：未成功处理任何国家，无法继续")
        sys.exit(1)
    
    # 【步骤3】组装最终面板数据
    print(f"\n========== 【步骤3】组装最终面板数据 ==========")
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 清理任何仍可能存在的缺失值或无穷值
    final_df = final_df.dropna(subset=['Inexpress'])
    final_df = final_df[~np.isinf(final_df['Inexpress'])]
    
    # 计算真实网络韧性（取负Inexpress）
    final_df['True_Resilience'] = -final_df['Inexpress']
    
    print(f"✓ 最终数据维度: {final_df.shape[0]} 行, {final_df.shape[1]} 列")
    print(f"  ├─ 覆盖国家数: {final_df['REF_AREA'].nunique()}")
    print(f"  └─ 覆盖年份数: {final_df['TIME_PERIOD'].nunique()}")
    
    # 【步骤4】数据导出
    print(f"\n========== 【步骤4】导出最终数据 ==========")
    
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建输出目录: {output_dir}")
    
    # 选择要导出的列（仅保留必要的列）
    export_cols = ['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality', 
                   'Trend', 'Volatility', 'Express', 'Inexpress', 'True_Resilience']
    final_df_export = final_df[export_cols].copy()
    
    try:
        final_df_export.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"✓ 已保存至: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ 错误：保存文件失败 - {e}")
        sys.exit(1)
    
    # 【步骤5】极值验证与统计输出
    print(f"\n========== 【步骤5】极值验证与统计摘要 ==========")
    
    print(f"\n被解释变量 Inexpress 的统计描述:")
    print(final_df_export['Inexpress'].describe())
    
    print(f"\n关键变量统计:")
    stats_cols = ['Out_Degree_Centrality', 'Trend', 'Volatility', 'Express', 'Inexpress']
    print(final_df_export[stats_cols].describe().round(6))
    
    print(f"\n极值检查 (Inexpress):")
    print(f"  ├─ 最小值: {final_df_export['Inexpress'].min():.6f}")
    print(f"  ├─ 最大值: {final_df_export['Inexpress'].max():.6f}")
    print(f"  ├─ 平均值: {final_df_export['Inexpress'].mean():.6f}")
    print(f"  ├─ 标准差: {final_df_export['Inexpress'].std():.6f}")
    print(f"  └─ 异常值数: {(final_df_export['Inexpress'].abs() > 5).sum()} (|值| > 5)")
    
    # 检查是否存在极端离群值（可能表示分母保护机制未生效）
    extreme_outliers = (final_df_export['Express'] > 1000).sum()
    if extreme_outliers > 0:
        print(f"\n⚠ 警告：存在 {extreme_outliers} 个极端离群值 (Express > 1000)")
    else:
        print(f"\n✓ 表达能力指标 Express 已被有效控制，无极端离群值")
    
    print(f"\n数据预览 (前 15 行):")
    print(final_df_export.head(15).to_string(index=False))
    
    # 【步骤6】可视化：前十名国家的韧性变化趋势
    print(f"\n========== 【步骤6】可视化 - 前十名国家的韧性变化 ==========")
    plot_top_n_resilience_trends(final_df_export, n=10)
    
    print("\n" + "="*75)
    print("✓ HP滤波与韧性计算完成！")
    print(f"✓ 最终数据已保存至: {OUTPUT_FILE}")
    print("="*75 + "\n")


def plot_top_n_resilience_trends(final_df, n=10):
    """
    绘制按最后一年真实韧性从大到小排名的前N个国家的韧性变化折线图。
    
    使用seaborn风格，为10个国家分配易于区分的颜色和不同的标记。
    
    参数:
        final_df (pd.DataFrame): 包含 TIME_PERIOD, REF_AREA, True_Resilience 的最终面板数据
        n (int): 要绘制的国家数量，默认为10
    """
    
    # 强制设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 设置seaborn主题
    sns.set_theme(style="whitegrid")
    
    # 获取最后一年的数据，按韧性从大到小排序
    last_year = final_df['TIME_PERIOD'].max()
    last_year_data = final_df[final_df['TIME_PERIOD'] == last_year].sort_values(
        'True_Resilience', ascending=False
    )
    
    # 选取前N个国家，保持排名顺序（从大到小）
    top_n_countries = last_year_data['REF_AREA'].head(n).tolist()
    
    print(f"\n✓ 按 {last_year} 年韧性指标排名（从大到小），前 {n} 个国家：")
    print("="*65)
    ranking_data = []
    for idx, (i, row) in enumerate(last_year_data.head(n).iterrows(), 1):
        print(f"  第{idx:2d}名：{row['REF_AREA']:5s}   真实韧性值 = {row['True_Resilience']:10.6f}")
        ranking_data.append((idx, row['REF_AREA'], row['True_Resilience']))
    print("="*65)
    
    # 筛选这些国家的全时间序列数据
    plot_data = final_df[final_df['REF_AREA'].isin(top_n_countries)].copy()
    plot_data = plot_data.sort_values(['REF_AREA', 'TIME_PERIOD'])
    
    # 定义颜色和标记
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers_list = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # 绘制折线图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for rank_idx, country in enumerate(top_n_countries):
        country_data = plot_data[plot_data['REF_AREA'] == country]
        
        # 图例中显示排名和国家代码
        label = f"#{rank_idx+1} {country}"
        
        ax.plot(
            country_data['TIME_PERIOD'],
            country_data['True_Resilience'],
            marker=markers_list[rank_idx % len(markers_list)],
            color=colors_list[rank_idx % len(colors_list)],
            label=label,
            linewidth=2,
            markersize=6,
            alpha=0.8
        )
    
    # 设置X轴刻度为整数年份
    years = sorted(plot_data['TIME_PERIOD'].unique())
    ax.set_xticks(years)
    ax.set_xticklabels([int(y) for y in years], rotation=45, ha='right')
    
    # 设置标签和标题
    ax.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax.set_ylabel('网络韧性 (-Inexpress)', fontsize=12, fontweight='bold')
    ax.set_title('数字服务出口网络中心度排名前十国家的网络韧性趋势 (2005-2024)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 将图例放置在图表外侧（右侧）
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10, 
              frameon=True, fancybox=True, shadow=True, ncol=1)
    
    # 网格设置
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 保存图示
    output_dir = os.path.dirname(OUTPUT_FILE)
    plot_file = os.path.join(output_dir, 'top_countries_resilience_trends.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 折线图已保存至: {plot_file}")
    
    plt.show()


if __name__ == "__main__":
    main()
