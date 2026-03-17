import numpy as np
import matplotlib.pyplot as plt

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')

# 1. 模拟数据
years = np.arange(1980, 2011)
policy_year = 2000
idx_policy = np.where(years == policy_year)[0][0]

# 2. 构造“真实的处理组” (True Treated) - 复杂的非线性趋势
# 假设加州早在政策前就开始剧烈波动并下降
true_treated = 100 + 5 * np.sin((years - 1980)/2) - 0.5 * (years - 1980)**2 
# 归一化一下方便绘图
true_treated = (true_treated - true_treated.min()) / (true_treated.max() - true_treated.min()) * 100

# 3. 构造“失败的合成控制组” (Failed Synthetic Control)
# 假设你的控制组里只有一些走势平稳的州，怎么凑也凑不出加州那种剧烈波动的样子
# 所以合成出来的线比较平滑，无法捕捉特征
synthetic_control = 50 + 0.5 * (years - 1980) + np.random.normal(0, 2, len(years))

# 4. 绘图
plt.figure(figsize=(12, 6))

# 画出两条线
plt.plot(years, true_treated, 'r-', linewidth=3, label='Actual Treated Unit (California)')
plt.plot(years, synthetic_control, 'k--', linewidth=2, alpha=0.7, label='Synthetic Control (Poor Fit)')

# 标注政策时间
plt.axvline(x=policy_year, color='gray', linestyle=':', label='Policy Intervention')

# ==========================================
# 5. 标注错误区域 (这是给审稿人看的死穴)
# ==========================================
# 高亮显示事前巨大的拟合误差
plt.fill_between(years[:idx_policy], true_treated[:idx_policy], synthetic_control[:idx_policy], 
                 color='red', alpha=0.2, label='Pre-intervention Prediction Error (MSPE)')

plt.text(1990, 70, 'FATAL ERROR:\nPoor Pre-trend Fit', color='darkred', fontsize=14, fontweight='bold', ha='center')

plt.annotate('Gap exists BEFORE policy!', xy=(1995, 40), xytext=(1985, 20),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

plt.title('FAILED SCM: Lack of Pre-intervention Fit\n(If they were different before, they are not comparable after)', 
          fontsize=16, fontweight='bold', color='darkred')
plt.ylabel('Outcome Variable', fontsize=12)
plt.legend(loc='upper right')
plt.show()