import matplotlib.pyplot as plt
import numpy as np

# 提取的数据
num_demos = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
avg_ret = np.array([1099, 1199, 869.3, 1221, 1349, 1727, 1308, 1505, 1501, 1333])
std_ret = np.array([12.6, 26.82, 34.59, 21.77, 149.1, 381.1, 54.33, 302.5, 439.6, 87.06])
expert_30_percent = 3718 * 0.3

plt.figure(figsize=(10, 6))

# 绘制均值折线
plt.plot(num_demos, avg_ret, label='BC Agent (Mean)', color='tab:blue', linewidth=2, marker='o')

# 绘制标准差阴影区域 (Shaded Area)
plt.fill_between(num_demos, avg_ret - std_ret, avg_ret + std_ret, 
                 alpha=0.2, color='tab:blue', label='Standard Deviation')

# 绘制 30% 专家表现基准线
plt.axhline(y=expert_30_percent, color='tab:red', linestyle='--', linewidth=2, label='30% Expert Baseline')

# 图表装饰
plt.xlabel('Training Batch Size', fontsize=12)
plt.ylabel('Average Evaluation Return', fontsize=12)
plt.xticks(num_demos)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best')

# 保存并显示
plt.savefig('bc_trend.png', dpi=300)
plt.show()