import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据
data = sio.loadmat('kos_doc_data.mat')
A = np.array(data['A'])
V = np.array(data['V'])

# A的列结构: [DocumentID, WordID, Count]
# 注意: WordID 在数据中通常是 1-based，需要转为 0-based 索引

# 2. 计算最大似然估计 (MLE)
# 统计每个单词在整个训练集中的总出现次数
num_words = np.max(A[:, 1])
word_counts = np.zeros(num_words)

for i in range(A.shape[0]):
    word_idx = A[i, 1] - 1  # 转换为 0-based 索引
    count = A[i, 2]
    word_counts[word_idx] += count

total_count = np.sum(word_counts)
theta_mle = word_counts / total_count  # 多项式分布参数 (Probability Mass)

# 3. 找出概率最大的 20 个单词
top_20_indices = np.argsort(theta_mle)[-20:]  # 获取最大的20个索引
top_20_probs = theta_mle[top_20_indices]
top_20_words = [V[i][0][0] for i in top_20_indices] # 获取对应的单词字符串

# 4. 绘制直方图
plt.figure(figsize=(10, 8))
plt.barh(range(20), top_20_probs, align='center')
plt.yticks(range(20), top_20_words)
plt.xlabel('Probability')
plt.title('Top 20 Words by MLE Probability')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('mle_histogram.png') # 保存图像
plt.show()