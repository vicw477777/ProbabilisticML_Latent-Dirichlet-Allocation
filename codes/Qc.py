import scipy.io as sio
import numpy as np

def solve_question_c_fixed():
    # 1. 加载数据
    data = sio.loadmat('kos_doc_data.mat')
    A = np.array(data['A'])
    B = np.array(data['B'])
    
    # 2. 设定参数
    alpha = 0.1
    # 词汇表大小 W：取 A 和 B 中最大的词 ID (确保涵盖所有词)
    W = max(np.max(A[:, 1]), np.max(B[:, 1]))
    
    # 3. 统计训练集词频 (使用向量化操作加速)
    # A[:, 1]-1 是词的索引 (0-based), A[:, 2] 是词频
    word_counts = np.zeros(W)
    np.add.at(word_counts, A[:, 1]-1, A[:, 2])
    
    N_total = np.sum(word_counts)

    # 4. 计算贝叶斯预测概率 (Bayesian Predictive Probabilities)
    # theta = (Nw + alpha) / (N + W * alpha)
    theta_bayes = (word_counts + alpha) / (N_total + W * alpha)

    # 5. 分析文档 2001
    doc_id = 2001
    # 提取文档 2001 的数据
    doc_mask = B[:, 0] == doc_id
    w_indices_2001 = B[doc_mask, 1] - 1
    # 【关键修改】转换为 float 以防止整数溢出
    counts_2001 = B[doc_mask, 2].astype(float) 
    
    # 计算 Log Probability: sum(count * log(theta))
    lp_2001 = np.sum(counts_2001 * np.log(theta_bayes[w_indices_2001]))
    n_2001 = np.sum(counts_2001)
            
    perplexity_2001 = np.exp(-lp_2001 / n_2001)

    # 6. 分析整个测试集 B
    # 提取所有测试集单词索引和计数
    all_w_indices = B[:, 1] - 1
    # 【关键修改】转换为 float 以防止整数溢出，这就是之前报错的原因
    all_counts = B[:, 2].astype(float) 
    
    # 一次性计算所有 log p
    lp_B = np.sum(all_counts * np.log(theta_bayes[all_w_indices]))
    n_B = np.sum(all_counts)
            
    perplexity_B = np.exp(-lp_B / n_B)

    # 7. 均匀分布 Perplexity = W
    perplexity_uniform = W

    # 输出结果
    print(f"Log Probability (Doc 2001): {lp_2001:.2f}")
    print(f"Perplexity (Doc 2001): {perplexity_2001:.2f}")
    print(f"Perplexity (All B): {perplexity_B:.2f}")
    print(f"Perplexity (Uniform): {perplexity_uniform}")

if __name__ == '__main__':
    solve_question_c_fixed()