import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# 辅助函数: 从非归一化概率中采样
def sampleDiscrete(p, ran=None):
    normalization_constant = np.sum(p)
    if normalization_constant == 0:
        return np.random.randint(len(p))
    uniform_number = ran or np.random.rand()
    r = uniform_number * normalization_constant
    a = p[0]
    i = 0
    while a < r and i < len(p) - 1:
        i += 1
        a += p[i]
    return i

def BMM_Question_D():
    # 1. 加载数据
    data = sio.loadmat('kos_doc_data.mat')
    A = np.array(data['A'])
    B = np.array(data['B'])
    
    # 参数设置
    K = 20
    alpha = 10
    gamma = 0.1
    num_iters = 50  # 题目要求画出随迭代变化的图，50次足够观察收敛
    
    W = max(np.max(A[:, 1]), np.max(B[:, 1]))
    D = np.max(A[:, 0])
    
    # --- 初始化 ---
    # 随机分配文档到簇
    sd = np.floor(K * np.random.rand(D)).astype(int)
    swk = np.zeros((W, K))
    sk_docs = np.zeros(K, dtype=int)
    
    # 填充初始计数
    print("Initializing BMM...")
    for d in range(D):
        doc_mask = A[:, 0] == (d + 1)
        w_ids = A[doc_mask, 1] - 1
        counts = A[doc_mask, 2]
        k = sd[d]
        swk[w_ids, k] += counts
        sk_docs[k] += 1
        
    sk_words = np.sum(swk, axis=0)
    
    # 用于记录混合比例的历史数据: [Iteration, Topic]
    proportions_history = np.zeros((num_iters, K))
    
    # --- Gibbs Sampling Loop ---
    print(f"Running {num_iters} Gibbs sweeps...")
    for it in range(num_iters):
        for d in range(D):
            # 获取文档 d 的数据
            doc_mask = A[:, 0] == (d + 1)
            w_ids = A[doc_mask, 1] - 1
            counts = A[doc_mask, 2]
            old_k = sd[d]
            
            # 1. 移除文档 d
            swk[w_ids, old_k] -= counts
            sk_docs[old_k] -= 1
            sk_words[old_k] -= np.sum(counts)
            
            # 2. 重新采样类别
            # 计算 log p(w|k)
            # 向量化计算: numer [W_in_doc, K], denom [K]
            numer = swk[w_ids, :] + gamma
            denom = sk_words + W * gamma
            # log_term: sum over words in doc [K]
            log_term = np.sum(counts[:, None] * (np.log(numer) - np.log(denom)), axis=0)
            
            # log p(z)
            log_prior = np.log(sk_docs + alpha)
            
            # Log posterior
            lb = log_prior + log_term
            b = np.exp(lb - np.max(lb)) # 减去 max 防止溢出
            
            new_k = sampleDiscrete(b)
            
            # 3. 添加文档 d 回去
            swk[w_ids, new_k] += counts
            sk_docs[new_k] += 1
            sk_words[new_k] += np.sum(counts)
            sd[d] = new_k
            
        # --- 【关键修改】记录本轮结束后的混合比例 ---
        # Posterior Expectation: (N_k + alpha) / (D + K*alpha)
        props = (sk_docs + alpha) / (D + K * alpha)
        proportions_history[it, :] = props
        
        if (it+1) % 10 == 0:
            print(f"Iteration {it+1}/{num_iters} completed.")

    # --- 绘图 ---
    plt.figure(figsize=(10, 6))
    for k in range(K):
        plt.plot(range(1, num_iters + 1), proportions_history[:, k])
    
    plt.xlabel('Gibbs Sweeps')
    plt.ylabel('Mixing Proportion')
    plt.title('Evolution of Mixing Proportions (BMM)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('qd_mixing_proportions.png')
    print("Plot saved: qd_mixing_proportions.png")

if __name__ == '__main__':
    BMM_Question_D()