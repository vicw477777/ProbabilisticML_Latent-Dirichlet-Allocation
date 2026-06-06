import scipy.io as sio
import numpy as np
from scipy.sparse import coo_matrix as sparse
import matplotlib.pyplot as plt

# 简单的 sampleDiscrete (保持不变或从 helper 导入)
def sampleDiscrete(p, ran=None):
    normalization_constant = np.sum(p)
    uniform_number = ran or np.random.rand()
    r = uniform_number * normalization_constant
    a = p[0]
    i = 0
    while a < r and i < len(p) - 1:
        i += 1
        a += p[i]
    return i

def LDA_with_analysis(A, B, K, alpha, gamma, num_iters=50):
    W = max(np.max(A[:, 1]), np.max(B[:, 1]))
    D = np.max(A[:, 0])

    # 稀疏矩阵转换 (0-based)
    swd = sparse((A[:, 2], (A[:, 1]-1, A[:, 0]-1))).tocsr()
    Swd = sparse((B[:, 2], (B[:, 1]-1, B[:, 0]-1))).tocsr()

    # 初始化
    skd = np.zeros((K, D)) 
    swk = np.zeros((W, K)) 
    s = [] 
    
    # 随机初始化
    print("Initializing LDA...")
    for d in range(D):
        z = np.zeros((W, K))
        words_in_doc = A[A[:, 0]==d+1, 1] - 1
        for w in words_in_doc:
            c = swd[w, d]
            for _ in range(c):
                k = int(np.floor(K*np.random.rand()))
                z[w, k] += 1
        skd[:, d] = np.sum(z, axis=0)
        swk += z
        s.append(sparse(z))

    sk = np.sum(skd, axis=1) # Topic counts (Global)
    
    # 记录历史数据
    topic_posteriors = np.zeros((num_iters, K))
    topic_entropies = np.zeros((num_iters, K))

    print(f"Running {num_iters} Gibbs sweeps...")
    for iter in range(num_iters):
        # --- Gibbs Sampling Loop ---
        for d in range(D):
            z = s[d].todense()
            words_in_doc = A[A[:, 0]==d+1, 1] - 1
            for w in words_in_doc:
                a = z[w, :].copy()
                indices = np.where(a > 0)[1]
                np.random.shuffle(indices)
                for k in indices:
                    k = int(k)
                    for _ in range(int(a[0, k])):
                        z[w, k] -= 1
                        swk[w, k] -= 1
                        sk[k] -= 1
                        skd[k, d] -= 1
                        
                        # Conditional probability
                        b_val = (alpha + skd[:, d]) * (gamma + swk[w, :]) / (W * gamma + sk)
                        kk = sampleDiscrete(b_val)
                        
                        z[w, kk] += 1
                        swk[w, kk] += 1
                        sk[kk] += 1
                        skd[kk, d] += 1
            s[d] = sparse(z)

        # --- Analysis per iteration ---
        # 1. Topic Posteriors (Proportion of words assigned to topic k)
        topic_posteriors[iter, :] = (sk + alpha) / np.sum(sk + alpha)
        
        # 2. Topic Entropy
        # P(w|k) = (N_wk + gamma) / (N_k + W*gamma)
        # H(k) = - sum(P * log2(P))
        for k in range(K):
            pw_k = (swk[:, k] + gamma) / (sk[k] + W * gamma)
            # Avoid log(0)
            topic_entropies[iter, k] = -np.sum(pw_k * np.log2(pw_k + 1e-12))
            
        if (iter+1) % 10 == 0:
            print(f"Iteration {iter+1}/{num_iters} completed.")

    # --- Perplexity on Test Set B ---
    print("Calculating Perplexity on B...")
    lp, nd = 0, 0
    unique_docs_in_b = np.unique(B[:, 0])
    for d in unique_docs_in_b:
        # Fold-in: random init for doc d
        z = np.zeros((W, K))
        words_in_d = B[B[:, 0]==d, 1] - 1
        for w in words_in_d:
            c = Swd[w, d-1]
            for _ in range(c):
                k = int(np.floor(K * np.random.rand()))
                z[w, k] += 1
        Skd_test = np.sum(z, axis=0)
        
        # Short Gibbs for test doc
        for _ in range(10): # usually fewer iters for fold-in
            for w in words_in_d:
                a = z[w, :].copy()
                indices = np.where(a > 0)[0] # Note: dense array index difference
                for k in indices:
                    k = int(k)
                    for _ in range(int(a[k])):
                        z[w, k] -= 1
                        Skd_test[k] -= 1
                        # Predictive probability using trained swk/sk
                        b_val = (alpha + Skd_test) * (gamma + swk[w, :]) / (W * gamma + sk)
                        kk = sampleDiscrete(b_val)
                        z[w, kk] += 1
                        Skd_test[kk] += 1
        
        # Calculate Log Prob
        # Theta_d * Phi_k
        theta_d = (alpha + Skd_test) / np.sum(alpha + Skd_test) # [K, 1]
        phi_k = (gamma + swk) / (W * gamma + sk) # [W, K]
        p_w = np.matmul(phi_k, theta_d) # [W, ]
        
        doc_words = B[B[:, 0]==d, 1:] 
        # doc_words: [word_id, count]
        
        # p_w index is 0-based
        valid_indices = doc_words[:, 0] - 1
        valid_probs = p_w[valid_indices]
        lp += np.sum(doc_words[:, 1] * np.log(valid_probs))
        nd += np.sum(doc_words[:, 1])

    perplexity = np.exp(-lp/nd)
    
    return perplexity, topic_posteriors, topic_entropies

# --- Execution & Plotting ---
if __name__ == '__main__':
    data = sio.loadmat('kos_doc_data.mat')
    A = np.array(data['A'])
    B = np.array(data['B'])
    
    perplexity, post_hist, ent_hist = LDA_with_analysis(A, B, K=20, alpha=0.1, gamma=0.1, num_iters=50)
    
    print(f"Final LDA Perplexity: {perplexity:.2f}")

    # Plot 1: Topic Posteriors
    plt.figure(figsize=(10, 5))
    plt.plot(post_hist)
    plt.xlabel('Gibbs Iterations')
    plt.ylabel('Topic Probability')
    plt.title('Evolution of Topic Posteriors (K=20)')
    plt.savefig('qe_topic_posteriors.png')
    
    # Plot 2: Topic Entropies
    plt.figure(figsize=(10, 5))
    plt.plot(ent_hist)
    plt.xlabel('Gibbs Iterations')
    plt.ylabel('Entropy (bits)')
    plt.title('Evolution of Word Entropy per Topic')
    plt.savefig('qe_topic_entropies.png')