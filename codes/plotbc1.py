import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.special import gammaln


# ==========================
# 数据加载 & 公共函数
# ==========================

def load_kos_data(mat_path="kos_doc_data.mat"):
    data = sio.loadmat(mat_path)
    A = np.asarray(data["A"])
    B = np.asarray(data["B"])
    V = np.asarray(data["V"])
    W = V.shape[0]  # 词表大小
    return A, B, W


def compute_word_counts(A, W):
    """
    A: (n,3) [doc_id, word_id, count], word_id 是 1-based
    """
    word_ids = A[:, 1].astype(int) - 1
    counts = A[:, 2].astype(float)
    word_counts = np.bincount(word_ids, weights=counts, minlength=W)
    return word_counts


# ==========================
# Question B: 画概率 vs rank
# ==========================

def plot_question_b(word_counts, W, output_prefix="qb_alpha"):
    N_total = float(word_counts.sum())
    theta_mle = word_counts / N_total

    sort_indices = np.argsort(theta_mle)[::-1]
    ranks = np.arange(1, W + 1)

    alphas_to_plot = [0.001, 0.1, 1.0, 10.0]

    # ----- 全局图 -----
    plt.figure(figsize=(10, 6))

    for a in alphas_to_plot:
        theta_bayes = (word_counts + a) / (N_total + W * a)
        plt.plot(
            ranks,
            theta_bayes[sort_indices],
            label=f"Bayes (alpha={a})",
            linewidth=1.5,
        )

    valid_mle_mask = theta_mle[sort_indices] > 0
    plt.plot(
        ranks[valid_mle_mask],
        theta_mle[sort_indices][valid_mle_mask],
        "k--",
        label="MLE (zero for unseen)",
        alpha=0.7,
        linewidth=1.5,
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Word Rank (log scale)")
    plt.ylabel("Probability (log scale)")
    plt.title("Effect of alpha on word probabilities (full range)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_full.png", dpi=200)
    print(f"-> 图表已保存: {output_prefix}_full.png")

    # ----- 长尾 zoom 图 -----
    plt.figure(figsize=(10, 6))

    for a in alphas_to_plot:
        theta_bayes = (word_counts + a) / (N_total + W * a)
        plt.plot(
            ranks,
            theta_bayes[sort_indices],
            label=f"Bayes (alpha={a})",
            linewidth=1.5,
        )

    plt.plot(
        ranks[valid_mle_mask],
        theta_mle[sort_indices][valid_mle_mask],
        "k--",
        label="MLE (zero for unseen)",
        alpha=0.7,
        linewidth=1.5,
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(100, W)
    plt.ylim(1e-6, 1e-2)
    plt.xlabel("Word Rank (log scale)")
    plt.ylabel("Probability (log scale)")
    plt.title("Effect of alpha on rare words (zoomed tail)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_tail.png", dpi=200)
    print(f"-> 图表已保存: {output_prefix}_tail.png")


# ==========================
# Question C: 表格 + 2 条曲线
# ==========================

def stats_for_alpha(alpha, word_counts, B, W):
    """
    返回：
    categorical LP (doc2001),
    multinomial LP (doc2001),
    perplexity(doc2001),
    perplexity(all B)
    """
    N_total = float(word_counts.sum())
    theta = (word_counts + alpha) / (N_total + W * alpha)

    # ---- Doc 2001 ----
    mask2001 = (B[:, 0] == 2001)
    d = B[mask2001]
    word_ids_d = d[:, 1].astype(int) - 1
    counts_d = d[:, 2].astype(float)
    N_doc = counts_d.sum()

    # categorical log p
    log_cat_2001 = float(np.sum(counts_d * np.log(theta[word_ids_d])))

    # multinomial log p = categorical + log(N! / ∏ c_w!)
    log_mult_2001 = log_cat_2001 + (
        gammaln(N_doc + 1) - np.sum(gammaln(counts_d + 1))
    )

    perplexity_doc = np.exp(-log_cat_2001 / N_doc)

    # ---- All test docs B ----
    ids_B = B[:, 1].astype(int) - 1
    counts_B = B[:, 2].astype(float)
    N_B = counts_B.sum()

    log_cat_B = float(np.sum(counts_B * np.log(theta[ids_B])))
    perplexity_B = np.exp(-log_cat_B / N_B)

    return log_cat_2001, log_mult_2001, perplexity_doc, perplexity_B


def table_and_plot_question_c(word_counts, B, W):
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    cat_LPs = []
    mult_LPs = []
    perp_doc2001 = []
    perp_B_list = []

    print("\n=== Question C: Table (with test set) ===")
    header = (
        f"{'Alpha (α)':>9}  "
        f"{'Categorical LP':>16}  "
        f"{'Multinomial LP':>16}  "
        f"{'Perplexity (Doc 2001)':>22}  "
        f"{'Perplexity (All B)':>20}"
    )
    print(header)
    print("-" * len(header))

    for a in alphas:
        log_cat, log_mult, perp_doc, perp_B = stats_for_alpha(a, word_counts, B, W)
        cat_LPs.append(log_cat)
        mult_LPs.append(log_mult)
        perp_doc2001.append(perp_doc)
        perp_B_list.append(perp_B)

        print(
            f"{a:9.2f}  "
            f"{log_cat:16.2f}  "
            f"{log_mult:16.2f}  "
            f"{perp_doc:22.2f}  "
            f"{perp_B:20.2f}"
        )

    print("-" * len(header))
    print(f"Uniform perplexity (W) = {W}")

    # ---- 画 Perplexity vs Alpha (Doc 2001 + All B) ----
    alphas_arr = np.array(alphas, dtype=float)
    perp_doc2001 = np.array(perp_doc2001)
    perp_B_list = np.array(perp_B_list)

    plt.figure(figsize=(8, 5))
    plt.plot(
        alphas_arr,
        perp_doc2001,
        marker="o",
        linewidth=1.5,
        label="Doc 2001",
    )
    plt.plot(
        alphas_arr,
        perp_B_list,
        marker="s",
        linewidth=1.5,
        label="All test docs (All B)",
    )

    # Uniform baseline
    plt.axhline(
        y=W,
        linestyle="--",
        label=f"Uniform baseline ({W})",
        alpha=0.6,
    )

    plt.xscale("log")
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Perplexity (lower is better)")
    plt.title("Test Perplexity vs Alpha")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("qc_perplexity_doc_vs_all.png", dpi=200)
    print("-> 图表已保存: qc_perplexity_doc_vs_all.png")


# ==========================
# 主入口
# ==========================

def main():
    A, B, W = load_kos_data()
    print(f"训练集 A 形状: {A.shape}, 测试集 B 形状: {B.shape}, 词表大小 W = {W}")

    word_counts = compute_word_counts(A, W)
    print(f"训练集总 token 数: {int(word_counts.sum())}")

    # Question B 图
    plot_question_b(word_counts, W, output_prefix="qb_alpha")

    # Question C 表格 + 图
    table_and_plot_question_c(word_counts, B, W)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
