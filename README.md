# ProbabilisticML Latent Dirichlet Allocation

## **Coursework 3:   Latent Dirichlet Allocation** 

## **Question A** 

_**Command 1** MLE Calculating_ _**Figure 1** Top 20 words by MLE probability_ 

# A has rows (docID, wordID, count) word_ids, word_counts  = A[:, 1].astype(int) – 1, A[:, 2] V_count = np.zeros(np.max(word_ids) + 1) for wid, cnt in zip(word_ids, word_counts): V_count[wid] += cnt # p(w) = count(w) / total_count theta_ML = V_count / V_count.sum() 

**Figure 1** shows the 20 words with largest 𝜃[̂] 𝑤 , mainly topical political words and common stop-words. **Maximum likelihood:** We use a unigram (bag-of-words) model where each token is drawn i.i.d. from a categorical distribution with parameter 𝜃= (𝜃1, … ,𝜃𝑀) , ∑𝜃𝑤 𝑤 = 1 . Let 𝑐𝑤 be the total count of word 𝑤 in the training set and 𝑁= ∑𝑐𝑤 𝑤 . Under the multinomial model as follows, maximizing log⁡𝐿(𝜃) subject to ∑𝜃𝑤 𝑤 = 1 gives 𝜃[̂] 𝑤[𝑀𝐿] = 𝑐𝑤, which is what **Command 1** computes. 𝑁 

**==> picture [191 x 30] intentionally omitted <==**

**Test-set log probability** : For a test document with word counts 𝑐𝑤 and total length 𝑁 test = ∑𝑐𝑤 𝑤 , the log probability is log⁡ 𝑝( doc ∣𝜃[̂] ) = ∑𝑐𝑤 𝑤log⁡ 𝜃[̂] 𝑤 . 

**Maximum log probability** : Let 𝜃[̂] max = max⁡𝑤 𝜃[̂] 𝑤 be the largest MLE word probability. For fixed 𝑁 test, log⁡𝑝( doc ∣𝜃[̂] ) is maximized by putting all tokens on this single most probable word, so log⁡𝑝max = 𝑁 test log⁡ 𝜃[̂] max. 

**Minimum possible test-set log probability** : If the test vocabulary can contain any word type not seen in training, its MLE probability is 𝜃[̂] 𝑤 new = 0 . Any test document that includes such a word has 𝑝( doc ∣𝜃[̂] ) = 0,log⁡𝑝min = log⁡ 0 = −∞. So the minimum possible test-set log probability for this MLE model is −∞ . 

This illustrates the zero-probability problem of a pure MLE unigram. Any document containing an unseen word gets log-probability −∞ , while a very unnatural document that simply repeats the single most frequent word can achieve the highest possible log-probability. This makes evaluation extremely sensitive to vocabulary mismatch between train and test sets and motivates the use of smoothing or Bayesian methods in later questions. 

## **Question B** 

We place a symmetric Dirichlet prior with concentration parameter 𝛼 on the unigram probabilities 𝜃= (𝜃1,… ,𝜃𝑊) , 𝜃∼Dir(𝛼,… , 𝛼) , where 𝑊 is the vocabulary size. With training counts 𝑁𝑤 and total 𝑁= ∑𝑁𝑤 𝑤 , the MLE from Question A is 𝑝 ML (𝑤∣𝐷) = 𝑁𝑤/𝑁 . Under the Dirichlet prior the 

Page 2 of 6 

posterior is 𝜃∣𝐷∼Dir(𝑁1 + 𝛼, … ,𝑁𝑊 + 𝛼) , giving Bayesian predictive probabilities as follows, which is equivalent to adding a pseudo-count 𝛼 to every word. 

**==> picture [125 x 25] intentionally omitted <==**

𝑁 𝑊𝛼 1 This can be written as 𝑝 Bayes (𝑤∣𝐷,𝛼) = 𝑁+𝑊𝛼 𝑝 ML (𝑤∣𝐷) + 𝑁+𝑊𝛼 𝑊 ~~,~~ so the Bayesian predictive is a convex combination of the MLE and the uniform distribution 1/𝑊 ; in the limits 𝑝 Bayes →𝑝 MLas 𝛼→0 and 𝑝 Bayes →1/𝑊 as 𝛼→∞ . Using the same counts as in Question A, we compute the Bayesian predictive probabilities for several values of 𝛼 with **Command 2** and compare them with the MLE in **Figure 2** , using linear and log–log plots and a zoom into the tail. 

## _**Command 2** Bayesian unigram with Dirichlet(α) prior_ 

theta_bayes = (word_counts + alpha) / (np.sum(word_counts) + W * alpha) 

**==> picture [88 x 84] intentionally omitted <==**

**==> picture [136 x 82] intentionally omitted <==**

**==> picture [130 x 76] intentionally omitted <==**

_(a) Results on  linear axes       (b) Results on log-log axes           (c) Results zoomed into the tail_ 

_**Figure 2** Effect of the Dirichlet prior parameter_ 𝛼 _on unigram word probabilities._ 

For **high-frequency words** with large 𝑁𝑤 , especially when 𝛼 is small, the pseudo-count 𝛼 is negligible, so 𝑝 Bayes (𝑤∣𝐷,𝛼) ≈𝑝 ML (𝑤∣𝐷) and the curves in **Figure 2** almost coincide in the head of the distribution, meaning the prior has little effect on very common words. For **rare or unseen words** with small 𝑁𝑤 , the difference is large: under the MLE a word with 𝑁𝑤 = 0 has 𝑝 ML (𝑤 new ∣𝐷) = 0 , giving log⁡ 𝑝= −∞ for any test document containing it, whereas under the Bayesian model 𝑝 Bayes (𝑤 new ∣𝐷,𝛼) = 𝛼/(𝑁+ 𝑊𝛼) > 0 , which lifts the tail and avoids zero probabilities. Increasing 𝛼 strengthens the prior: probabilities of common words are shrunk towards 1/𝑊 and probabilities of rare words are boosted, so the distribution becomes more uniform; when 𝛼 is small the distribution is close to the MLE but with non-zero mass on unseen words. Thus the Dirichlet( 𝛼 ) Bayesian unigram smoothly interpolates between an unsmoothed MLE model and a strongly regularised almostuniform model while resolving the zero-probability issue for rare and unseen words. 

## **Question C** 

Using the Bayesian predictive distribution from Question B with 𝛼= 0.1 , the log probability of the test document with ID 2001 is log⁡ 𝑝(𝑑2001 ∣𝐷,𝛼) = ∑𝑐𝑤 𝑤,2001log⁡ 𝑝 Bayes (𝑤∣𝐷,𝛼) = −3691.22, see the Categorical LP entry for 𝛼= 0.10 in **Table 1** . We use the **categorical** likelihood rather than the multinomial as the multinomial adds a combinatorial factor 𝑁𝑑!/ ∏𝑐𝑤 𝑑𝑤! that depends only on the document length 𝑁𝑑 and therefore shifts all log-likelihoods for that document by the same constant, which is unhelpful when defining per-word perplexity. 

Page 3 of 6 

## _**Command 3** Compute Log Probability (Categorical) & Perplexity_ 

lp_2001 = np.sum(counts_2001 * np.log(theta_bayes[indices_2001])) perplexity_2001 = np.exp(-lp_2001 / np.sum(counts_2001)) 

**==> picture [145 x 87] intentionally omitted <==**

**==> picture [106 x 89] intentionally omitted <==**

_**Figure 3** Test per–word perplexity of document 2001 and the full test set as a function of_ 𝛼 

1 Per-word perplexity is ppl(𝑑) = exp⁡(− log⁡ 𝑝(𝑑)), so it reflects how well the model predicts the 𝑁𝑑 specific word distribution of document 𝑑 . Documents whose empirical word frequencies are close to the training distribution get lower perplexity, whereas documents like 2001 that contain more rare or idiosyncratic words get higher perplexity. **Table 1** and **Figure 3** show how ppl(𝑑2001) and ppl(𝐵) vary with 𝛼 . Both curves are lowest for moderate values of 𝛼 around 0.1–1. When 𝛼 is very small the amount of smoothing is insufficient, the model assigns too little probability to rare words, and the perplexity increases. When 𝛼 is very large the prior pushes the predictive distribution towards being almost uniform, the model underfits the data, and the perplexity also increases. From **Table 1** , the per-word perplexity for document 2001 is 4398.98 and the average per-word perplexity over all documents in 𝐵 is 2697.11. 

**Table 1** Log Probabilities and Perplexity for varying 𝑎 

|**Alpha(α)**|**Categorical LP**|**Multinomial LP**|**Perplexity (Doc 2001)**|**Perplexity (All B)**|
|---|---|---|---|---|
|**0.01**|-3691.51|-1691.07|4401.93|2704.25|
|**0.10**|-3691.22|-1690.77|4398.98|2697.11|
|**1.00**|-3688.62|-1688.18|4373.11|2683.98|
|**10.00**|-3680.75|-1680.30|4295.55|2730.41|
|**100.00**|-3744.23|-1743.78|4962.19|3703.39|



For a **uniform multinomial** , 𝜃𝑤 = 1/𝑊 for all 𝑤 . Then every token has log-probability log⁡(1/𝑊) = −log⁡ 𝑊 , and for any document or corpus we have: 

**==> picture [285 x 41] intentionally omitted <==**

With vocabulary size 𝑊= 6906 , the per-word perplexity of a uniform multinomial is 6906 , higher than all Bayesian perplexities in **Table 1** and **Figure 3** , showing that the Bayesian unigram model captures meaningful structure in the data compared with a chance-level uniform model. 

## **Question D** 

To monitor convergence I modified the provided bmm.py to record the posterior mixing proportions after each Gibbs sweep. With 𝐷 documents, 𝐾= 20 components and a symmetric Dirichlet prior 

Page 4 of 6 

𝜋∼ Dir (𝛼,… , 𝛼) , the posterior mean of the mixing weight for component 𝑘 after sweep 𝑡 is as (𝑡) follows, where 𝑁𝑘 is the number of documents currently assigned to component 𝑘 . 

**==> picture [142 x 30] intentionally omitted <==**

_**Command 4** Mixing proportions inside Gibbs loop_ 

props = (sk_docs.flatten() + alpha) / (D + K * alpha)   # posterior mean π̂ history[t, :] = props 

**==> picture [303 x 139] intentionally omitted <==**

_**Figure 4** Gibbs sampling mixing proportions and convergence diagnostics_ 

We run the sampler for 50 sweeps with three different random seeds (1, 50, 100). For each run we also record the step-to-step change Δ[(𝑡)] =∣𝜋̂[(𝑡)] −𝜋̂[(𝑡−1)] ∣ . The top row of **Figure 4** shows the trajectories of 𝜋̂𝑘(𝑡) for each seed; the bottom row shows the corresponding Δ(𝑡) . The resulting testset perplexities for seed 1, 50 and 100 are: **2092.27, 2117.59 and 2100.74** . 

Across all three seeds the mixing proportions display a short **burn-in phase** (roughly sweeps 1–10), during which several components rapidly gain most of the mass and many others shrink towards zero. After this period the curves become flat and only fluctuate around stable means, indicating an approximately **stationary phase** . The diagnostic curves Δ[(𝑡)] drop sharply in the first few sweeps and then stay close to zero, showing that successive sweeps make only very small updates once burn-in has finished. Different seeds produce similar stable patterns of high- and low-weight components and very similar test perplexities, suggesting that the chains are sampling from comparable regions of the posterior. Within the scope of this coursework, these trace plots and diagnostics indicate that the Gibbs sampler has effectively converged to the posterior mixing proportions after about 10–20 sweeps, and that 50 sweeps are adequate for this mixture-of-multinomials model on the Daily Kos data. 

## **Question E** 

## _**Command 5** Compute topic posteriors & word entropies per sweep_ 

topic_posterior[it, :] = (sk + alpha) / np.sum(sk + alpha) pw = (swk + gamma) / (sk[None, :] + W * gamma)     # p(w | k) topic_entropy[it, :] = -np.sum(pw * np.log2(pw + 1e-12), axis=0) 

Page 5 of 6 

To analyze convergence of the LDA model, we modified the code so that after each Gibbs sweep we record both the posterior topic proportions and the entropy of each topic’s word distribution. Let 𝑠𝑘𝑑 be the number of tokens in document 𝑑 assigned to each topic and 𝑠𝑘 the total token count for each topic across the corpus. With symmetric Dirichlet priors 𝜃𝑑 ∼Dir(𝛼, … , 𝛼) and 𝛽𝑘 ∼ Dir(𝛾, … , 𝛾) , the posterior expectations are as follows, where 𝑁𝑤𝑘 is the number of times word 𝑤 is assigned to topic 𝑘 and 𝑁𝑘 = ∑𝑁𝑤 𝑤𝑘 : 

**==> picture [271 x 40] intentionally omitted <==**

In the implementation I track the global topic proportions 𝜋̂𝑘 = 𝑠𝑘𝑘+𝛼 and the word entropy of ∑(𝑠𝑘𝑗 𝑗+𝛼) 

each topic 𝐻𝑘 = −∑𝑝(𝑤∣𝑘)log⁡𝑤 2 𝑝(𝑤∣𝑘), measured in bits as we use log⁡2 . 

**==> picture [163 x 79] intentionally omitted <==**

**==> picture [157 x 75] intentionally omitted <==**

_**Figure 5** topic posteriors & entropies_ 

**Figure 5** (left) plots 𝜋̂𝑘 over 50 sweeps. All topics start with similar probabilities; during the first 10– 15 sweeps there is a clear burn-in phase where a few topics rapidly gain most of the mass and many shrink towards near zero. After roughly 20 sweeps the curves flatten and only show small random fluctuations, indicating that the sampler has reached an approximately stationary regime. Using the state after 50 sweeps I compute the per-word perplexity on the held-out test set 𝐵 ; the LDA perplexity (≈ 1.6 × 10[3] ) is much lower than for the Bayesian unigram in Question C (≈ 2.7 × 10[3] ) and the mixture-of-multinomials model in Question D (≈ 2.1 × 10[3] ), so 50 sweeps already give a model that fits the data substantially better than the earlier baselines and appears adequate for this dataset. 

**Figure 5** (right) shows the topic entropies 𝐻𝑘 as a function of sweep. Initially, when assignments are almost random, all topics have high entropy, corresponding to diffuse, almost uniform word distributions. During the first 20–30 sweeps the entropies of most topics drop and then stabilise: low-entropy topics concentrate probability on a small, coherent set of words, whereas higherentropy topics are broader or mix several subthemes. The fact that these entropy curves stabilise at different levels, in bits, indicates that topics have learned different degrees of specificity rather than remaining indistinguishable. 

The final topic posteriors and entropy traces show neither a large group of almost-empty topics nor a set of 20 nearly identical, high-entropy topics. Instead, a modest number of topics carry substantial posterior mass and cover a range of entropy levels, while all topics are used to some extent. Together with the low test-set perplexity compared to simpler models, this suggests that 𝐾= 20 provides a **reasonable level of complexity** for the Daily Kos corpus. It is large enough to capture multiple political themes, but not so large that many components are redundant. 

Page 6 of 6
