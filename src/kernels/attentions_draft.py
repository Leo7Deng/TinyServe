# Goal: Compute Attention for ONE head of ONE sequance
# Attention(Q, K, V) = Softmax((Q * K^T) / sqrt(d)) * V
# Q: One vector (Current user query)
# K: Matrix of all past tokens (History)
# V: Matrix of all past values (History)
# Input: Q vector (size 64), Block Table: [4, 7, 2...]

# To write a kernel, we have to "de-vectorize" this, and write what happens to one single number

# Q * K^T is a loop of dot products between the user's current query and each past token
scores = []
for i in range(num_tokens):
    score = dot(Q, K[i])
    scores.append(score)
    
# Softmax is just normalizing
# Stability trick to subtract max score from all scores, then do the softmax exponetiation
max_score = max(scores)

exps = []
sum_exps = 0
for score in scores:
    e = exp(score - max_score)
    exps.append(e)
    sum_exps += e

probs = []
for e in exps:
    probs.append(e / sum_exps)
    

# Get the weighted attention sum (P * V)
output = 0
for i in range(num_tokens):
    output += probs[i] * V[i]


# But in Paged Attention, we cannot simple get K[i]
# Instead of iteration 0..N, we iterate over blocks
for block_idx in block_table:
    for offset in range(16):
        K_vector = K_cache[block_idx][offset]
        score = dot(Q, K_vector)
        scores.append(score)