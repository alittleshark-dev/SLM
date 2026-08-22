# -*- coding: utf-8 -*-
import tokenizer
import embedding
import random
import model
import math
import os

debug = False

print("Load Vocab...")

vocab = {}
for vocab_file in tokenizer.vocab_files:
    vocab.update(tokenizer.import_vocab(vocab_file))
    print(f"Vocab {vocab_file} import complete!")

print("Vocab loading complete!")
print(f"Size: {len(vocab)}")
tok = tokenizer.Tokenizer(vocab)

# ========== 输出头：把 128 维向量映射到词表 ==========
class OutputHead:
    def __init__(self, embedding_dim, vocab_size):
        self.W = [
            [random.uniform(-0.1, 0.1) for _ in range(vocab_size)]
            for _ in range(embedding_dim)
        ]
        self.b = [0.0] * vocab_size

    def forward(self, vec):
        vocab_size = len(self.b)
        logits = list(self.b)
        for i in range(len(vec)):
            for j in range(vocab_size):
                logits[j] += vec[i] * self.W[i][j]
        return logits


def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ========== 加载 Embedding ==========
embed = embedding.Embedding(128)

if os.path.exists("./data/weight.txt"):
    print("weight.txt exists! \nloading...")
    embed.read("weight.txt")
else:
    print("weight.txt no mush file!")
    print("Generate weight matrix...")
    embed.build_weight_matrix()
    embed.save("weight.txt")

print("Generate position matrix...")
embed.gen_pos_matrix()

# ========== 初始化模型和输出头 ==========
mod = model.Model(embedding_dim=128, num_heads=8)
output_head = OutputHead(embedding_dim=128, vocab_size=len(vocab))

# ========== 生成循环 ==========
text = input("You: ")
input_id = tok.encode(text)
if debug:
    print(f"ID: {input_id}")

vecs = embed.id2vec(input_id)
if debug:
    print("Input_vec: ")
    print(vecs)

# 先跑一次模型，得到输出向量
output = mod.block(vecs)
if debug:
    print(f"输入序列长度: {len(vecs)}")
    print(f"输出序列长度: {len(output)}")
    print(f"输出向量维度: {len(output[0])}")

# 开始一个 token 一个 token 地往外蹦
result_ids = list(input_id)
max_gen = 20

for step in range(max_gen):
    safe_ids = [i for i in result_ids if i < len(embed.weight)]
    if not safe_ids:
        safe_ids = [0]

    current_vecs = embed.id2vec(safe_ids)
    output = mod.block(current_vecs)

    last_vec = output[-1]
    logits = output_head.forward(last_vec)
    probs = softmax(logits)

    r = random.random()
    cumulative = 0.0
    next_id = 0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            next_id = i
            break

    result_ids.append(next_id)
    if debug:
        print(f"Step {step+1}: {tok.decode([next_id])}")

final_text = tok.decode(result_ids)
print(f"AI: {final_text}")