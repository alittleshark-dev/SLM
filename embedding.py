# -*- coding: utf-8 -*-
import math
import random

class Embedding:
    def __init__(self, embedding_dim: int, max_seq_len: int = 512):
        self.weight = []
        self.pos = []
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
    
    def build_weight_matrix(self) -> None:
        """
            SLM这辈子只用一次的初始化权重函数
        """
        for _ in range(5000):
            self.weight.append([random.uniform(-1, 1) for _ in range(self.embedding_dim)])

    def gen_pos_matrix(self) -> None:
        for pos in range(self.max_seq_len):
            pos_vec = []
            for i in range(self.embedding_dim):
                if i % 2 == 0:
                    pos_vec.append(math.sin(pos / (10000 ** (i / self.embedding_dim))))
                else:
                    pos_vec.append(math.cos(pos / (10000 ** ((i - 1) / self.embedding_dim))))
            self.pos.append(pos_vec)

    def id2vec(self, ids: list) -> list:
        temp_list = []
        for i, input_id in enumerate(ids):
            token_vec = self.weight[input_id]
            pos_vec = self.pos[i]
            combined = [a + b for a, b in zip(token_vec, pos_vec)]
            temp_list.append(combined)
        return temp_list

    def save(self, file):
        with open(f"./data/{file}", "w", encoding="utf-8") as f:
            for row in self.weight:
                f.write(" ".join(str(x) for x in row) + "\n")

    def read(self, file):
        with open(f"./data/{file}", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        lines = [l for l in lines if l.strip()]
        self.weight = [[float(x) for x in line.split()] for line in lines]

if __name__ == "__main__":
    import os
    
    embed = Embedding(128)
    ids = [123, 3, 12, 3]
    print("Input_id:", ids)

    if os.path.exists("./data/weight.txt"):
        print("weight.txt exists! \nloading...")
        embed.read("weight.txt")

    else:
        print("weight.txt no mush file!")
        print("Generate weight matrix...")
        embed.build_weight_matrix()
        embed.save("weight.txt")

    # 生成位置矩阵
    print("Generate position matrix...")
    embed.gen_pos_matrix()
    vecs = embed.id2vec(ids)
    print("Input_vec: ")
    print(vecs)
