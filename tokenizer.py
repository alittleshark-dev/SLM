# -*- coding: utf-8 -*-

def import_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip("\n").split("\t")[0]
            vocab[token] = index
    return vocab


class Tokenizer:
    def __init__(self, vocab: dict):
        self.vocab = vocab
        # 反向映射：索引 → token
        self.id2token = {v: k for k, v in vocab.items()}

    def tokenize(self, text: str):
        tokens = []
        ptr = 0
        max_len = 1

        while ptr < len(text):
            if text[ptr:max_len] in self.vocab:
                tokens.append(text[ptr:max_len])
                ptr = max_len
                max_len = ptr + 1
            else:
                max_len += 1
                if max_len > len(text):
                    tokens.append(text[ptr])
                    ptr += 1
                    max_len = ptr + 1

        return tokens

    def encode(self, text: str):
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, 0))
        return ids

    def decode(self, ids):
        return "".join(self.id2token.get(i, "[UNK]") for i in ids)

vocab_files = ["./THUOCL/data/THUOCL_animal.txt",
               "./THUOCL/data/THUOCL_car.txt",
               "./THUOCL/data/THUOCL_diming.txt",
               "./THUOCL/data/THUOCL_IT.txt",
               "./THUOCL/data/THUOCL_lishimingren.txt",
               "./THUOCL/data/THUOCL_poem.txt",
               "./THUOCL/data/THUOCL_caijing.txt",
               "./THUOCL/data/THUOCL_chengyu.txt",
               "./THUOCL/data/THUOCL_food.txt",
               "./THUOCL/data/THUOCL_law.txt",
               "./THUOCL/data/THUOCL_medical.txt",
               "./vocab/Daily_vocab.txt"]

if __name__ == "__main__":
    print("加载词库中...")
    vocab = {}
    for vocab_file in vocab_files:
        vocab.update(import_vocab(vocab_file))
        print(f"词表 {vocab_file} 导入成功")
    print(f"总词表大小: {len(vocab)}")
    print("词表加载完成！")
    tokenizer = Tokenizer(vocab)
    text = input("请输入文本: ")
    tokens = tokenizer.tokenize(text)
    print("分词结果: ", tokens)
    id = tokenizer.encode()
    print(f"ID: {id}")