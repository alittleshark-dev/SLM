# SLM - Small Language Model

> 一个只会乱说话的小语言模型 

A small language model that only talks nonsense.

- **词库数据**：使用清华大学 THUNLP 的 THUOCL [中文词库](https://github.com/thunlp/THUOCL)
- **核心模块**：
  - Tokenizer 分词器 ---- tokenizer.py
  - Embedding 词嵌入层 ---- embedding.py
  - 前向传播与损失计算 ---- model.py
  - 反向传播与梯度更新 ---- model.py
  - 文本生成推理 ---- train.py

## 使用方式 /Usage

```bash
# 克隆项目 /Clone project
git clone https://github.com/alittleshark-dev/SLM.git
cd SLM
```

#### 训练 /Train

```bash
$ python train.py
```

## 致谢 /Thanks
本项目使用了清华大学 THUNLP 的 THUOCL [中文词库](https://github.com/thunlp/THUOCL)。
*This project uses the [THUOCL](https://github.com/thunlp/THUOCL) Chinese thesaurus of THUNLP, Tsinghua University.*

Copyright © 2026 alittleshark-dev
