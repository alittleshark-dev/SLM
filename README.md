# SLM - Small Language Model

一个从零手写的小型语言模型，用于中文文本生成。

> 一个只会乱说话的小语言模型 

A small language model that only talks nonsense.

---

## 项目简介/Project Introduction

本项目是一个**纯手工实现**的小型语言模型（Small Language Model），用于中文文本生成任务。

*不依赖任何深度学习框架（如 PyTorch、TensorFlow、HuggingFace 等），仅使用 Python 基础语法完成全部开发。*

通过从零实现模型的前向传播、反向传播与梯度更新全流程，深入理解语言模型的底层训练原理。

*This project is a manually implemented Small Language Model for Chinese text generation tasks.*

*Not relying on any deep learning frameworks such as PyTorch, TensorFlow, HuggingFace, etc., only using Python basic syntax to complete all development.*

*By implementing the entire process of forward propagation, backward propagation, and gradient update of the model from scratch, we can gain a deeper understanding of the underlying training principles of language models.*

## 技术栈 /Tech stack

- **实现方式**：纯 Python 手写，零第三方依赖
- **模型结构**：基于 Transformer 架构的小型语言模型，包含多头自注意力机制（Multi-Head Self-Attention）、前馈神经网络（FFN）、层归一化（Layer Normalization）及残差连接（Residual Connection）
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
