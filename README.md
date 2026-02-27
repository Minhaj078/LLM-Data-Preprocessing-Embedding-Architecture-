# 🧠 Mini GPT From Scratch (PyTorch)

This project implements a minimal GPT-style language model completely from scratch using PyTorch.

It covers the full pipeline:
- Custom tokenizer
- Dataset creation (next-token prediction)
- Self-attention mechanism
- GPT architecture
- Training loop
- Text generation
- Model saving

---

## 🚀 Features

✔ Custom word-level tokenizer  
✔ Sliding window dataset for next-token prediction  
✔ Multi-head self-attention  
✔ Positional embeddings  
✔ Training on custom text  
✔ Text generation with sampling  
✔ Model checkpoint saving  

---

## 📁 Project Structure

```text
LLM-Data-Preprocessing-Embedding-Architecture/
│
├── data/
│   └── input.txt
│
├── src/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
│
├── gpt_model.pth
└── README.md
```

---

## 🏗 Model Architecture

- Token Embedding
- Positional Embedding
- Self-Attention Layer
- LayerNorm
- Linear Output Layer

This model predicts the next token given previous context.

---

## ▶️ How to Run

1. Install dependencies:
   ```pip install torch numpy```

2. Add training text inside:
  ```data/input.txt```

3. Train the model:
   ```python src/train.py```

4. Generated text will be printed in terminal.
  
5. Model weights will be saved as:

---

## 📊 Sample Output
```
Epoch 0 | Loss: 3.77
Epoch 4 | Loss: 3.11

=== Text Generation ===
project scratch use builds sentence from in data .
```

---

## 🎯 Learning Outcomes

This project demonstrates understanding of:

- Transformer architecture fundamentals
- Self-attention implementation
- Language modeling
- PyTorch training workflows
- End-to-end NLP pipeline development

---

## 🔥 Future Improvements

- Add multiple transformer blocks
- Add temperature sampling
- Add model loading script
- Train on larger dataset
- Add evaluation metrics (Perplexity)

---

## 👨‍💻 Author

Minhaj Ahmad  
Built as part of LLM deep learning exploration.

