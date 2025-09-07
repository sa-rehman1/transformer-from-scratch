# transformer-from-scratch


```markdown
# 🧠 Transformer From Scratch

This project implements a **Transformer-based language model entirely from scratch in PyTorch**, without relying on high-level libraries such as Hugging Face.  
The model is trained on raw text and learns to generate coherent, human-like sequences character by character.

---

## ✨ Highlights
- Built the **Transformer architecture step by step**:
  - Multi-Head Self-Attention
  - Position-wise Feedforward Networks
  - Layer Normalization
  - Residual Connections
  - Embedding + Positional Encoding
- Designed a **character-level language model**
- Fully custom **training loop with AdamW optimizer**
- **Text generation** capability after training

---

## 📂 Project Structure
```

transformer-from-scratch/
│
├── model.py       # Transformer architecture (attention, blocks, model)
├── train.py       # Data preprocessing, training loop, and text generation
├── input.txt      # Raw training text (any plain text can be used)
├── README.md      # Project documentation

````

---

## 🚀 Usage

### 1. Clone the repo
```bash
git clone https://github.com/sa-rehman1/transformer-from-scratch.git
cd transformer-from-scratch
````

### 2. Train the model

```bash
python train.py
```

This will:

* Load and preprocess the text dataset
* Train the Transformer model
* Print the training loss during iterations

### 3. Generate text

After training, the script automatically generates text:

```bash
python train.py --generate
```

---

## 📊 Example Output

After training for a few thousand iterations, the model starts generating text like:

```
KING:
I will speak no more, and so I see the night.
My heart is full of fear, yet still I stay.
```

---

## 💡 Key Learning Outcomes

* Deep understanding of **how attention works under the hood**
* Hands-on experience in **implementing Transformers without shortcuts**
* Built a **mini GPT-style model** capable of text generation
* Strengthened skills in PyTorch, optimization, and training pipelines

---

## 🔗 Inspiration

This project is inspired by foundational work on Transformers and simplified educational implementations.
Special thanks to open-source learning resources that made deep learning more accessible.

---

```

---

```
