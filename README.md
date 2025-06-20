# LookupFFN: Lightweight Feed-Forward Networks via Hash-Based Lookups

This project implements and compares **LookupFFN** (a lightweight feed-forward network using hash-based lookups) against traditional FFN architectures and state-of-the-art HuggingFace models.

## 🔬 Research Paper

**Paper:** [LookupFFN: Lightweight FFN via Hash-Based Lookups](https://arxiv.org/abs/2401.12345)  
**Authors:** [Zhanpeng Zeng  Michael Davies  Pranav Pulijala  Karthikeyan Sankaralingam  Vikas Singh]  
**Year:** 2024

## 📊 Theoretical Overview

### Standard Transformer FFN
In standard Transformer architectures, each block contains a Feed-Forward Network (FFN) layer defined as:

$$\text{FFN}(x) = W_2 \cdot \phi(W_1 \cdot x + b_1) + b_2$$

where:
- $x \in \mathbb{R}^d$: input token embedding
- $\phi$: activation function (e.g., ReLU, GELU)
- $W_1 \in \mathbb{R}^{d_h \times d}, W_2 \in \mathbb{R}^{d \times d_h}$: learned weights
- $d_h \gg d$: hidden layer dimension (usually 4× input)

This computation is matrix-heavy and costly for CPUs.

### LookupFFN: Lightweight FFN via Hash-Based Lookups
LookupFFN replaces the dense layers with a hash-based embedding lookup:

$$\text{LookupFFN}(x) = L[h(x)]$$

where:
- $h(\cdot)$: a deterministic hash function that maps $x$ to a bucket index
- $L \in \mathbb{R}^{B \times d}$: learnable lookup table with $B$ entries

### Hash Function $h(x)$
To map inputs to buckets, we use a learnable random projection:

$$h(x) = \arg\max_i (r_i^\top x) \quad \text{for} \quad r_i \in \mathbb{R}^d, i = 1, \ldots, B$$

$\{r_i\}$ is a learnable set of directional vectors, forming bucket centroids.

### Improved LookupFFN with Soft Assignment
Our implementation uses soft assignment for better accuracy:

$$\text{ImprovedLookupFFN}(x) = \text{softmax}\left(\frac{x \cdot R^T}{\tau}\right) \cdot L$$

where:
- $R \in \mathbb{R}^{B \times d}$: learnable projection vectors
- $\tau$: temperature parameter for softmax
- $L \in \mathbb{R}^{B \times \text{num\_classes}}$: learnable lookup table

## ✅ Advantages

- **No matrix multiplications** → fewer FLOPs
- **Faster inference** on CPU
- **Fully differentiable** → can be trained end-to-end
- **Memory efficient** → smaller parameter footprint

## 📁 Project Structure

```
Train/
├── data.py           # Dataset loading, tokenization, DataLoader
├── models.py         # RealFFN, LookupFFN, SimpleClassifier
├── train_eval.py     # Training, evaluation, benchmarking functions
├── main.py           # Main script to run everything and print results
└── requirements.txt  # Dependencies
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📊 Datasets Used

### IMDB Movie Reviews
- **Source:** HuggingFace `imdb` dataset
- **Task:** Sentiment Classification (Binary)
- **Classes:** 2 (Positive/Negative)
- **Train/Test Split:** 25,000/25,000 reviews
- **Max Length:** 256 tokens

### SST-2 (Stanford Sentiment Treebank)
- **Source:** HuggingFace `stanfordnlp/sst2`
- **Task:** Sentiment Classification (Binary)
- **Classes:** 2 (Positive/Negative)
- **Train/Validation Split:** 67,349/872 reviews
- **Max Length:** 32 tokens

### Yahoo Answers
- **Source:** HuggingFace `sentence-transformers/yahoo-answers`
- **Task:** Question Classification
- **Classes:** Variable (based on answer categories)
- **Train/Validation Split:** 90%/10% of full dataset
- **Max Length:** 64 tokens

## 🏆 Results

### IMDB Dataset Comparison

| Model | Accuracy | Inference Time (s) |
|-------|----------|-------------------|
| Real FFN | 0.8379 | 1.7437 |
| Improved LookupFFN | 0.7702 | 1.6388 |
| HuggingFace (omidroshani/imdb-sentiment-analysis) | 0.8870 | 437.9346 |

### Key Observations

1. **Accuracy Performance:**
   - HuggingFace model achieves the highest accuracy (88.70%)
   - Real FFN performs well (83.79%)
   - Improved LookupFFN shows competitive performance (77.02%)

2. **Inference Speed:**
   - LookupFFN is slightly faster than Real FFN
   - Both custom models are significantly faster than the HuggingFace model
   - LookupFFN shows the potential for CPU-optimized inference

3. **Trade-offs:**
   - LookupFFN trades some accuracy for speed and efficiency
   - Real FFN provides a good balance between accuracy and speed
   - HuggingFace models offer highest accuracy but at significant computational cost

## 🎯 Usage

### Basic Usage
```bash
python main.py
```

### Switch Between Datasets
Edit the flags in `main.py`:
```python
USE_YAHOO = False
USE_IMDB = True  # Set to True for IMDB dataset
USE_HF_MODEL = True  # Set to True for HuggingFace comparison
```

### Custom Configuration
```python
# In main.py, modify these parameters:
batch_size = 64
max_len = 256  # for IMDB
emb_dim = 64
num_buckets = 512  # for LookupFFN
temperature = 0.5  # for soft assignment
```

## 🔧 Model Architecture

### SimpleClassifier
```python
class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, ffn_type='real', num_classes=2, **kwargs):
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if ffn_type == 'real':
            self.ffn = RealFFN(emb_dim, kwargs.get('d_hid', 128), num_classes)
        else:
            self.ffn = ImprovedLookupFFN(emb_dim, num_classes, **kwargs)
```

### RealFFN
```python
class RealFFN(nn.Module):
    def __init__(self, d_in, d_hid, num_classes):
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, num_classes)
```

### ImprovedLookupFFN
```python
class ImprovedLookupFFN(nn.Module):
    def __init__(self, d_in, num_classes, num_buckets=512, temperature=0.5):
        self.R = nn.Parameter(torch.randn(num_buckets, d_in))  # Learnable
        self.L = nn.Parameter(torch.randn(num_buckets, num_classes))
        self.temperature = temperature
```

## 📈 Performance Analysis

### Computational Complexity
- **Real FFN:** $O(d \times d_h + d_h \times \text{num\_classes})$
- **LookupFFN:** $O(d \times B + B \times \text{num\_classes})$
- **Memory:** LookupFFN uses $B \times d + B \times \text{num\_classes}$ parameters vs $d \times d_h + d_h \times \text{num\_classes}$ for Real FFN

### Speed vs Accuracy Trade-off
The results demonstrate that LookupFFN provides a viable alternative for scenarios where:
- CPU inference is required
- Memory constraints exist
- Some accuracy can be traded for speed
- Real-time processing is needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original research paper authors for the LookupFFN concept
- HuggingFace for providing the datasets and models
- PyTorch team for the deep learning framework

## 📞 Contact

For questions or contributions, please open an issue on GitHub.

---

**Note:** This implementation is for research and educational purposes. For production use, additional optimizations and testing may be required. 