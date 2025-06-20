from data import get_dataloaders, get_dataloaders_yahoo, get_dataloaders_imdb
from models import SimpleClassifier
from train_eval import train, evaluate, benchmark_inference
import torch
import time

# For HuggingFace model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Using device:", device)

    # Switch between datasets here
    USE_YAHOO = False
    USE_IMDB = True
    USE_HF_MODEL = True  # Set to True to use HuggingFace model for comparison

    if USE_YAHOO:
        print("\nLoading Yahoo Answers dataset...")
        train_loader, test_loader, tokenizer = get_dataloaders_yahoo(batch_size=64, max_len=64)
        num_classes = len(set([y.item() for _, y in train_loader.dataset]))
        print(f"Number of classes: {num_classes}")
        print("Sample:", train_loader.dataset[0])
    elif USE_IMDB:
        print("\nLoading IMDB dataset...")
        train_loader, test_loader, tokenizer = get_dataloaders_imdb(batch_size=64, max_len=256)
        num_classes = 2
        print("Sample:", train_loader.dataset[0])
    else:
        print("\nLoading SST-2 dataset...")
        train_loader, test_loader, tokenizer = get_dataloaders(batch_size=64, max_len=32)
        num_classes = 2
        print("Sample:", train_loader.dataset[0])

    vocab_size = len(tokenizer.vocab)
    emb_dim = 64
    
    results = []

    # Real FFN
    model_real = SimpleClassifier(vocab_size, emb_dim, ffn_type='real', d_hid=128, num_classes=num_classes).to(device)
    opt_real = torch.optim.Adam(model_real.parameters(), lr=1e-3)
    for epoch in range(3):
        train(model_real, train_loader, opt_real, device)
    acc_real = evaluate(model_real, test_loader, device)
    time_real = benchmark_inference(model_real, test_loader, device)
    results.append({'Model': 'Real FFN', 'Accuracy': acc_real, 'Inference Time (s)': time_real})


    # Improved Lookup FFN
    print("\nUsing ImprovedLookupFFN (soft assignment, learnable hash)...")
    model_lookup = SimpleClassifier(vocab_size, emb_dim, ffn_type='lookup', num_buckets=512, temperature=0.5, num_classes=num_classes).to(device)
    opt_lookup = torch.optim.Adam(model_lookup.parameters(), lr=1e-3)
    for epoch in range(3):
        train(model_lookup, train_loader, opt_lookup, device)
    acc_lookup = evaluate(model_lookup, test_loader, device)
    time_lookup = benchmark_inference(model_lookup, test_loader, device)
    results.append({'Model': 'Improved LookupFFN', 'Accuracy': acc_lookup, 'Inference Time (s)': time_lookup})
    
    # HuggingFace model evaluation
    if USE_HF_MODEL and (USE_YAHOO or USE_IMDB):
        if USE_IMDB:
            print("\nEvaluating HuggingFace model (omidroshani/imdb-sentiment-analysis) on IMDB test set...")
            hf_tokenizer = AutoTokenizer.from_pretrained("omidroshani/imdb-sentiment-analysis")
            hf_model = AutoModelForSequenceClassification.from_pretrained("omidroshani/imdb-sentiment-analysis").to(device)
            hf_model.eval()
            all_preds, all_labels = [], []
            start_time = time.time()
            with torch.no_grad():
                for x, y in tqdm(test_loader, desc="HF Model Eval"):
                    texts = [tokenizer.decode([t.item() for t in tokens if t.item() != 0]) for tokens in x]
                    encodings = hf_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
                    input_ids = encodings['input_ids'].to(device)
                    attention_mask = encodings['attention_mask'].to(device)
                    outputs = hf_model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y.cpu().numpy())
            hf_time = time.time() - start_time
            hf_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            results.append({'Model': 'HuggingFace (omidroshani/imdb-sentiment-analysis)', 'Accuracy': hf_acc, 'Inference Time (s)': hf_time})

    # Print final results table
    print("\n--- Detailed Comparison ---")
    print(f"{'Model':<40} {'Accuracy':<10} {'Inference Time (s)':<20}")
    print("-" * 75)
    for res in results:
        print(f"{res['Model']:<40} {res['Accuracy']:<10.4f} {res['Inference Time (s)']:<20.4f}") 