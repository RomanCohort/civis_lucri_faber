"""
语言皮层训练 - 最终版
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TextDataset(Dataset):
    def __init__(self, texts, vocab_size=200):
        self.vocab = {'<PAD>': 0}
        for text in texts:
            for w in text.split():
                if w not in self.vocab and len(self.vocab) < vocab_size:
                    self.vocab[w] = len(self.vocab)

        self.data = []
        for text in texts:
            tokens = [self.vocab.get(w, 0) for w in text.split()]
            # 截断
            tokens = tokens[:16]
            if len(tokens) < 16:
                tokens += [0] * (16 - len(tokens))
            self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])


def train():
    print("=== 语言皮层训练 ===")

    base = os.path.dirname(__file__)
    lang = load(os.path.join(base, 'core', 'language_cortex.py'), 'language_cortex')

    # 数据
    phrases = [
        "the cat sat on the mat",
        "a dog runs in the park",
        "the bird flies in sky",
        "i love to read books",
        "she sings a song",
    ]
    texts = phrases * 50

    dataset = TextDataset(texts, vocab_size=100)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"数据: {len(dataset)}, vocab: {len(dataset.vocab)}")

    # 模型
    model = lang.create_language_cortex(vocab_size=100, use_parallel=True)
    model.lm_head = nn.Linear(64, 100)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ce = nn.CrossEntropyLoss(ignore_index=0)

    # 训练
    model.train()
    for epoch in range(5):
        losses = []
        for tokens in loader:
            # 预测下一个词
            result = model(tokens)
            feat = result['features']

            # 下一个词 = 句子中下一个token (简化)
            target = torch.roll(tokens, -1, dims=1)[:, -1]

            pred = model.lm_head(feat)
            loss = ce(pred, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1}: loss={sum(losses)/len(losses):.4f}")

    # 保存
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/language.pt')
    print("保存到 checkpoints/language.pt")
    print("训练完成!")


if __name__ == "__main__":
    train()