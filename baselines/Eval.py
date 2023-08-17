from .DownStream import LogReg
import torch

import torch.nn.functional as F

def eval_clf(embeds, labels, idx_train, idx_test):
    train_embs = embeds[idx_train]
    test_embs = embeds[idx_test]

    train_lbls = labels[idx_train]
    test_lbls = labels[idx_test]

    wd = 0.0

    log = LogReg(embeds.shape[1], torch.max(labels) + 1)
    opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
    log.cuda()
    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = F.nll_loss(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]

    print(f'accuracy: {acc * 100}%')

    return acc.item()
