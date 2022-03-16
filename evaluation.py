from torch import nn
import pandas as pd
import numpy as np
import torch
from sklearn import metrics
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from models import load_model

model_path = './best_classification.pth'
model = load_model("regnet")
model.load_state_dict(torch.load(model_path))
model = model.to("cpu")
test_transforms = transforms.Compose([
    transforms.Resize((229, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

test_data = datasets.ImageFolder('./fake_real-faces' + '/test', transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8)


def test(test_dataloader, model, criterion, device):
    model.eval()
    loss_ = 0
    valid_acc = 0
    num_image = 0
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []
    for x, y_true in test_dataloader:
        X = x.to(device)
        Y = y_true.to(device)
        logit = model(X)
        loss = criterion(logit.squeeze(1), Y.long())
        loss_ += loss.item() * x.size(0)
        Max, num = torch.max(logit, 1)
        valid_acc += torch.sum(num == Y)
        num_image += x.size(0)

    preds = num.numpy()
    target = Y.numpy()
    preds = np.reshape(preds, (len(preds), 1))
    target = np.reshape(target, (len(preds), 1))
    for i in range(len(preds)):
        pred.append(preds[i])
        true.append(target[i])
        if (preds[i] != target[i]):
            pred_wrong.append(preds[i])
            true_wrong.append(target[i])
            image.append(x[i])

    total_loss_valid = loss_ / num_image
    total_acc_valid = valid_acc / num_image
    print(f"Accuracy: {total_acc_valid.item()}")
    print(f"Loss: {total_loss_valid}")
    return true, pred, image, true_wrong, pred_wrong, total_loss_valid, total_acc_valid.item(), model


def performance_matrix(true, pred, model):
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    print('Confusion Matrix:\n', metrics.confusion_matrix(true, pred))
    print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision * 100, recall * 100, accuracy * 100,
                                                                         f1_score * 100))
    #metrics.plot_confusion_matrix(model, true, pred)

if __name__ == '__main__':
    batch_size = 2
    criterion = nn.CrossEntropyLoss()
    true, pred, image, true_wrong, pred_wrong, total_loss_valid, total_acc_valid, model = test(test_loader, model, criterion,
                                                                                    "cpu")
    performance_matrix(true, pred, model)

