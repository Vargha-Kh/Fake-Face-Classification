from torch import nn
import numpy as np
import torch
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from models import load_model

model_path = './best_classification.pth'
model = load_model("regnet")
model.load_state_dict(torch.load(model_path))
model = model.cuda()
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
    print(total_acc_valid, total_loss_valid)
    return true, pred, image, true_wrong, pred_wrong, total_loss_valid, total_acc_valid.item()


def performance_matrix(true, pred):
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    print('Confusion Matrix:\n', metrics.confusion_matrix(true, pred))
    print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision * 100, recall * 100, accuracy * 100,
                                                                         f1_score * 100))


if __name__ == '__main__':
    batch_size = 1
    criterion = nn.CrossEntropyLoss()
    true, pred, image, true_wrong, pred_wrong, total_loss_valid, total_acc_valid = test(test_loader, model, criterion,
                                                                                    "cuda")
    performance_matrix(true, pred)

# def test(model, dataloader):
#     running_corrects = 0
#     running_loss = 0
#     pred = []
#     true = []
#     pred_wrong = []
#     true_wrong = []
#     image = []
#
#     for batch_idx, (data, target) in enumerate(dataloader):
#         data, target = Variable(data), Variable(target)
#         data = data.type(torch.cuda.FloatTensor)
#         target = target.type(torch.cuda.LongTensor)
#         model.eval()
#         output = model(data)
#         loss = criterion(output, target)
#         output = sm(output)
#         _, preds = torch.max(output, 1)
#         running_corrects = running_corrects + torch.sum(preds == target.data)
#         running_loss += loss.item() * data.size(0)
#         preds = preds.cpu().numpy()
#         target = target.cpu().numpy()
#         preds = np.reshape(preds, (len(preds), 1))
#         target = np.reshape(target, (len(preds), 1))
#         data = data.cpu().numpy()
#
#         for i in range(len(preds)):
#             pred.append(preds[i])
#             true.append(target[i])
#             if (preds[i] != target[i]):
#                 pred_wrong.append(preds[i])
#                 true_wrong.append(target[i])
#                 image.append(data[i])
#
#     epoch_acc = running_corrects.double() / (len(dataloader) * batch_size)
#     epoch_loss = running_loss / (len(dataloader) * batch_size)
#     print(epoch_acc, epoch_loss)
#     return true, pred, image, true_wrong, pred_wrong
