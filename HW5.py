import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Ref: https://github.com/rwightman/pytorch-image-models
import timm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
y_train = y_train.squeeze()
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
y_test = y_test.squeeze()

print('Number of Data:')
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# It's a multi-class classification problem
class_index = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
               'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

TRAIN = False
TEST = True

MODEL_PATH = "best.pt"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR = 0.00001
IMG_SIZE = (224, 224)
NUM_CLASSES = len(np.unique(y_train))
print("Class #: ", NUM_CLASSES)


class Cifar10(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


# Data augmentation
train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)

train_dataset = Cifar10(x_train, y_train, transform=train_transforms)
val_dataset = Cifar10(x_val, y_val, transform=test_transforms)
test_dataset = Cifar10(x_test, y_test, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

dataloaders = {
    "train": train_loader,
    "val": val_loader,
    "test": test_loader
}
dataset_sizes = {
    "train": len(x_train),
    "val": len(x_val),
    "test": len(x_test)
}


def train_model(model, criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            loop = tqdm(
                enumerate(dataloaders[phase]), total=len(dataloaders[phase])
            )
            for i, (inputs, labels) in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                # print(epoch, scheduler.get_last_lr()[0])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, MODEL_PATH)
                print('Save model. Its acc is {:.4f}'.format(epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


if TRAIN:
    # Model structure
    # print(timm.list_models('cait*', pretrained=True))
    model = timm.create_model(
        'cait_xxs24_224', pretrained=True, num_classes=NUM_CLASSES
    )
    model = model.to(device)
    # print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    model = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)


# evaluate the model by the best weight
if TEST:
    test_model = torch.load(MODEL_PATH, map_location=torch.device(device))
    y_pred = []
    test_model.eval()  # set the model to evaluation mode

    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, (inputs, labels) in loop:
            inputs = inputs.to(device)
            outputs = test_model(inputs)
            # get the index of the class with the highest probability
            _, test_pred = torch.max(outputs, 1)
            for y in test_pred.cpu().numpy():
                y_pred.append(y)

    print("Accuracy of my model on test-set: ", accuracy_score(y_test, y_pred))
