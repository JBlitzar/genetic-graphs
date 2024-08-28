import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

device = "mps" if torch.backends.mps.is_available() else "cpu"


def get_accuracy(model, images,target):
    outputs = model(images)

    prediction = torch.argmax(outputs, dim=1) # 64*10 

    num_correct = (prediction == target).sum()
    total_amt = target.size(0)

    return num_correct




def train_model(model, epochs=10,subdir="test"):

    model = model.to(device)


    log_dir = os.path.join("./runs/",subdir, "tensorboard")
    writer = SummaryWriter(log_dir=log_dir)
    
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),

        
        ])
    mnist_train = datasets.MNIST(root=os.path.expanduser("~/torch_datasets/MNIST"), train=True, download=True, transform=transform)
    mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=64, num_workers=0)
    val_loader = DataLoader(mnist_val, batch_size=64, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    


    for epoch in trange(epochs):
        cumulative_loss = 0
        cumulative_accuracy = 0
        total_samples = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            #print(images.size())
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            cumulative_loss += loss.item()
            total_samples += labels.size(0)
            optimizer.step()
            
            cumulative_accuracy += get_accuracy(model, images, labels)

        avg_train_loss = cumulative_loss / total_samples
        train_acc = cumulative_accuracy / total_samples


        cumulative_loss_val = 0
        total_val_samples = 0
        cumulative_val_accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                predictions = model(images)
                loss = criterion(predictions, labels)

                cumulative_loss_val += loss.item()
                total_val_samples += labels.size(0)

                cumulative_val_accuracy += get_accuracy(model, images, labels)

            model.train()


        avg_val_loss = cumulative_loss / total_samples
        val_acc = cumulative_val_accuracy / total_val_samples

        writer.add_scalar("Loss/train", avg_train_loss, global_step=epoch)
        writer.add_scalar("Loss/val", avg_val_loss, global_step=epoch)
        writer.add_scalar("Acc/train", train_acc, global_step=epoch)
        writer.add_scalar("Acc/val", val_acc, global_step=epoch)


    final_val_loss = avg_val_loss
    print(f"Final Validation Loss: {final_val_loss}")
    return final_val_loss, val_acc

if __name__ == "__main__":
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.layer_1 = torch.nn.Linear(28 * 28, 128)
            self.layer_2 = torch.nn.Linear(128, 256)
            self.layer_3 = torch.nn.Linear(256, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.layer_1(x))
            x = torch.relu(self.layer_2(x))
            x = self.layer_3(x)
            return x


    model = SimpleModel()
    v = train_model(model)

    print(f"Final Validation Loss: {v}")
