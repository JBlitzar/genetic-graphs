import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import os
from pytorch_lightning.loggers import TensorBoardLogger

class LightningTrainer(pl.LightningModule):
    def __init__(self, model, train_loader, val_loader=None, lr=1e-3):
        super(LightningTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x, y = batch
        
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        print(x.size())
        logits = self(x)
        print(logits.size())
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        if self.val_loader:
            return self.val_loader
        return None

    def get_val_loss(self):
        return self.trainer.callback_metrics.get('avg_val_loss').item()

def train_model(model, epochs=5,subdir="test"):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),

        
        ])
    mnist_train = datasets.MNIST(root=os.path.expanduser("~/torch_datasets/MNIST"), train=True, download=True, transform=transform)
    mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=64, num_workers=0)
    val_loader = DataLoader(mnist_val, batch_size=64, num_workers=0)
    trainer = LightningTrainer(model=model, train_loader=train_loader, val_loader=val_loader, lr=1e-3)



    log_dir = os.path.join("./runs/",subdir)


    logger = TensorBoardLogger(save_dir=log_dir, name='pl')


    pl_trainer = pl.Trainer(max_epochs=epochs,logger=logger)
    pl_trainer.fit(trainer)


    final_val_loss = trainer.get_val_loss()
    print(f"Final Validation Loss: {final_val_loss}")
    return final_val_loss

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
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),

        
        ])
    mnist_train = datasets.MNIST(root=os.path.expanduser("~/torch_datasets/MNIST"), train=True, download=True, transform=transform)
    mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=64, num_workers=0)
    val_loader = DataLoader(mnist_val, batch_size=64, num_workers=0)
    trainer = LightningTrainer(model=model, train_loader=train_loader, val_loader=val_loader, lr=1e-3)

    # PyTorch Lightning trainer
    pl_trainer = pl.Trainer(max_epochs=5)
    pl_trainer.fit(trainer)

    # Get final validation loss
    final_val_loss = trainer.get_val_loss()
    print(f"Final Validation Loss: {final_val_loss}")
