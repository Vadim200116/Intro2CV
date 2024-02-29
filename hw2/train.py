import logging

import json
import os
import glob
import pandas as pd
import torch

import torch.nn as nn
import torch.optim as optim
    
from torchvision import models, transforms

from pytorch_metric_learning import distances, losses, miners, reducers

from torch.utils.data import Dataset
import skimage


NUM_EPOCHS = 3
BATCH_SIZE = 200

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(message)s')


class MetricLearningDataset(Dataset):
    def __init__(self, img_dir, img_labels, transform=None, target_transform=None) -> None:
        self.img_dir = img_dir
        self.img_labels = img_labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
          return MetricLearningDataset(self.img_dir, self.img_labels[idx], self.transform, self.target_transform)

        img_name = self.img_labels.iloc[idx].name + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = skimage.io.imread(img_path)
        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        logging.info(
            "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                epoch, batch_idx, loss, mining_func.num_triplets
            )
        )


if __name__ == "__main__":
    try:
        data_path = "train"
        with open(os.path.join(data_path, "labels.json")) as f:
            labels = pd.DataFrame.from_dict(json.load(f), orient="index")
        img_paths = glob.glob(os.path.join(data_path, '*.jpg'))

        transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = MetricLearningDataset(img_dir=data_path, img_labels=labels, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True
        )

        shuffle_net = models.shufflenet_v2_x0_5(weights=None)
        shuffle_net.fc = nn.Identity()
        shuffle_net.trainable = False

        model = nn.Sequential(
            shuffle_net,
            nn.Linear(1024, 256),
            nn.Linear(256, 128)
        )
        model.load_state_dict(torch.load('weights.pth', map_location=torch.device('cpu')))


        optimizer = optim.Adam(model.parameters(), lr=3e-5)
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
        mining_func = miners.TripletMarginMiner(
            margin=0.2, distance=distance, type_of_triplets="semihard"
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            logging.info("Training started...")
            train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        

        torch.save(model.state_dict(), "weights.pth")
        logging.info("Model saved...")
    except Exception as e:
        logging.exception("Error:")
        torch.save(model.state_dict(), "weights.pth")
        logging.info("Model saved...")

