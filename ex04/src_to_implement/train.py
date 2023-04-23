import torch as t
import numpy as np
import pandas as pd
from data import ChallengeDataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from trainer import Trainer
from model import ResNet
from pathlib import Path


# Load the data from the csv file and perform a train-test-split
PATH_DATA_CSV = Path(__file__).parent.absolute() / 'data.csv'
dataset = pd.read_csv(PATH_DATA_CSV, sep=";")
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# Set up data loading for the training and validation dataset
train_dataset = t.utils.data.DataLoader(ChallengeDataset(train, 'train'), batch_size=128, shuffle=False)
val_dataset = t.utils.data.DataLoader(ChallengeDataset(test, 'val'), batch_size=128, shuffle=True)

# Create an instance of our ResNet model
model = ResNet()

# Set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# Set up the optimizer (see t.optim)
# Create an object of type Trainer and set its early stopping criterion
crit = t.nn.BCELoss()
optim = t.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
trainer = Trainer(model,
                  crit,
                  optim=optim,
                  train_dl=train_dataset,
                  val_test_dl=val_dataset,
                  cuda=True,
                  early_stopping_patience=20)

# go, go, go... call fit on trainer
res = trainer.fit(100)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
