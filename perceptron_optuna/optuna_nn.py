import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

EPOCHS = 3

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [0 if label == 0 else 1 for label in df['HeartDisease']]
        self.features = df.drop(columns=['HeartDisease'], axis=1).values.tolist()

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_features(self, idx):
        return np.array(self.features[idx])

    def __getitem__(self, idx):
        batch_features = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_features, batch_y

# Build neural network model
def build_model(params):
    in_features = 20
    return nn.Sequential(
        nn.Linear(in_features, params['n_unit']),
        nn.LeakyReLU(),
        nn.Linear(params['n_unit'], 2),
        nn.LeakyReLU()
    )

# Train and evaluate the accuarcy of neural network model
def train_and_evaluate(param, model):

    df = pd.read_csv('./data/heart.csv')
    df = pd.get_dummies(df)

    train_data, val_data = train_test_split(df, test_size = 0.2, random_state = 42)
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])
    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(EPOCHS):
            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in train_dataloader:
                train_label = train_label.to(device)
                train_input = train_input.to(device)
                output = model(train_input.float())
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(device)
                    val_input = val_input.to(device)
                    output = model(val_input.float())
                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            accuracy = total_acc_val/len(val_data)

    return accuracy

def objective(trial):

    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              'n_unit': trial.suggest_int("n_unit", 4, 18)
              }

    model = build_model(params)
    accuracy = train_and_evaluate(params, model)
    return accuracy

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))
