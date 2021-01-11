import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 构建dataset 具体的思路如下
# 根据每一个block id的所有的的number_id序列，来构造基于窗口的dataset
def generate_session_seq_dataset(file_path, window_length, pattern_vec_file):
    with open(pattern_vec_file, 'r') as pattern_file:
        numberId_to_vec = json.load(pattern_file)
    input_data, output_data = [],[]
    train_file = pd.read_csv(file_path)
    for _, row in train_file.iterrows():
        session_list = [int(number_id) for number_id in row['Sequence'].strip().split()]
        if len(session_list) < window_length + 1:
            seq = [[-1]*300 for x in range(window_length+1-len(session_list))]
            seq.extend([numberId_to_vec[str(x)] for x in session_list[:-1]])
            input_data.append(seq)
            output_data.append(int(session_list[-1]))
        else:
            for i in range(len(session_list) - window_length):
                input_data.append([numberId_to_vec[str(x)] for x in session_list[i:i + window_length]])
                output_data.append(int(session_list[i + window_length]))
    data_set = TensorDataset(torch.tensor(input_data, dtype=torch.float),
                             torch.tensor(output_data))

    return data_set


def train_model(window_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size,
                root_path, model_output_directory, data_file, pattern_vec_file):
    # log setting
    log_directory = root_path + 'log_out/'
    log_template = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)

    print("Train num_classes: ", num_of_classes)
    model = Model(input_size, hidden_size, num_of_layers, num_of_classes).to(device)
    # create data set
    sequence_data_set = generate_session_seq_dataset(data_file, window_length, pattern_vec_file)
    # create data_loader

    data_loader = DataLoader(dataset=sequence_data_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    writer = SummaryWriter(logdir=log_directory + log_template)

    # Loss and optimizer  classify job
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    # TODO: learning rate scheduler

    # Training
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (seq, label) in enumerate(data_loader):
            seq = seq.clone().detach().view(-1, window_length, input_size).to(device)

            output = model(seq)

            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(
            'Epoch [{}/{}], training_loss: {:.6f}'.format(epoch + 1, num_epochs, train_loss / len(data_loader.dataset)))
        if (epoch + 1) % num_epochs == 0:
            if not os.path.isdir(model_output_directory):
                os.makedirs(model_output_directory)
            e_log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(epoch + 1)
            torch.save(model.state_dict(), model_output_directory + '/' + e_log + '.pt')
        # TODO : validation

    writer.close()
    print('Training finished')


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_of_layers, out_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_of_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, out_size)
        # self.out = nn.Linear(in_features=in_features, out_features=out_features)

    def init_hidden(self, size):
        h0 = torch.zeros(self.num_of_layers, size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_of_layers, size, self.hidden_size).to(device)
        return (h0, c0)

    def forward(self, input):
        # h_n: hidden state h of last time step
        # c_n: hidden state c of last time step
        out, _ = self.lstm(input, self.init_hidden(input.size(0)))
        # the output of final time step
        out = self.fc(out[:, -1, :])
        # print('out[:, -1, :]:')
        # print(out)
        return out

window_length = 5
input_size = 300
hidden_size = 128
num_of_layers = 2
num_of_classes = 48
num_of_epochs = 50
batch_size = 4096*2*2+5120+1024+1024
root_path = 'Data/'
model_output_directory = root_path + 'model_out/'
data_file = 'Data/output_and_input/train_file'
patter_vec_file = 'Data/word_vec/pattern_out'

if __name__ == '__main__':
    # python train_lstm_model.py D:/anomaly_detection/  D:/anomaly_detection/Data/output_and_input/train_file
    params = sys.argv[1:]
    if not os.path.exists(params[0] + model_output_directory):
        os.makedirs(params[0] + model_output_directory)
    train_model(
        window_length,
        input_size,
        hidden_size,
        num_of_layers,
        num_of_classes,
        num_of_epochs,
        batch_size,
        params[0] + root_path,
        params[0] + model_output_directory,
        params[1],
        params[0] + patter_vec_file
    )