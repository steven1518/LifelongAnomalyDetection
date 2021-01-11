# -*- coding: UTF-8 -*-

import json
import sys

import torch
import pandas as pd
import time
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import os
from life_long_anomaly_detection.train_lstm_model import Model

# use cuda if available otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
BND = 1.4
lamda = 1


# TODO: need to think about the problem of "file path","how to load file" and "load which file"

class myLoss(nn.Module):
    def __init__(self, param_old):
        super(myLoss, self).__init__()
        self.param_old = param_old


    def forward(self,output,labels,param_now):
        tag_label_list, abnormal_label_list = [], []
        for label in labels:
            tag_label_list.append(label[0])
            # print("*"*50)
            # print(label[1])
            if label[1] == 0:
                # print("hello")
                abnormal_label_list.append(-1)
            else:
                abnormal_label_list.append(1)
        tag_label=torch.tensor(tag_label_list)
        abnormal_label=torch.tensor(abnormal_label_list)
        criterion_first = nn.CrossEntropyLoss(reduction='none')
        loss_list = criterion_first(output, tag_label.to(device))
        train_loss = torch.zeros(1,1)
        for step, loss in enumerate(loss_list):
            relu=nn.ReLU()
            relu_out=relu(torch.sub(BND, torch.mul(abnormal_label[step], loss)))
            train_loss=torch.add(train_loss.to(device), relu_out.to(device))
        # mul_out=torch.mul(lamda, torch.mul((param_now - self.param_old), (param_now - self.param_old)))
        # print("mul_out:")
        # print(mul_out)
        # train_loss=torch.sub(train_loss,mul_out)
        # print("train_loss:")
        # print(train_loss)
        return train_loss


# 加载模型参数，返回模型
def load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path):
    model1 = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model1.load_state_dict(torch.load(model_path, map_location='cpu'))
    model1.train()
    print('model_path: {}'.format(model_path))
    return model1


# 构建dataset
# 根据每一个block id的所有的的number id序列，来构造基于窗口的dataset
# TODO: need to do sth to handle the problem of id 1
def generate_session_seq_dataset(file_path, window_length, pattern_vec_file):
    # file_path='Data/output_and_input/train_file'
    # window_length=5
    # pattern_vec_file='Data/word_vec/pattern_out'

    with open(pattern_vec_file, 'r') as pattern_file:
        numberId_to_vec = json.load(pattern_file)
    input_data, output_data = [], []
    train_file = pd.read_csv(file_path)
    # 导入pattern_vec_file为numberId_to_vec(‘pattern_out’)，导入train_file为train_file。

    # train_file:
    # BlockId                , Sequence                 ,label
    # blk_8486584878183985736, 1 1 1 2 3 4 3 4 5 5 3 4 5, 0
    # blk_3675462303345607220, 1 1 3 4 3 4 1 3 4 5 2 5 5 32 32 32 17 17 17, 0

    # pattern_vec_file:
    # {"1":[-0.022226713406503295,……, 0.0684038510344474],"2":[-0.022226713406503295,……, 0.0684038510344474],……}
    # "1"/"2"/……下的向量维度为128维
    for _, row in train_file.iterrows():
        session_list = [int(number_id) for number_id in row['Sequence'].strip().split()]
        # session_list: 将sequence中的标签id放入list
        label = int(row['label'])
        if len(session_list) < window_length + 1:
            continue
        # 如果id的个数小于窗口长度，直接略过。
        else:
            for i in range(len(session_list) - window_length):
                tmp = []
                tmp.append(int(session_list[i + window_length]))
                tmp.append(int(label))
                input_data.append([numberId_to_vec[str(x)] for x in session_list[i:i + window_length]])
                output_data.append(tmp)

    data_set = TensorDataset(torch.tensor(input_data, dtype=torch.float),
                             torch.tensor(output_data))
    # TensorDataset 对数据进行打包
    # 在[1,1,3,4,3,4,1,3,4,5,2,5,5,32,32,32,17,17,17]每次选择5个id，然后将对应id的词向量打包放入input_data中
    # 然后将5个id中的第一个id以int形式放入output_data中
    # input_data格式：
    # input_data中的每一个元素都是5x128维的张量，5是window的长度，128是词向量的维度。
    # output_data格式：
    # output_data中的每一个元素是
    return data_set

    # #param
    # window_length = 5 时间窗口长度？？
    # input_size = 300 词向量维度
    # hidden_size = 128 隐藏单元个数
    # num_of_layers = 2 隐藏层层数
    # num_of_classes = 48 训练结果的维度？？
    # num_epochs = 50 训练次数
    # batch_size = 24567 + 2048 一次喂给模型的句子个数
    # root_path = 'Data/'
    # model_output_directory = 'Data/model_out/'
    # data_file = 'Data/output_and_input/train_file'
    # pattern_vec_file = 'Data/word_vec/pattern_out'


def retrain_lifelong(window_length, input_size, hidden_size, num_of_layers, num_of_classes, model_path, num_epochs,
                     batch_size, root_path, model_output_directory, data_file, pattern_vec_file):
    # log setting
    log_directory = root_path + 'retrain_log_out/'
    # 日志文件路径=‘Data/log_out/’
    log_template = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)
    # log_template='Adam_batch_size=24567 + 2048;epoch=50'
    print("Train num_classes[retrain] : ", num_of_classes)

    # load model
    model = load_sequential_model(input_size, hidden_size, num_of_layers, num_of_classes, model_path)

    param_old = model.parameters()

    # create data set 创建数据集
    sequence_data_set = generate_session_seq_dataset(data_file, window_length, pattern_vec_file)
    # create data_loader 创建data_loader
    print('test[retrain]', len(sequence_data_set))
    data_loader = DataLoader(dataset=sequence_data_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    print('major[retrain]', len(data_loader))
    print(len(sequence_data_set) % batch_size)
    writer = SummaryWriter(logdir=log_directory + log_template)
    # 路径：‘Data/log_out/Adam_batch_size=24567 + 2048;epoch=50’
    # tensorboard可视化

    # TODO: modify this criterion
    # Loss and optimizer  classify job
    criterion = myLoss(param_old)
    # 损失函数
    optimizer = optim.Adam(model.parameters())
    # 优化器，将模型参数输入优化器，优化器将在后续步骤中对模型参数进行更新与优化。

    # training
    for epoch in range(num_epochs):
        # 训练50次
        train_loss = 0
        for step, (seq, label) in enumerate(data_loader):
            seq = seq.clone().detach().view(-1, window_length, input_size).to(device)
            # 创建序列
            output = model(seq)
            # 将序列输入到模型中得出结果--48维向量

            loss=criterion(output,label.to(device), model.parameters())
            # 计算损失函数

            # Backward and optimize
            optimizer.zero_grad()
            # 使用优化器将梯度清零
            loss.backward()
            train_loss += loss.item()
            # 进行反向梯度传播的计算
            optimizer.step()
            # 使用优化器的step函数来更新参数
            # print("retrain_model_param:")
            # print(model.state_dict())

        print(
            '[retrain] Epoch [{}/{}], training_loss: {:.6f}'.format(epoch + 1, num_epochs,
                                                                    train_loss / len(data_loader.dataset)))
        if (epoch + 1) % num_epochs == 0:
            if not os.path.isdir(model_output_directory):
                os.makedirs(model_output_directory)
                # 如果系统中没有Data/model_out/这个文件路径，则创建
            e_log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(epoch + 1)
            torch.save(model.state_dict(), model_output_directory + '/' + e_log + '.pt')
            # 最终将每一次的模型以参数形式保存在‘Data/model_out/Adam_batch_size=24567 + 2048;epoch=第几次训练.pt’的文件里。

    writer.close()
    print('Retraining finished')


log_structured_file_path = 'Data/ log/HDFS.log_structured.csv'
log_template_file_path = 'Data/ log/HDFS.log_templates.csv'
anomaly_label_file_path = 'Data/ log/anomaly_label.csv'
out_dic_path = 'Data/output_and_input/'
train_file_name = 'train_file'
validation_file_name = 'validation_file'
test_file_name = 'test_file'
validation_small_file_name = 'validation_small_file'
word2vec_file_path = 'Data/word_vec/word2vec.vec'
pattern_vec_out_path = 'Data/word_vec/pattern_out'
variable_symbol = '<*>'
retrain_model_output = 'Data/retrain_model_out/'
retrain_model_input_file = 'Data/retrain_model_input/detection_file'
# param
window_length = 5
input_size = 300
hidden_size = 128
num_of_layers = 2
num_of_classes = 48
num_of_epochs = 50
batch_size = 512
root_path = 'Data/'
model_output_directory = root_path + 'model_out/'
data_file = 'Data/output_and_input/train_file'
patter_vec_file = 'Data/word_vec/pattern_out'
expert_file = 'Data/expert/detection_file'
test_file_path = out_dic_path + validation_small_file_name

if __name__ == '__main__':
    # python predict_lstm_model.py D:/anomaly_detection/ D:/anomaly_detection/Data/output_and_input/train_file
    params = sys.argv[1:]
    params = ['D:/anomaly_detection/','D:/anomaly_detection/Data/output_and_input/validation_small_file']
    if not os.path.exists(params[0] + 'Data/expert'):
        os.makedirs(params[0] + 'Data/expert')
    retrain_lifelong(
        window_length,
        input_size,
        hidden_size,
        num_of_layers,
        num_of_classes,
        model_output_directory + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_of_epochs) + '.pt',
        # model_output_directory
        num_of_epochs,
        batch_size,
        root_path,
        retrain_model_output,
        retrain_model_input_file,
        patter_vec_file
    )