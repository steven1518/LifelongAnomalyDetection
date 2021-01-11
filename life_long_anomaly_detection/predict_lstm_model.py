# -*- coding: UTF-8 -*-
# add
import json
import sys
import torch
import pandas as pd
import time
from life_long_anomaly_detection.train_lstm_model import Model
import os

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path):
    model1 = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model1.load_state_dict(torch.load(model_path, map_location='cpu'))
    model1.eval()
    print('model_path: {}'.format(model_path))
    return model1


def filter_small_top_k(predicted, output):
    filter_list = []
    for p in predicted:
        if output[0][p] > 0.001:
            filter_list.append(p)
    return [x.item() for x in filter_list]


def do_predict_new(input_size, hidden_size, num_layers, num_classes, window_length, model_path, test_file_path,
                   pattern_vec_json, num_candidates, expert_file):
    with open(pattern_vec_json, 'r') as pattern_file:
        number_id_to_vec = json.load(pattern_file)
    sequential_model = load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path)
    detection_data = pd.read_csv(test_file_path)

    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    cn=0
    with torch.no_grad():
        print('predict start')
        # choose one line to detect
        detection_data['prediction'] = 0
        for _, row in detection_data.iterrows():
            cn+=1
            print("times:"+str(cn))
            print('FP: {}, FN: {}, TP: {}, TN: {}'.format(FP, FN, TP, TN))
            id_list = [int(x) for x in row['Sequence'].strip().split()]
            sequence_length = len(id_list)
            if sequence_length < window_length + 1:
                seq = [[-1] * 300 for i in range(window_length + 1 - sequence_length)]
                seq.extend([number_id_to_vec[str(x)] for x in id_list[:-1]])
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size).to(device)
                output = sequential_model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                predicted = filter_small_top_k(predicted, output)
                if id_list[-1] not in predicted:
                    row['prediction'] = 1
                    detection_data.loc[_,'prediction'] = 1
                    # print(row)
                    if int(row['label']) == 0:
                        FP += 1
                    else:
                        TP += 1
                else:
                    # print(row)
                    if int(row['label']) == 0:
                        TN += 1
                    else:
                        FN += 1
            else:
                # 判断每一个窗口的异常与否
                is_anomaly = False
                for i in range(sequence_length - window_length):
                    seq = [number_id_to_vec[str(number_id)] for number_id in id_list[i:i + window_length]]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size).to(device)
                    output = sequential_model(seq)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    predicted = filter_small_top_k(predicted, output)
                    # next is the key phrase
                    if id_list[i + window_length] not in predicted:
                        is_anomaly = True
                        break
                if is_anomaly:
                    row['prediction'] = 1
                    detection_data.loc[_,'prediction'] = 1

                    # print(row)
                    if int(row['label']) == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    # print(row)
                    if int(row['label'] == 0):
                        TN += 1
                    else:
                        FN += 1
    # Compute precision, recall and F1-measure
    if TP + FP == 0:
        P = 0
    else:
        P = 100 * TP / (TP + FP)

    if TP + FN == 0:
        R = 0
    else:
        R = 100 * TP / (TP + FN)

    if P + R == 0:
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)

    detection_data.to_csv(expert_file, index=False)
    Acc = (TP + TN) * 100 / (TP + TN + FP + FN)
    print('FP: {}, FN: {}, TP: {}, TN: {}'.format(FP, FN, TP, TN))
    print('Acc: {:.3f}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(Acc, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))


input_size = 300
hidden_size = 128
num_of_layers = 2
num_of_classes = 48
window_length = 5
root_path = 'Data/'
model_output_directory = root_path + 'model_out/'
batch_size = 4096 * 2 * 2 + 5120 + 1024 + 1024
num_of_epochs = 50
patter_vec_file = 'Data/word_vec/pattern_out'
expert_file = 'Data/expert/detection_file'

if __name__ == '__main__':
    # python predict_lstm_model.py D:/anomaly_detection/ D:/anomaly_detection/Data/output_and_input/train_file
    params = sys.argv[1:]
    params = ['D:/anomaly_detection/','D:/anomaly_detection/Data/output_and_input/validation_small_file']
    if not os.path.exists(params[0] + 'Data/expert'):
        os.makedirs(params[0] + 'Data/expert')
    do_predict_new(
        input_size,
        hidden_size,
        num_of_layers,
        num_of_classes,
        window_length,
        params[0] + model_output_directory + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(
            num_of_epochs) + '.pt',
        params[1],
        params[0] + patter_vec_file,
        10,
        params[0] + expert_file
    )
