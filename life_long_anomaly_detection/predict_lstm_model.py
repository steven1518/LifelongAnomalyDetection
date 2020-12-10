# -*- coding: UTF-8 -*-
import json

import torch
import pandas as pd
import time
from life_long_anomaly_detection.train_lstm_model import Model

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
                   pattern_vec_json, num_candidates):
    with open(pattern_vec_json, 'r') as pattern_file:
        number_id_to_vec = json.load(pattern_file)
    sequential_model = load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path)
    detection_data = pd.read_csv(test_file_path)

    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    with torch.no_grad():
        print('predict start')
        # choose one line to detect
        for _, row in detection_data.iterrows():
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
                    if int(row['label']) == 0:
                        FP += 1
                    else:
                        TP += 1
                else:
                    if int(row['label']) == 0:
                        TN += 1
                    else:
                        FN += 1
            else:
                # this is the num of the window that can be made
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
                    if int(row['label']) == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
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

    Acc = (TP + TN) * 100 / (TP + TN + FP + FN)
    print('FP: {}, FN: {}, TP: {}, TN: {}'.format(FP, FN, TP, TN))
    print('Acc: {:.3f}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(Acc, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
