# -*- coding: UTF-8 -*-
# add
import json

import torch
import pandas as pd
import time
from life_long_anomaly_detection.train_lstm_model import Model

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# len(line) < window_length

def generate(name, window_length):
    log_keys_sequences = list()
    with open(name, 'r') as f:
        for line in f.readlines():
            line = tuple(
                map(lambda n: tuple(map(float, n.strip().split())), [x for x in line.strip().split(',') if len(x) > 0]))
            # for i in range(len(line) - window_size):
            #     inputs.add(tuple(line[i:i+window_size]))
            log_keys_sequences.append(tuple(line))
    return log_keys_sequences


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


def do_predict(input_size, hidden_size, num_layers, num_classes, window_length, model_path, anomaly_test_line_path,
               test_file_path, num_candidates, pattern_vec_file):
    # TODO modify
    vec_to_class_type = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for line in pattern_file.readlines():
            pattern, vec = line.split('[:]')
            pattern_vector = tuple(map(float, vec.strip().split(' ')))
            vec_to_class_type[pattern_vector] = i
            i = i + 1

    sequential_model = load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path)

    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    skip_count = 0
    # TODO one file but two types (anomaly and normal) -> 1 and 0
    abnormal_loader = generate(test_file_path, window_length)
    with open(anomaly_test_line_path) as f:
        abnormal_label = [int(x) for x in f.readline().strip().split()]

    print('predict start')
    with torch.no_grad():
        count_num = 0
        current_file_line = 0
        for line in abnormal_loader:
            i = 0
            # first traverse [0, window_size)
            while i < len(line) - window_length:
                lineNum = current_file_line * 200 + i + window_length + 1
                count_num += 1
                seq = line[i:i + window_length]
                label = line[i + window_length]
                for n in range(len(seq)):
                    if current_file_line * 200 + i + n + 1 in abnormal_label:
                        i = i + n + 1
                        continue
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size).to(device)
                # label = torch.tensor(label).view(-1).to(device)
                output = sequential_model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                predicted = filter_small_top_k(predicted, output)
                # print(output)
                # print('{} - predict result: {}, true label: {}'.format(count_num, predicted, vec_to_class_type[tuple(label)]))
                if lineNum in abnormal_label:  ## 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                    i += window_length + 1
                    skip_count += 1
                else:
                    i += 1
                ALL += 1
                if vec_to_class_type[tuple(label)] not in predicted:
                    if lineNum in abnormal_label:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if lineNum in abnormal_label:
                        FN += 1
                    else:
                        TN += 1
            current_file_line += 1
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

    Acc = (TP + TN) * 100 / ALL

    print('FP: {}, FN: {}, TP: {}, TN: {}'.format(FP, FN, TP, TN))
    print('Acc: {:.3f}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(Acc, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
    print('skip_count: {}'.format(skip_count))
    # draw_evaluation("Evaluations", ['Acc', 'Precision', 'Recall', 'F1-measure'], [Acc, P, R, F1], 'evaluations', '%')


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
    SKIP_COUNT = 0
    with torch.no_grad():
        print('predict start', time.time())
        # choose one line to detect
        for _, row in detection_data.iterrows():
            id_list = [int(x) for x in row['Sequence'].strip().split()]
            sequence_length = len(id_list)
            # TODO :跳过了所有的长度不够的问题，但是这之中还是有很多的问题，尤其是很多的异常日志的长度都很短，还有对于ALL的数值的计算问题
            #  可以进行几个实验来尝试一下
            if sequence_length < window_length + 1:
                SKIP_COUNT += 1
                print('The sequence length is shorter than window length , and its type is', row['label'])
                continue
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
    print('skip block count: {}'.format(SKIP_COUNT))
