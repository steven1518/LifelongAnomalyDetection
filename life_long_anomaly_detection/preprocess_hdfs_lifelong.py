# -*- coding: UTF-8 -*-
import sys
import os
import io
import re
import random
import math
import json
import pandas as pd
import numpy as np

block_id_regex = r'blk_(|-)[0-9]+'
special_patterns = {'dfs.FSNamesystem:': ['dfs', 'FS', 'Name', 'system'], 'dfs.FSDataset:': ['dfs', 'FS', 'dataset']}

log_structured_file_path = 'Data/log/HDFS.log_structured.csv'
log_template_file_path = 'Data/log/HDFS.log_templates.csv'
anomaly_label_file_path = 'Data/log/anomaly_label.csv'
out_dic_path = 'Data/output_and_input/'
train_file_name = 'train_file'
validation_file_name = 'validation_file'
test_file_name = 'test_file'
validation_small_file_name = 'validation_small_file'
word2vec_file_path = 'Data/word_vec/word2vec.vec'
pattern_vec_out_path = 'Data/word_vec/pattern_out'
variable_symbol = '<*>'


# 给template文件添加numberID 作为分别的标识数据
def add_numberid(logparser_templates_file):
    df = pd.read_csv(logparser_templates_file, header=0)
    df['numberID'] = range(1, len(df) + 1)
    print(df)

    df.to_csv(logparser_templates_file, columns=df.columns, index=0, header=1)


# 获得所有的anomaly标识的block_id
def get_anomaly_block_id_set(anomaly_label_file):
    datafile = open(anomaly_label_file, 'r', encoding='UTF-8')
    data = pd.read_csv(datafile)

    data = data[data['Label'].isin(['Anomaly'])]
    # 16838 anomaly block right with the log anomaly paper
    anomaly_block_set = set(data['BlockId'])
    return anomaly_block_set


# 构建字典，key是eventId value是之前标识的numberId
def get_log_template_dic(logparser_event_file):
    dic = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    data = pd.read_csv(datafile)
    for _, row in data.iterrows():
        dic[row['EventId']] = row['numberID']
    return dic


# log parser_file should be structured.csv
# 生成所有的train文件 test文件和validation文件
def generate_train_test_validation_template2vec_file(logparser_structed_file, logparser_event_file, anomaly_label_file,
                                                     out_dic,
                                                     train_out_file_name, validation_out_file_name, test_out_file_name,
                                                     wordvec_path,
                                                     pattern_vec_out_path, variable_symbol):
    add_numberid(logparser_event_file)
    anomaly_block_set = get_anomaly_block_id_set(anomaly_label_file)
    log_template_dic = get_log_template_dic(logparser_event_file)
    session_dic = {}
    logparser_result = pd.read_csv(logparser_structed_file, header=0)
    normal_block_ids = set()
    abnormal_block_ids = set()
    for _, row in logparser_result.iterrows():
        key = row['EventTemplate']
        content = row['Content']
        block_id = re.search(block_id_regex, content).group()
        session_dic.setdefault(block_id, []).append(log_template_dic[row['EventId']])
        if block_id in anomaly_block_set:
            abnormal_block_ids.add(block_id)
        else:
            normal_block_ids.add(block_id)
    abnormal_block_ids = list(abnormal_block_ids)
    normal_block_ids = list(normal_block_ids)
    random.shuffle(abnormal_block_ids)
    random.shuffle(normal_block_ids)
    with open(out_dic + train_out_file_name, 'w+') as train_file_obj, open(out_dic + test_out_file_name,
                                                                           'w+') as test_file_obj, open(
        out_dic + validation_out_file_name, 'w+') as validation_file_obj:
        train_file_obj.write('BlockId,Sequence,label\n')
        test_file_obj.write('BlockId,Sequence,label\n')
        validation_file_obj.write('BlockId,Sequence,label\n')
        for i in range(len(normal_block_ids)):
            if i < 6000:
                train_file_obj.write(str(normal_block_ids[i]) + ', ')
                train_file_obj.write(' '.join([str(num_id) for num_id in session_dic[normal_block_ids[i]]]))
                train_file_obj.write(', 0\n')
            elif i < 6000 + 50000:
                validation_file_obj.write(str(normal_block_ids[i]) + ', ')
                validation_file_obj.write(' '.join([str(num_id) for num_id in session_dic[normal_block_ids[i]]]))
                validation_file_obj.write(', 0\n')
            else:
                test_file_obj.write(str(normal_block_ids[i]) + ', ')
                test_file_obj.write(' '.join([str(num_id) for num_id in session_dic[normal_block_ids[i]]]))
                test_file_obj.write(', 0\n')

        # 此处进行了修改，因为训练集不需要异常的数据类型，所以只处理正常的数据
        for i in range(len(abnormal_block_ids)):
            if i < 1200:
                validation_file_obj.write(str(abnormal_block_ids[i]) + ', ')
                validation_file_obj.write(' '.join([str(num_id) for num_id in session_dic[abnormal_block_ids[i]]]))
                validation_file_obj.write(', 1\n')
            else:
                test_file_obj.write(str(abnormal_block_ids[i]) + ', ')
                test_file_obj.write(' '.join([str(num_id) for num_id in session_dic[abnormal_block_ids[i]]]))
                test_file_obj.write(', 1\n')

    pattern_to_vec(logparser_event_file, wordvec_path, pattern_vec_out_path, variable_symbol)


# 构建词典，导入词向量文件 key是相应的字符 value是相应的300维的词向量
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


# 中间处理，将句子按照大写字母进行拆分
def get_lower_case_name(text):
    word_list = []
    if text in special_patterns:
        return
    for index, char in enumerate(text):
        if not char.isupper():
            break
        else:
            if index == len(text) - 1:
                return [text]
    lst = []
    for index, char in enumerate(text):
        if char.isupper() and index != 0:
            word_list.append("".join(lst))
            lst = []
        lst.append(char)
    word_list.append("".join(lst))
    return word_list


# 去除掉字符串中的所有符号
def preprocess_pattern(log_pattern):
    special_list = []
    if log_pattern.split(' ')[0] in special_patterns.keys():
        special_list = special_patterns[log_pattern.split(' ')[0]]
        log_pattern = log_pattern[len(log_pattern.split(' ')[0]):]
    pattern = r'\*|,|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    result_list = [x for x in re.split(pattern, log_pattern) if len(x) > 0]
    final_list = list(map(get_lower_case_name, result_list))
    final_list.append(special_list)
    return [x for x in re.split(pattern, final_list.__str__()) if len(x) > 0]


# 将相应的template根据频率转换成为词向量
# 这也是预处理的很重要的一个部分，就是将template文件转化成为相应的文件
def pattern_to_vec(logparser_event_file, wordvec_path, pattern_vec_out_path, variable_symbol):
    data = load_vectors(wordvec_path)
    pattern_to_words = {}
    pattern_to_vectors = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    df = pd.read_csv(datafile)
    pattern_num = len(df)
    for _, row in df.iterrows():
        wd_list = preprocess_pattern(row['EventTemplate'].replace(variable_symbol, '').strip())
        pattern_to_words[row['EventTemplate'].replace(variable_symbol, '').strip()] = wd_list
    print(pattern_to_words)
    IDF = {}
    for key in pattern_to_words.keys():
        wd_list = pattern_to_words[key]
        pattern_vector = np.array([0.0 for _ in range(300)])
        word_used = 0
        for word in wd_list:
            if not word in data.keys():
                print('out of 0.1m words', ' ', word)
            else:
                word_used = word_used + 1
                weight = wd_list.count(word) / 1.0 / len(pattern_to_words[key])
                if word in IDF.keys():
                    pattern_vector = pattern_vector + weight * IDF[word] * np.array(data[word])
                else:
                    pattern_occur_num = 0
                    for k in pattern_to_words.keys():
                        if word in pattern_to_words[k]:
                            pattern_occur_num = pattern_occur_num + 1
                    IDF[word] = math.log10(pattern_num / 1.0 / pattern_occur_num)
                    # print('tf', weight, 'idf', IDF[word], word)
                    # print(data[word])
                    pattern_vector = pattern_vector + weight * IDF[word] * np.array(data[word])
        pattern_to_vectors[key] = pattern_vector / word_used
    numberid2vec = {}
    for _, row in df.iterrows():
        numberid2vec[row['numberID']] = pattern_to_vectors[
            row['EventTemplate'].replace(variable_symbol, '').strip()].tolist()
    json_str = json.dumps(numberid2vec)
    with open(pattern_vec_out_path, 'w+') as file_obj:
        file_obj.write(json_str)
    return pattern_to_vectors


if __name__ == '__main__':
    params = ['D:/anomaly_detection/','D:/anomaly_detection/Data/log/HDFS.log_structured.csv']
    params = sys.argv[1:]
    if not os.path.exists(params[0] + log_template_file_path):
        os.makedirs(params[0] + log_template_file_path)
    if not os.path.exists(params[0] + anomaly_label_file_path):
        os.makedirs(params[0] + anomaly_label_file_path)
    if not os.path.exists(params[0] + out_dic_path):
        os.makedirs(params[0] + out_dic_path)
    print(params[0] + log_template_file_path)
    generate_train_test_validation_template2vec_file(
        params[1],
        params[0] + log_template_file_path,
        params[0] + anomaly_label_file_path,
        params[0] + out_dic_path,
        train_file_name,
        validation_file_name,
        test_file_name,
        params[0] + word2vec_file_path,
        params[0] + pattern_vec_out_path,
        variable_symbol
    )
