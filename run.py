import life_long_anomaly_detection.preprocess_hdfs_lifelong as preprocess
import life_long_anomaly_detection.train_lstm_model as train
import life_long_anomaly_detection.predict_lstm_model as predict
import life_long_anomaly_detection.retrain_lstm_lifelong as retrain

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


def lifelong_preprocess():
    preprocess.generate_train_test_validation_template2vec_file(
        log_structured_file_path,
        log_template_file_path,
        anomaly_label_file_path,
        out_dic_path,
        train_file_name,
        validation_file_name,
        test_file_name,
        word2vec_file_path,
        pattern_vec_out_path,
        variable_symbol
    )


def lifelong_train():
    train.train_model(
        window_length,
        input_size,
        hidden_size,
        num_of_layers,
        num_of_classes,
        num_of_epochs,
        batch_size,
        root_path,
        model_output_directory,
        data_file,
        patter_vec_file
    )

def lifelong_retrain():
    retrain.retrain_lifelong(
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

def lifelong_predict():
    predict.do_predict_new(
        input_size,
        hidden_size,
        num_of_layers,
        num_of_classes,
        window_length,
        root_path + 'retrain_model_out/' + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_of_epochs) + '.pt',
        # model_output_directory
        test_file_path,
        patter_vec_file,
        10,
        expert_file
    )


# run run run
# lifelong_preprocess()
# lifelong_train()
# lifelong_predict()
lifelong_retrain()
