from train_model.trainer import Trainer
import running_param


appliance_name_list = running_param.appliance_name_list
batch_size = running_param.batch_size
model_type = running_param.model_type
predict_mode = running_param.predict_mode
epochs = running_param.epochs
input_window_length = running_param.input_window_length
validation_frequency = running_param.validation_frequency
dataset = running_param.dataset
learning_rate = running_param.learning_rate
is_load_model = running_param.is_load_model


def train_model():
    if predict_mode == 'single':
        for appliance_name in appliance_name_list:
            training_directory = 'data_process/' + dataset + '/processed_dataset/1min_csv/' + predict_mode + "/" + appliance_name + '_training_.csv'
            validation_directory = 'data_process/' + dataset + '/processed_dataset/1min_csv/' + predict_mode + "/" + appliance_name + '_validation_.csv'
            save_model_dir = "saved_models/" + model_type + "_1min/" + predict_mode + "/" + appliance_name + "_" + model_type + "_model.h5"

            trainer = Trainer(appliance_name, batch_size, model_type,
                              training_directory, validation_directory,
                              save_model_dir, predict_mode, len(appliance_name_list),
                              epochs=epochs, input_window_length=input_window_length,
                              validation_frequency=validation_frequency, learning_rate=learning_rate, is_load_model=is_load_model)
            trainer.train_model()

    elif predict_mode == 'multiple' or predict_mode == 'multi_label':
        training_directory = 'data_process/redd/processed_dataset/1min_csv/' + predict_mode + "/" + 'all' + '_training_.csv'
        validation_directory = 'data_process/redd/processed_dataset/1min_csv/' + predict_mode + "/" + 'all' + '_validation_.csv'
        save_model_dir = "saved_models/" + model_type + "_1min/" + predict_mode + "/" + 'all' + "_" + model_type + "_model.h5"

        trainer = Trainer('all', batch_size, model_type,
                          training_directory, validation_directory,
                          save_model_dir, predict_mode, len(appliance_name_list),
                          epochs=epochs, input_window_length=input_window_length,
                          validation_frequency=validation_frequency, learning_rate=learning_rate, is_load_model=is_load_model)
        trainer.train_model()
