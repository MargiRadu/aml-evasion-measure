import os
from data.data_feeder import MNISTFeeder
from models.models import MNISTModel
from utils.paths import path_to_mnist, path_to_models

if __name__ == '__main__':
    model_path = os.path.join(path_to_models, 'vanilla_mnist_dropout')
    model = MNISTModel('default', restore=False)
    data = MNISTFeeder(path_to_mnist)

    model.train(data.train_data, data.train_labels, file_name=model_path)
    metrics = model.test(data.test_data, data.test_labels)
    print(metrics)

