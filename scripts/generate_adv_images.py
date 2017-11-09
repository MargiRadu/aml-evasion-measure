from attack.jsma import generate as jsma
from attack.carlini_wagner import generate as cwl2
from data.data_feeder import MNISTFeederEvasion
from models.models import MNISTModel
from utils.paths import path_to_mnist, path_to_models, path_to_adv_data
import keras.backend as K
import os
import tensorflow as tf


def jsma_generate():
    with tf.Session() as sess:
        K.set_session(sess)
        data = MNISTFeederEvasion(path_to_mnist)
        model = MNISTModel(architecture='default', restore=os.path.join(path_to_models, 'vanilla_mnist'))
        for i in range(10):
            for j in range(10):
                print(f"Generating S:{i} T:{j}...")
                jsma(sess=sess,
                     model=model,
                     data_feeder=data,
                     source=i,
                     target=j,
                     adv_dump_dir=os.path.join(path_to_adv_data, 'jsma', f'src{i}_tg{j}'),
                     nb_samples=15,
                     perturbation_step=1.0,
                     max_perturbation=0.1)


def cwl2_generate():
    with tf.Session() as sess:
        K.set_session(sess)
        data = MNISTFeederEvasion(path_to_mnist)
        model = MNISTModel(architecture='default', restore=os.path.join(path_to_models, 'vanilla_mnist_cw'))
        for i in range(10):
            for j in range(10):
                print(f"Generating S:{i} T:{j}...")
                cwl2(sess=sess,
                     model=model,
                     data_feeder=data,
                     source=i,
                     target=j,
                     adv_dump_dir=os.path.join(path_to_adv_data, 'cwl2', f'src{i}_tg{j}'),
                     nb_samples=32)


if __name__ == '__main__':
    cwl2_generate()

