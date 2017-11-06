from attack.jsma import generate
from data.data_feeder import MNISTFeederEvasion
from models.models import MNISTModel
from utils.paths import path_to_mnist, path_to_models, path_to_adv_data
import keras.backend as K
import os
import tensorflow as tf

if __name__ == '__main__':
    with tf.Session() as sess:
        K.set_session(sess)
        data = MNISTFeederEvasion(path_to_mnist)
        model = MNISTModel(architecture='default', restore=os.path.join(path_to_models, 'vanilla_mnist'))
        for i in range(1):
            for j in range(1):
                print(f"Generating S:{i} T:{j}...")
                generate(sess=sess,
                         model=model,
                         data_feeder=data,
                         source=i,
                         target=j,
                         adv_dump_dir=os.path.join(path_to_adv_data, 'jsma', f'src{i}_tg{j}'),
                         nb_samples=20,
                         perturbation_step=1.0,
                         max_perturbation=0.12)

