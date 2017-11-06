import os
import json
import numpy as np
from scipy.misc import imsave


def save_json(path, d):
    fpath = os.path.join(path, 'descriptor.json')
    with open(fpath, 'w') as f:
        json.dump(d, f)


class AdversarialWriterPoison:
    def __init__(self, examination_class, craft_class, target_img, adversarial_data_path):
        if not os.path.isdir(adversarial_data_path):
            os.mkdir(adversarial_data_path)
        target_img_path = os.path.join(adversarial_data_path, 'target.png')
        imsave(target_img_path, target_img)

        self.data_dict = {'examination': int(examination_class),
                          'craft': int(craft_class),
                          'target_path': target_img_path,
                          'root_path': adversarial_data_path,
                          'imgs': {}}
        self.counter = 0
        self.path = adversarial_data_path

    def put(self, base_img, adv_img, final_norm, norm_delta):
        base_img_path = os.path.join(self.path, '{}_base.png'.format(self.counter))
        imsave(base_img_path, base_img)

        adv_img_path = os.path.join(self.path, '{}_adv.png'.format(self.counter))
        imsave(adv_img_path, adv_img)

        self.data_dict['imgs'][self.counter] = {'final_norm': float(final_norm), 'norm_delta': float(norm_delta)}
        self.counter += 1

    def commit(self):
        save_json(self.path, self.data_dict)


class AdversarialWriterEvasion:
    def __init__(self, source_class, target_class, attack_params, adversarial_data_path):
        if not os.path.isdir(adversarial_data_path):
            os.mkdir(adversarial_data_path)

        self.data_dict = {'source': int(source_class),
                          'target': int(target_class),
                          'root_path': adversarial_data_path,
                          'attack_params': attack_params,
                          'imgs': {}}
        self.counter = 0
        self.path = adversarial_data_path

    def batch_put(self, base_imgs, adv_imgs):
        for base_img, adv_img in zip(base_imgs, adv_imgs):
            self.put(base_img, adv_img)

    def put(self, base_img, adv_img):
        def reshape_img(img):
            return np.reshape(img, newshape=(28,28))

        base_img_path = os.path.join(self.path, '{}_base.png'.format(self.counter))
        imsave(base_img_path, reshape_img(base_img))

        adv_img_path = os.path.join(self.path, '{}_adv.png'.format(self.counter))
        imsave(adv_img_path, reshape_img(adv_img))

        self.data_dict['imgs'][self.counter] = {}
        self.counter += 1

    def commit(self):
        save_json(self.path, self.data_dict)
