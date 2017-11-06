import os
import numpy as np
import json

from PIL import Image


class AdversarialInfo:
    def __init__(self, path):
        """
        Loads the base and adversarial samples from the specified path, as well as the info json.
        :param path:    Path to adversarial directory.
        """

        # Load json
        with open(os.path.join(path, 'descriptor.json'), 'r') as f:
            info_dict = json.load(f)

        self.root_path = info_dict['root_path']
        self.source = info_dict['source']
        self.target = info_dict['target']
        self.attack_params = info_dict['attack_params']
        self.imgs_dict = {int(k):v for k,v in info_dict['imgs'].items()}


class AdversarialData(AdversarialInfo):
    def __init__(self, path):
        """
        Loads the base and adversarial samples from the specified path, as well as the info json.
        :param path:    Path to adversarial directory.
        """
        super().__init__(path)

        # Load images
        adv_imgs = []
        base_imgs = []
        img_indices = list(self.imgs_dict.keys())
        for index in img_indices:
            adv_fname = str(index) + '_adv.png'
            base_fname = str(index) + '_base.png'

            adv_img = Image.open(os.path.join(self.root_path, adv_fname))
            base_img = Image.open(os.path.join(self.root_path, base_fname))

            # TODO: the image size should somehow be specified programatically.
            adv_imgs.append(np.array(adv_img).reshape((28, 28, 3)) / 255.0)
            base_imgs.append(np.array(base_img).reshape((28, 28, 3)) / 255.0)

        def make_ohe(i, n):
            return np.eye(n, dtype=np.float32)[i]

        self.adv_data = np.array(adv_imgs)
        self.target_labels = np.repeat(np.array([make_ohe(self.target, 10)]), repeats=len(img_indices), axis=0)
        self.base_data = np.array(base_imgs)
        self.source_labels = np.repeat(np.array([make_ohe(self.source, 10)]), repeats=len(img_indices), axis=0)


class BoxedAdversarialData(AdversarialData):
    def __init__(self, path, frames):
        super().__init__(path)

        boxed_data = []
        for adv_datum, clean_datum in zip(self.adv_data, self.base_data):
            adv_datum[:frames, :, :] = clean_datum[:frames, :, :]
            adv_datum[-frames:, :, :] = clean_datum[-frames:, :, :]
            adv_datum[:, :frames, :] = clean_datum[:, :frames, :]
            adv_datum[:, -frames:, :] = clean_datum[:, -frames:, :]
            boxed_data.append(np.copy(adv_datum))
        self.adv_data = np.array(boxed_data)


class BlackBoxedAdversarialData(AdversarialData):
    def __init__(self, path, frames):
        super().__init__(path)

        # TODO: the image size should somehow be specified programatically.
        clean_datum = np.full(shape=(28,28,3), fill_value=0.0)
        boxed_data = []
        for adv_datum in self.adv_data:
            adv_datum[:frames, :, :] = clean_datum[:frames, :, :]
            adv_datum[-frames:, :, :] = clean_datum[-frames:, :, :]
            adv_datum[:, :frames, :] = clean_datum[:, :frames, :]
            adv_datum[:, -frames:, :] = clean_datum[:, -frames:, :]
            boxed_data.append(np.copy(adv_datum))
        self.adv_data = np.array(boxed_data)