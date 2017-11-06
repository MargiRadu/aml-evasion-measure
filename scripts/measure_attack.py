import os

from results.result_writer import ResultWriter
from models.models import MNISTModel
from data.adversarial_data_feeder import AdversarialData

from utils.paths import path_to_models, path_to_adv_data, path_to_results

if __name__ == '__main__':
    model_path = os.path.join(path_to_models, 'vanilla_mnist_dropout')
    adv_path = os.path.join(path_to_adv_data, 'jsma')
    result_path = os.path.join(path_to_results, 'jsma')

    model = MNISTModel(architecture='default', restore=model_path)
    result_writer = ResultWriter(result_path, adv_path, model_path)
    for dir_name in os.listdir(adv_path)[:1]:
        adv_feeder = AdversarialData(os.path.join(adv_path, dir_name))

        metrics = model.test(adv_feeder.adv_data, adv_feeder.target_labels)
        print(adv_feeder.target_labels[0])
        print(metrics)
        #
        # result_writer.put(
        #     attack_path=dir_name,
        #     source=adv_feeder.source,
        #     target=adv_feeder.target,
        #     success_ratio=,
        #     perturbation_mean=adv_feeder.get_perturbation_mean(),
        #     perturbation_std_dev=adv_feeder.get_perturbation_std_dev()
        # )
