import os

from results.result_writer import ResultWriter
from models.models import MNISTModel
from data.adversarial_data_feeder import AdversarialData

from utils.paths import path_to_models, path_to_adv_data, path_to_results


def measure(model_name, model_architecture, adv_name, result_name, result_descriptor):
    model_path = os.path.join(path_to_models, model_name)
    adv_path = os.path.join(path_to_adv_data, adv_name)
    result_path = os.path.join(path_to_results, result_name)

    model = MNISTModel(architecture=model_architecture, restore=model_path)
    result_writer = ResultWriter(result_path, adv_path, model_path, result_descriptor)
    for dir_name in os.listdir(adv_path):
        adv_feeder = AdversarialData(os.path.join(adv_path, dir_name))
        acc = model.test(adv_feeder.adv_data, adv_feeder.target_labels)['acc']

        result_writer.put(
            attack_path=dir_name,
            source=adv_feeder.source,
            target=adv_feeder.target,
            success_ratio=acc,
            perturbation_mean=adv_feeder.get_perturbation_mean(),
            perturbation_std_dev=adv_feeder.get_perturbation_std_dev()
        )
    result_writer.commit()


def measure_jsma(model_name, model_architecture, experiment_name):
    measure(model_name=model_name,
            model_architecture=model_architecture,
            adv_name='jsma',
            result_name='jsma',
            result_descriptor=experiment_name)


def measure_cwl2(model_name, model_architecture, experiment_name):
    measure(model_name=model_name,
            model_architecture=model_architecture,
            adv_name='cwl2',
            result_name='cwl2',
            result_descriptor=experiment_name)


def measure_fgsm(model_name, model_architecture, experiment_name):
    measure(model_name=model_name,
            model_architecture=model_architecture,
            adv_name='fgsm',
            result_name='fgsm',
            result_descriptor=experiment_name)


if __name__ == '__main__':
    pass