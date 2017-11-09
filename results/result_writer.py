import json
import os


class ResultWriter:
    def __init__(self, result_path, attack_path, victim_model_path, result_descriptor=''):
        self.results = []
        self.result_path = os.path.join(result_path, 'results_{}.json'.format(result_descriptor))
        self.attack_path = attack_path
        self.victim_path = victim_model_path

        self.result_counter = 0
        if not os.path.isdir(result_path):
            os.mkdir(result_path)

    def put(self, attack_path, source, target, success_ratio, perturbation_mean, perturbation_std_dev):
        """
        Puts the result into the result dict.
        :param adv_data_path:   Path to adversarial data.
        :param faction:         "attack" or "victim"
        :param mixing:          mixing coefficient * batch_size(e.g. "16", "32", "8")
        :param success:         "True" or "False"
        :param batches:         nb of batches used
        :param samples_used:    nb of adv samples used
        :param confusion:       the confusion mtx (as a list of lists)
        """

        result = {}
        result['attack_path'] = attack_path
        result['source'] = str(source)
        result['target'] = str(target)
        result['success_ratio'] = str(success_ratio)
        result['perturbation_mean'] = str(perturbation_mean)
        result['perturbation_std_dev'] = str(perturbation_std_dev)

        self.results.append(result)

    def commit(self):
        self._dump()

    def _dump(self):
        with open(self.result_path, 'w') as f:
            json.dump(self.results, f)

