from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils_keras import KerasModelWrapper
from data.adversarial_data_writer import AdversarialWriterEvasion


def generate(sess, model, data_feeder, source, target, adv_dump_dir, nb_samples, perturbation_step, max_perturbation):
    wrap = KerasModelWrapper(model.model)
    jsma = SaliencyMapMethod(wrap, sess=sess)
    jsma_params = {
        'gamma': max_perturbation,
        'theta': perturbation_step,
        'clip_min': 0.0,
        'clip_max': 1.0,
        'y_target': data_feeder.get_labels(target, nb_samples)
    }

    craft_data = data_feeder.get_evasion_craft_data(source_class=source, total_count=nb_samples)
    adv_data = jsma.generate_np(craft_data, **jsma_params)

    # Commit data
    adv_writer = AdversarialWriterEvasion(source_class=source,
                                          target_class=target,
                                          attack_params={
                                              'step': perturbation_step,
                                              'max_perturbation':max_perturbation},
                                          adversarial_data_path=adv_dump_dir)
    adv_writer.batch_put(craft_data, adv_data)
    adv_writer.commit()
