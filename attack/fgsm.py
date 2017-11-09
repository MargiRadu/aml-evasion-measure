from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from data.adversarial_data_writer import AdversarialWriterEvasion


def generate(sess, model, data_feeder, source, target, adv_dump_dir, nb_samples, perturbation=0.35):
    wrap = KerasModelWrapper(model.model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {
        'eps': perturbation,
        'clip_min': 0.0,
        'clip_max': 1.0,
        'y_target': data_feeder.get_labels(target, nb_samples)
    }

    craft_data = data_feeder.get_evasion_craft_data(source_class=source, total_count=nb_samples)
    adv_data = fgsm.generate_np(craft_data, **fgsm_params)

    # Commit data
    adv_writer = AdversarialWriterEvasion(source_class=source,
                                          target_class=target,
                                          attack_params={
                                              'eps': perturbation},
                                          adversarial_data_path=adv_dump_dir)
    adv_writer.batch_put(craft_data, adv_data)
    adv_writer.commit()
