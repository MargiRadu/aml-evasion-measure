from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils_keras import KerasModelWrapper
from data.adversarial_data_writer import AdversarialWriterEvasion


def generate(sess, model, data_feeder, source, target, adv_dump_dir, nb_samples, learning_rate=0.1, confidence=0):
    wrap = KerasModelWrapper(model.model)
    cwl2 = CarliniWagnerL2(wrap, sess=sess)

    batch_size = 32
    max_iterations = 450
    abort_early = True
    bin_search_steps = 1
    cwl2_params = {
        'confidence': confidence,
        'learning_rate': learning_rate,
        'binary_search_steps': bin_search_steps,
        'batch_size': batch_size,
        'max_iterations': max_iterations,
        'abort_early': abort_early,
        'initial_const': 10,
        'clip_min': 0.0,
        'clip_max': 1.0,
        'y_target': data_feeder.get_labels(target, nb_samples)
    }

    craft_data = data_feeder.get_evasion_craft_data(source_class=source, total_count=nb_samples)
    adv_data = cwl2.generate_np(craft_data, **cwl2_params)

    # Commit data
    adv_writer = AdversarialWriterEvasion(source_class=source,
                                          target_class=target,
                                          attack_params={
                                              'confidence': confidence,
                                              'learning_rate': learning_rate,
                                              'binary_search_steps': bin_search_steps,
                                              'batch_size': batch_size,
                                              'max_iterations': max_iterations,
                                              'abort_early': abort_early},
                                          adversarial_data_path=adv_dump_dir)
    adv_writer.batch_put(craft_data, adv_data)
    adv_writer.commit()