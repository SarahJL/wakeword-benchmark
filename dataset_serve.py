import logging

from dataset import *

def get_mixed_dataset(common_voice_directory, alexa_directory, demand_directory, add_noise=True):
    keyword = 'alexa'

    background_dataset = Dataset.create(Datasets.COMMON_VOICE, common_voice_directory, exclude_words=keyword)
    logging.info('loaded background speech dataset with %d examples' % background_dataset.size())

    keyword_dataset = Dataset.create(Datasets.ALEXA, alexa_directory)
    logging.info('loaded keyword dataset with %d examples' % keyword_dataset.size())

    if add_noise:
        noise_dataset = Dataset.create(Datasets.DEMAND, demand_directory)
        logging.info('loaded noise dataset with %d examples' % noise_dataset.size())
    else:
        noise_dataset = None

    # Interleave the keyword dataset with background dataset to simulate the real-world conditions.
    return CompositeDataset(datasets=(background_dataset, keyword_dataset), shuffle=True)

def get_alexa_dataset(alexa_directory, demand_directory, add_noise=True):
    keyword = 'alexa'

    # background_dataset = Dataset.create(Datasets.COMMON_VOICE, common_voice_directory, exclude_words=keyword)
    # logging.info('loaded background speech dataset with %d examples' % background_dataset.size())

    keyword_dataset = Dataset.create(Datasets.ALEXA, alexa_directory)
    logging.info('loaded keyword dataset with %d examples' % keyword_dataset.size())

    if add_noise:
        noise_dataset = Dataset.create(Datasets.DEMAND, demand_directory)
        logging.info('loaded noise dataset with %d examples' % noise_dataset.size())
    else:
        noise_dataset = None

    # Interleave the keyword dataset with background dataset to simulate the real-world conditions.
    return CompositeDataset(datasets=(keyword_dataset,keyword_dataset), shuffle=True)
