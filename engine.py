#
# Copyright 2018 Picovoice Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import platform
from enum import Enum

import numpy as np
from pocketsphinx import get_model_path
from pocketsphinx.pocketsphinx import Decoder

from engines import Porcupine
# from engines import snowboydetect


class Engines(Enum):
    """Different wake-word engines."""

    # KERAS_CNN = 'KerasCNN'
    KERAS_CAPSULE = 'KerasCapsuleEngine'
    # POCKET_SPHINX = 'Pocketsphinx'
    # PORCUPINE = 'Porcupine'
    # PORCUPINE_TINY = "PorcupineTiny"
    # SNOWBOY = 'Snowboy'


class Engine(object):
    """Base class for wake-word engine."""

    def process(self, pcm):
        """
        Processes a frame of audio looking for a specific wake-word.

        :param pcm: frame of audio.
        :return: result of detection.
        """

        raise NotImplementedError()

    def release(self):
        """Releases the resources acquired by the engine."""

        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    @property
    def frame_length(self):
        """Number of audio samples per frame expected by the engine."""

        return 512

    @staticmethod
    def sensitivity_range(engine_type):
        """Getter for sensitivity range of different engines to use in the benchmark."""
        # if engine_type is Engines.KERAS_CNN:
        #     # return np.linspace(0.0, 1.0, 10)
        #     return np.array([0.1, 0.5, 0.9])
        if engine_type is Engines.KERAS_CAPSULE:
            return np.array([0.1, 0.5, 0.9])
        # if engine_type is Engines.PORCUPINE:
        #     return np.linspace(0.0, 1.0, 10)
        # if engine_type is Engines.PORCUPINE_TINY:
        #     return np.linspace(0.0, 1.0, 10)
        # if engine_type is Engines.POCKET_SPHINX:
        #     return np.logspace(-10, 20, 10)
        # if engine_type is Engines.SNOWBOY:
        #     return np.linspace(0.4, 0.6, 10)

        raise ValueError('No sensitivity range for %s', engine_type.value)

    @staticmethod
    def create(engine_type, keyword, sensitivity, kwargs):
        """
        Factory method.

        :param engine_type: type of engine.
        :param keyword: keyword to be detected.
        :param sensitivity: detection sensitivity.
        :return: engine instance.
        """
        if engine_type is Engines.KERAS_CAPSULE:
            return KerasCapsuleEngine(keyword, sensitivity, kwargs)
        if engine_type is Engines.KERAS_CNN:
            return KerasCNNEngine(keyword, sensitivity, kwargs)
        if engine_type is Engines.POCKET_SPHINX:
            return PocketSphinxEngine(keyword, sensitivity)
        if engine_type is Engines.PORCUPINE:
            return PorcupineEngine(keyword, sensitivity)
        if engine_type is Engines.PORCUPINE_TINY:
            return PorcupineTinyEngine(keyword, sensitivity)
        # if engine_type is Engines.SNOWBOY:
        #     return SnowboyEngine(keyword, sensitivity)

        return ValueError('Cannot create engine of type %s', engine_type.value)

class KerasCNNEngine(Engine):
    """ Custom engine. """

    def __init__(self, keyword, sensitivity, kwargs={}):
        """
        Constructor.

        :param keyword: keyword to be detected.
        :param sensitivity: detection sensitivity.
        """
        input_shape = (self.frame_length, 1, 1)
        num_classes=2
        self.sensitivity = sensitivity

        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
        from keras.losses import categorical_crossentropy
        from keras.optimizers import Adadelta

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 1),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adadelta(),
                      metrics=['accuracy'])
        self.model = model

    @property
    def frame_length(self):
        """Number of audio samples per frame expected by the engine."""

        # return max(self.model.input.shape.as_list()[1:])
        return 2**13

    def process(self, pcm):
        """
        Method to check each frame for the keyword
        :param pcm: input audio
        :return: bool
        """
        # pcm = (np.iinfo(np.int16).max * pcm).astype(np.int16).tobytes()
        pcm2 = np.expand_dims(pcm, axis=-1)
        pcm2 = np.expand_dims(pcm2, axis=-1)
        pcm2 = np.expand_dims(pcm2, axis=0)
        y_pred = self.model.predict(pcm2)

        detected = y_pred[0][0] > self.sensitivity

        return detected

    def release(self):
        # clear keras session
        from keras import backend as K
        K.clear_session()

        # # force cuda to release memory
        # from numba import cuda
        # try:
        #     cuda.select_device(0)
        #     cuda.close()
        # except:
        #     print("Couldn't release cuda: are you on cpu?")

        return

    def __str__(self):
        return 'KerasCNN'

class KerasCapsuleEngine(Engine):
    """ Custom engine. """

    def __init__(self, keyword, sensitivity, kwargs={}):
        """
        Constructor.

        :param keyword: keyword to be detected.
        :param sensitivity: detection sensitivity.
        """

        self.keyword = keyword
        self.sensitivity = sensitivity
        self.model = self.load_capsnet_model(
            model_def=kwargs['model_def'],
            model_weights=kwargs['model_weights']
        )

    def load_capsnet_model(self, model_def='model_def.json', model_weights=None):

        from engines.capsule1d import CapsNet, Mask, CapsuleLayer, PrimaryCap, Length

        custom_objects = {
            'CapsNet': CapsNet,
            'CapsuleLayer': CapsuleLayer,
            'PrimaryCap': PrimaryCap,
            'Mask': Mask,
            'Length': Length
        }

        model = self.load_keras_model(model_def, custom_objects=custom_objects)

        if model_weights:
            print('Loading weights "{}"'.format(model_weights))
            model.load_weights(model_weights, by_name=True)
            print('Weights loaded.')

        return model

    def load_keras_model(self, json_path, custom_objects={}):
        """
        function loads keras model def (notincluding weights) from json file
        :param model:
        :param json_path:
        :return:
        """
        from keras.models import model_from_json

        with open(json_path, 'r') as fi:
            json_string = fi.read()

        model = model_from_json(json_string, custom_objects=custom_objects)
        return model

    @property
    def frame_length(self):
        """Number of audio samples per frame expected by the engine."""

        # return max(self.model.input.shape.as_list()[1:])
        return 2**13

    def process(self, pcm):
        """
        Method to check each frame for the keyword
        :param pcm: input audio
        :return: bool
        """
        # pcm = (np.iinfo(np.int16).max * pcm).astype(np.int16).tobytes()
        pcm = np.expand_dims(pcm, axis=-1)
        pcm = np.expand_dims(pcm, axis=0)
        y_dummy = np.array([[1,0]])
        y_pred, x_recon = self.model.predict([pcm, y_dummy])

        detected = y_pred[0][0] > self.sensitivity

        return detected

    def release(self):
        # clear keras session
        from keras import backend as K
        K.clear_session()

        # # force cuda to release memory
        # from numba import cuda
        # try:
        #     cuda.select_device(0)
        #     cuda.close()
        # except:
        #     print("Couldn't release cuda: are you on cpu?")

        return

    def __str__(self):
        return 'KerasCapsule'


class PocketSphinxEngine(Engine):
    """Pocketsphinx engine."""

    def __init__(self, keyword, sensitivity):
        """
        Constructor.

        :param keyword: keyword to be detected.
        :param sensitivity: detection sensitivity.
        """

        # Set the configuration.
        config = Decoder.default_config()
        config.set_string('-logfn', '/dev/null')
        # Set recognition model to US
        config.set_string('-hmm', os.path.join(get_model_path(), 'en-us'))
        config.set_string('-dict', os.path.join(get_model_path(), 'cmudict-en-us.dict'))
        config.set_string('-keyphrase', keyword)
        config.set_float('-kws_threshold', sensitivity)
        self._decoder = Decoder(config)
        self._decoder.start_utt()

    def process(self, pcm):
        pcm = (np.iinfo(np.int16).max * pcm).astype(np.int16).tobytes()
        self._decoder.process_raw(pcm, False, False)

        detected = self._decoder.hyp()
        if detected:
            self._decoder.end_utt()
            self._decoder.start_utt()

        return detected

    def release(self):
        self._decoder.end_utt()

    def __str__(self):
        return 'PocketSphinx'


class PorcupineEngineBase(Engine):
    """Base class for different variants of Porcupine engine."""

    def __init__(self, sensitivity, model_file_path, keyword_file_path):
        """
        Constructor.

        :param sensitivity: detection sensitivity.
        :param model_file_path: path to model file.
        :param keyword_file_path: path to keyword file.
        """

        library_path = os.path.join(
            os.path.dirname(__file__),
            'engines/porcupine/lib/linux/%s/libpv_porcupine.so' % platform.machine())

        self._porcupine = Porcupine(
            library_path=library_path,
            model_file_path=model_file_path,
            keyword_file_path=keyword_file_path,
            sensitivity=sensitivity)

    def process(self, pcm):
        pcm = (np.iinfo(np.int16).max * pcm).astype(np.int16)
        return self._porcupine.process(pcm)

    def release(self):
        self._porcupine.delete()

    def __str__(self):
        raise NotImplementedError()


class PorcupineEngine(PorcupineEngineBase):
    """Original variant of Porcupine."""

    def __init__(self, keyword, sensitivity):
        """
        Constructor.

        :param keyword: keyword to be detected.
        :param sensitivity: detection sensitivity.
        """

        model_file_path = os.path.join(
            os.path.dirname(__file__),
            'engines/porcupine/lib/common/porcupine_params.pv')

        keyword_file_path = os.path.join(
            os.path.dirname(__file__),
            'engines/porcupine/resources/keyword_files/linux/%s_linux.ppn' % keyword.lower())

        super().__init__(sensitivity, model_file_path, keyword_file_path)

    def __str__(self):
        return 'Porcupine'


class PorcupineTinyEngine(PorcupineEngineBase):
    """Tiny variant of Porcupine engine."""

    def __init__(self, keyword, sensitivity):
        """
        Constructor.

        :param keyword: keyword to be detected.
        :param sensitivity: detection sensitivity.
        """

        model_file_path = os.path.join(
            os.path.dirname(__file__),
            'engines/porcupine/lib/common/porcupine_tiny_params.pv')

        keyword_file_path = os.path.join(
            os.path.dirname(__file__),
            'engines/porcupine/resources/keyword_files/linux/%s_linux_tiny.ppn' % keyword.lower())

        super().__init__(sensitivity, model_file_path, keyword_file_path)

    def __str__(self):
        return 'Porcupine Tiny'

#
# class SnowboyEngine(Engine):
#     """Snowboy engine."""
#
#     def __init__(self, keyword, sensitivity):
#         """
#         Constructor.
#
#         :param keyword: keyword to be detected.
#         :param sensitivity: detection sensitivity.
#         """
#
#         keyword = keyword.lower()
#         if keyword == 'alexa':
#             model_relative_path = 'engines/snowboy/resources/alexa/alexa-avs-sample-app/alexa.umdl'
#         else:
#             model_relative_path = 'engines/snowboy/resources/models/%s.umdl' % keyword
#
#         model_str = os.path.join(os.path.dirname(__file__), model_relative_path).encode()
#         resource_filename = os.path.join(os.path.dirname(__file__), 'engines/snowboy/resources/common.res').encode()
#         self._snowboy = snowboydetect.SnowboyDetect(resource_filename=resource_filename, model_str=model_str)
#         self._snowboy.SetSensitivity(str(sensitivity).encode())
#
#     def process(self, pcm):
#         pcm = (np.iinfo(np.int16).max * pcm).astype(np.int16).tobytes()
#         return self._snowboy.RunDetection(pcm) == 1
#
#     def release(self):
#         pass
#
#     def __str__(self):
#         return 'Snowboy'
