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

import logging
from scipy.signal import resample, decimate

from dataset import *
from engine import *
from noise_mixer import *


class WakeWordExecutor(object):
    """Executor for running (noisy) speech datasets through wake-word engines and collect their accuracy metrics."""

    def __init__(self, engine_type, sensitivity, keyword, speech_dataset, noise_dataset=None, kwargs={}):
        """
        Constructor.

        :param engine_type: type of the wake-word engine.
        :param sensitivity: wake-word engine's detection sensitivity.
        :param keyword: keyword to be detected.
        :param speech_dataset: speech dataset containing both background and keyword utterances.
        :param noise_dataset: dataset containing noise samples.
        """

        self._sensitivity = sensitivity
        self._speech_dataset = speech_dataset
        self._num_keywords =\
            sum([speech_dataset.get_metadata(i).contains_keyword for i in range(speech_dataset.size())])

        if noise_dataset is not None:
            self._noise_mixer = NoiseMixer(noise_dataset)
        else:
            self._noise_mixer = None

        self._engine = Engine.create(engine_type, keyword=keyword, sensitivity=sensitivity, kwargs=kwargs)

    def execute(self):
        """
        Runs the engine on the (noisy) speech dataset.

        :return: tuple of false alarm per hour and miss detection rate.
        """

        num_false_alarms = 0
        num_caught=0
        num_misses = 0
        total_duration_sec = 0

        for index in range(self._speech_dataset.size()):
            data = self._speech_dataset.get_data(index)

            pcm = data.pcm
            duration_sec = pcm.size / Dataset.SAMPLE_RATE
            total_duration_sec += duration_sec

            if self._noise_mixer:
                pcm = self._noise_mixer.mix(pcm)

            if self._engine.requires_resample:
                # first decimate to filter before applying resample (avoids aliasing)
                pcm = decimate(pcm, int(np.round(Dataset.SAMPLE_RATE/self._engine.target_sample_rate)))
                # then make dimensions precise with resample
                pcm = resample(pcm, int(duration_sec * self._engine.target_sample_rate))

            frame_length = self._engine.frame_length
            num_frames = len(pcm) // frame_length
            num_detected = 0
            for i in range(num_frames):
                frame = pcm[i * frame_length:(i + 1) * frame_length]
                if self._engine.process(frame):
                    num_detected += 1

            if data.metadata.contains_keyword:
                num_caught += num_detected
                if num_detected == 0:
                    num_misses += 1
            else:
                num_false_alarms += num_detected

        total_duration_hour = total_duration_sec / 3600
        false_alarm_per_hour = num_false_alarms / total_duration_hour
        miss_rate = num_misses / self._num_keywords

        logging.info('%s (%s):', str(self._engine), self._sensitivity)
        logging.info('false alarms per hour: %f (%d / %f)', false_alarm_per_hour, num_false_alarms, total_duration_hour)
        logging.info('miss detection rate: %f (%d / %d)', miss_rate, num_misses, self._num_keywords)

        return dict(
            sensitivity=self._sensitivity,
            total_duration_hour=total_duration_hour,
            num_false_alarms=num_false_alarms,
            num_caught=num_caught,
            num_misses=num_misses,
            miss_rate=miss_rate,
            false_alarm_per_hour=false_alarm_per_hour,
            num_keywords=self._num_keywords
        )


    def release(self):
        """Releases the resources acquired by the engine."""

        self._engine.release()
