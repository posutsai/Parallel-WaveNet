#!/usr/local/bin/python3

'''
This is the version of using Tensorflow Dataset, instead of Tensorflow Queue, which is referred to tensorflow-wavenet repo.
'''

import sys
import os
import tensorflow as tf
import abc # use abc module to implement abstract class attribute
import librosa
import fnmatch
import re
import numpy as np

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

class BaseAudioReader(metaclass=abc.ABCMeta):
    '''
    Generic audio reader that preprocesses audio files and transforms to usable Tensorflow Dataset.
    p.s. Tensorflow is only for greater than 1.3.0 version
    '''

    def __init__(self,
                 audio_dir,
                 sample_rate,
                 gc_enabled,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None):
    # !!! todo list !!!
    # 1. deal with global condition enabled

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled

        filenames = find_files(audio_dir)
        if not filenames:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

    @abc.abstractmethod
    def load_audio_dataset(self, directory, sample_rate):
        '''
        return a Tensorflow Dataset which element is audio tensor
        '''
        return NotImplemented
    @staticmethod
    def calculate_receptive_field(filter_width, dilations, scalar_input,
                                  initial_filter_width):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += filter_width - 1
        return receptive_field

class VCTKAudioReader(BaseAudioReader):

    def __init__(self,
                 audio_dir,
                 sample_rate,
                 gc_enabled,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None):
    # !!! todo list !!!
    # 1. deal with global condition enabled

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'

        filenames = find_files(audio_dir)
        if not filenames:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))


    def load_audio_dataset(self, directory=None, sample_rate=None):

        # set default argument
        if directory is None:
            directory = self.audio_dir
        if sample_rate is None:
            sample_rate = self.sample_rate

        files = find_files(directory)
        id_reg_exp = re.compile(self.FILE_PATTERN)
        print("files length: {}".format(len(files)))
        for filename in files:
            ids = id_reg_exp.findall(filename)
            if not ids:
                # the file name does not match the pattern containing ids, so
                # there is no id.
                category_id = None
            else:
                # the file name matches the pattern for containing ids.
                category_id = int(ids[0][0])
            audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
            audio = audio.reshape(-1, 1)

            if self.silence_threshold is not None:
                # Remove silence
                audio = trim_silence(audio[:, 0], self.silence_threshold)
                audio = audio.reshape(-1, 1)
                if audio.size == 0:
                    print("Warning: {} was ignored as it contains only "
                          "silence. Consider decreasing trim_silence "
                          "threshold, or adjust volume of the audio."
                          .format(filename))
            audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]], 'constant')

            if self.sample_size:
                # Cut samples into pieces of size receptive_field +
                # sample_size with receptive_field overlap
                while len(audio) > self.receptive_field:
                    piece = audio[:(self.receptive_field +
                                    self.sample_size), :]
                    yield piece, category_id
                    audio = audio[self.sample_size:, :]
            else:
                yield audio, category_id

    def sample_process_pipeline(self):
        dataset = tf.data.Dataset.from_generator(self.load_audio_dataset, (tf.float32, tf.int32))
        return dataset

if __name__ == '__main__':
    import json
    SAMPLE_SIZE = 10000
    SILENCE_THRESHOLD = 0.1
    with open('./wavenet_params.json', 'r') as jsonfile:
        params = json.load(jsonfile)
    audio_reader = VCTKAudioReader('./corpus/VCTK-Corpus/wav48/p225',
                                   16000,
                                   False,
                                   BaseAudioReader.calculate_receptive_field(params['filter_width'],
                                                                             params['dilations'],
                                                                             params['scalar_input'],
                                                                             params['initial_filter_width']),
                                   SAMPLE_SIZE,
                                   SILENCE_THRESHOLD)
    dataset = audio_reader.sample_process_pipeline()
    iterator = dataset.make_one_shot_iterator()
    nxt_el = iterator.get_next()
    sess = tf.Session()
    for i in range(5):
        print(sess.run(nxt_el)[0].shape)
