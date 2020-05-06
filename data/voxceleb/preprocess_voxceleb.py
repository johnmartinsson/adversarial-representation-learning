import tqdm
import sys
import numpy as np
import os
import glob

import wave
import librosa

from scipy.io import wavfile
import skimage.filters as filters
from skimage import morphology

def create_basename(filepath):
    basename = filepath.split(os.extsep)[0].split('/')[-1]
    return basename

def write_wave_to_file(filename, rate, wave):
    wavfile.write(filename, rate, wave)

def preprocess_sound_file(filename, output_dir, segment_size_seconds):
    """ Preprocess sound file. Loads sound file from filename and
    splits the signal into equal length segments of size segment size seconds.

    # Arguments
        filename : the sound file to preprocess
        output_dir : the directory to save the extracted signal segments in
        segment_size_seconds : the size of each segment in seconds
    # Returns
        nothing, simply saves the preprocessed sound segments
    """

    # need to explicitly tell librosa NOT to resmaple ...
    wave, samplerate = librosa.load(filename, sr=None)

    if len(wave) == 0:
        raise ValueError("An empty sound file ...")

    basename = create_basename(filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wave_segments = split_into_segments(wave, samplerate, segment_size_seconds)
    save_segments_to_file(output_dir, wave_segments, basename, samplerate)

def save_segments_to_file(output_dir, segments, basename, samplerate):
    # print("save segments ({}) to file".format(str(len(segments))))
    i_segment = 0
    for segment in segments:
        segment_filepath = os.path.join(output_dir, basename + "_seg_" + str(i_segment) + ".wav")
        # print("save segment: {}".format(segment_filepath))
        write_wave_to_file(segment_filepath, samplerate, segment)
        i_segment += 1

def split_into_segments(wave, samplerate, segment_time):
    """ Split a wave into segments of segment_size. Repeat signal to get equal
    length segments.
    """
    # print("split into segments")
    segment_size = samplerate * segment_time
    wave_size = wave.shape[0]

    nb_repeat = segment_size - (wave_size % segment_size)
    nb_tiles = 2
    if wave_size < segment_size:
        nb_tiles = int(np.ceil(segment_size/wave_size))
    repeated_wave = np.tile(wave, nb_tiles)[:wave_size+nb_repeat]
    nb_segments = repeated_wave.shape[0]/segment_size

    if not repeated_wave.shape[0] % segment_size == 0:
        raise ValueError("reapeated wave not even multiple of segment size")

    segments = np.split(repeated_wave, int(nb_segments), axis=0)

    return segments

def main():
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    files = glob.glob(os.path.join(in_dir, '*.wav'))
    for i in tqdm.tqdm(range(len(files))):
        preprocess_sound_file(files[i], out_dir, 2)

if __name__ == '__main__':
    main()
