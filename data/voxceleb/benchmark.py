import warnings
import librosa
import torch
import torchaudio
import time
import glob
import tqdm

def main():
    wav_files = glob.glob('wav/*.wav')
    m4a_files = glob.glob('wav/*.m4a')
    
    print("wav_files: ", len(wav_files))
    print("m4a_files: ", len(m4a_files))

    print("-----------------------------")
    print("- torchaudio")
    print("-----------------------------")
    t1 = time.time()
    dc = torch_load_files(wav_files)
    t2 = time.time()
    print("torch: ", t2-t1)
    print("dummy count: ", dc)
    print("")

    print("-----------------------------")
    print("- librosa (wav)")
    print("-----------------------------")
    t1 = time.time()
    dc = librosa_load_files(wav_files)
    t2 = time.time()
    print("librosa (wav): ", t2-t1)
    print("dummy count: ", dc)
    print("")

    # SLOW!
    #print("-----------------------------")
    #print("- librosa (m4a)")
    #print("-----------------------------")
    #t1 = time.time()
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    dc = librosa_load_files(m4a_files)
    #t2 = time.time()
    #print("librosa (m4a): ", t2-t1)
    #print("dummy count: ", dc)

def torch_load_files(files):
    dummy_count = torch.tensor(0.0)
    for f in files:
        x, sr = torchaudio.load(f)
        #print(f)
        #print(sr)
        #print(x.shape)
        dummy_count += x.shape[1]

    return dummy_count

def librosa_load_files(files):
    dummy_count = torch.tensor(0.0)
    for f in files:
        x, sr = librosa.load(f, sr=None)
        x = torch.tensor(x)
        #print(f)
        #print(sr)
        #print(x.shape)
        dummy_count += x.shape[0]

    return dummy_count

if __name__ == '__main__':
    main()
