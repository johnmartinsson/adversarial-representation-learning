import os
import shutil
import sys
import librosa

def create_basename(filepath):
    basename = '_'.join(filepath.split('/')[-3:])
    return basename

def main():
    in_dir = sys.argv[1]
    files = librosa.util.find_files(in_dir)

    out_dir = sys.argv[2]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for f in files:
        shutil.move(f, os.path.join(out_dir, create_basename(f)))

if __name__ == '__main__':
    main()
