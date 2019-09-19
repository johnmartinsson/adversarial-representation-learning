#python3 get_drive_file.py 0B7EVK8r0v71pZjFTYXZWM3FlRnM imgs_aligned.zip
#python3 get_drive_file.py 0B7EVK8r0v71pblRyaVFSWGxPY0U annotations.txt
#python3 get_drive_file.py 0B7EVK8r0v71pY0NSMzRuSXJEVkk data_split.txt
#unzip imgs_aligned.zip
rm imgs_aligned.zip
mv img_align_celeba imgs
pyrhon3 preprocess_annotations.py
