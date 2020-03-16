for part in a b c d e f g h
do
	curl -u $1:$2 http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aaca$part -o vox2_dev_aaca$part
done
curl -u $1:$2 http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_aac.zip -o vox2_test_aac.zip
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv
