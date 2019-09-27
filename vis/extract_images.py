import os
import argparse
import tensorflow as tf
import glob

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
import scipy.misc
import tensorflow as tf
count = 0
def save_images_from_event(fn, tag, output_dir):
    assert(os.path.isdir(output_dir))
    global count

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/{}_{:05d}.png'.format(output_dir,
                        tag.split('/')[1], count))
                    print("Saving '{}'".format(output_fn))
                    scipy.misc.imsave(output_fn, im)
                    count += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--experiment_path", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    args = parser.parse_args()

    images = []

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    event_files = glob.glob(os.path.join(args.experiment_path, 'events.out.tfevents.*'))
    event_files = sorted(event_files)
    global count
    count = 0
    for event_file in event_files:
        save_images_from_event(event_file, tag='valid/real_images', output_dir=args.output_dir)
    count = 0
    for event_file in event_files:
        save_images_from_event(event_file, tag='valid/fake_images', output_dir=args.output_dir)
    count = 0
    for event_file in event_files:
        save_images_from_event(event_file, tag='valid/weights', output_dir=args.output_dir)
if __name__ == '__main__':
    main()
