import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
#plt.rc('xtick', labelsize=8)
#plt.rc('ytick', labelsize=8)
#plt.rc('title', labelsize=8)
#plt.rc('axes', labelsize=8)
#plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
#plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
font = {'family' : 'serif',
        'size'   : 10}

matplotlib.rc('font', **font)

def main():
    idx = int(sys.argv[1])
    width = 3.25 * 2
    height = width / 1.618

    fig, axarr = plt.subplots(2, 3)
    image_dir = 'stacked_images'
    images = np.load(os.path.join(image_dir, 'images.npy'))
    images_0_0 = np.load(os.path.join(image_dir, 'images_0_0.npy'))
    images_0_1 = np.load(os.path.join(image_dir, 'images_0_1.npy'))
    images_1_1 = np.load(os.path.join(image_dir, 'images_1_1.npy'))
    images_1_0 = np.load(os.path.join(image_dir, 'images_1_0.npy'))

    axarr[0, 0].imshow(images_0_0[idx].transpose(1,2,0))
    axarr[0, 1].imshow(images_0_1[idx].transpose(1,2,0))
    axarr[1, 1].imshow(images_1_1[idx].transpose(1,2,0))
    axarr[1, 0].imshow(images_1_0[idx].transpose(1,2,0))
    axarr[0, 2].imshow(images[idx].transpose(1,2,0))

    axarr[0, 0].set_ylabel('not male')
    axarr[1, 0].set_ylabel('male')
    axarr[0, 0].set_title('not smiling')
    axarr[0, 1].set_title('smiling')
    axarr[0, 2].set_axis_off()
    axarr[1, 2].set_axis_off()

    for ax in axarr.flat:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

    #fig.set_size_inches(width, height)
    #fig.savefig('test_stacked_fig.pdf')
    plt.show()

if __name__ == '__main__':
    main()
