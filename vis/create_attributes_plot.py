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

font = {'family' : 'serif',
        'size'   : 8}

matplotlib.rc('font', **font)

def main():
    width = 3.25 * 2
    height = width / 2.2 #1.618

    attributes = ['Smiling', 'Male', 'Wearing_Lipstick', 'Young']
    fig, axarr = plt.subplots(2, 5)
    idx = int(sys.argv[1])
    for i, attr in enumerate(attributes):
        artifacts_dir = 'artifacts/attributes_experiment/'
        artifacts_dir = os.path.join(artifacts_dir, attr + '_eps_0.005/0')
        images = np.load(os.path.join(artifacts_dir, 'images.npy'))
        images_0 = np.load(os.path.join(artifacts_dir, 'images_0.npy'))
        images_1 = np.load(os.path.join(artifacts_dir, 'images_1.npy'))

        axarr[0, 0].imshow(images[idx].transpose(1,2,0))
        axarr[0, 0].set_title("Input")
        if attr == 'Wearing_Lipstick':
            axarr[0, i+1].set_title('Lipstick')
        else:
            axarr[0, i+1].set_title(attr)
        #axarr[0, i].axis('off')
        axarr[0, i+1].imshow(images_0[idx].transpose(1,2,0))
        #axarr[1, i].axis('off')
        axarr[1, i+1].imshow(images_1[idx].transpose(1,2,0))
        #axarr[2, i].axis('off')

    #axarr[0,0].set_ylabel('Original')
    #axarr[1,0].set_ylabel('0')
    #axarr[2,0].set_ylabel('1')

    for ax in axarr.flat:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

    axarr[1,0].axis('off')

    #plt.tight_layout()
    fig.set_size_inches(width, height)
    fig.savefig('test_fig.pdf')
    #plt.show()

if __name__ == '__main__':
    main()
