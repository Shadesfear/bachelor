from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, pyplot as plt
import matplotlib.cm as cm
import argparse
import os
import numpy as np


datafile = "../data/recon32.bin"



class AnimatedGif:
    def __init__(self, size=(800, 600)):
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []

    def add(self, image, label=''):

        I = np.dstack([image, image, image])
        x = 15
        y = 15
        # I[x, y, :] = [1, 0, 0]

        plt_im = plt.imshow(image,cmap=cm.Greys_r, interpolation='nearest' , animated=True)
        plt.show()

        plt_txt = plt.text(10, 310, label, color='red')
        self.images.append([plt_im, plt_txt])

    def save(self, filename, fps):
        ani = animation.ArtistAnimation(self.fig, self.images)
        ani.save(filename, writer='imagemagick', fps=fps)




data = np.fromfile(datafile, dtype=np.float32)
size = np.cbrt(data.size)
assert size**3 == data.size
size = size.astype(int)
data = data.reshape([size.astype(int)] * 3)

# data[15, 15, 15] = 0

video = AnimatedGif()
# for frame_id in range(size):
    # video.add(data[frame_id])
video.add(data[15])
video.save("test.gif", 10)
