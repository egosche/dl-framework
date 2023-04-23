import numpy as np
import matplotlib.pyplot as plt


class Checker:
    # https://stackoverflow.com/questions/32704485/drawing-a-checkerboard-in-python
    output = None

    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        """
        Draws the checkerboard.

        :return: the checkerboard
        :rtype: numpy.ndarray
        :raises ValueError: if the resolution is not evenly dividable by 2 times tile_size
        """
        if self.resolution % (2 * self.tile_size) != 0:
            raise ValueError("Resolution must be evenly dividable by 2 times tile_size")
        tile = np.concatenate((np.zeros(self.tile_size), np.ones(self.tile_size)))
        row = np.pad(tile, int((self.resolution ** 2) / 2 - self.tile_size), 'wrap').reshape((self.resolution,
                                                                                              self.resolution))
        row_overlap = (row + row.T)
        self.output = np.where(row_overlap == 1, 1, 0)
        return self.output.copy()

    def show(self):
        """
        Shows the checkerboard.
        """
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


class Circle:
    # https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    output = None

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        """
        Draws a circle.
        :return: numpy.ndarray
        """
        Y, X = np.ogrid[:self.resolution, :self.resolution]
        dist_from_center = np.sqrt((X - self.position[0]) ** 2 + (Y - self.position[1]) ** 2)
        mask = dist_from_center <= self.radius

        self.output = mask
        return self.output.copy()

    def show(self):
        """
        Shows the circle.
        """
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


class Spectrum:
    output = None

    def __init__(self, resolution):
        self.resolution = resolution

    def draw(self):
        x = np.zeros((self.resolution, self.resolution, 3))

        x[:, :, 0] = np.linspace(0, 1, self.resolution)  #No Red to Full Red

        x[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)  #No Green to Full Green

        x[:, :, 2] = np.linspace(1, 0, self.resolution)  #Full Blue to No Blue
        self.output = x
        return self.output.copy()

    def show(self):
        """
        Shows the spectrum.
        """
        plt.imshow(self.draw())
        plt.show()

