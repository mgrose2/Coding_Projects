# image_segmentation.py
"""Volume 1: Image Segmentation.
<Mark Rose>
<Section 2>
<11/12/18>
"""

import numpy as np
from scipy import linalg as la
import scipy.sparse.linalg as lo
import scipy.sparse as sp
import scipy
from imageio import imread
import matplotlib.pyplot as plt


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    D = np.diag(np.sum(A, axis=1))                                                      #Get the D matrix by summing across the rows of A
    L = D - A                                                                           #Return L where L = D - A
    return(L)
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    A = laplacian(A)
    eigs = np.real(la.eigvals(A))                                                       #Make an array of the real eigenvalues of the Laplace of A
    zero_count = 0
    for i in range(len(eigs)):
        if eigs[i] < tol:
            zero_count+=1                                                               #Count all the zeros in the eigenvalues
    eigs = np.sort(eigs)
    if eigs[1] < tol:
        return(zero_count, 0)
    else:
        return(zero_count, eigs[1])                                                     #Sort all the eigenvalues and return the zero count along with the second smallest value
    raise NotImplementedError("Problem 2 Incomplete")                                   


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        image = imread(filename)
        self.scaled = image/255
        if len(self.scaled.shape) == 3:                                                                         #Read in the image, and divide by 255 to put it to scale.
            self.brightness = self.scaled.mean(axis=2)                                                          #If the scaled image has color, take the mean to put it into a gray scale and save it as an attribute.
        else:
            self.brightness = self.scaled
        self.m, self.n = np.shape(self.brightness)                                                              #Save the x, y dimensions of the image
        self.brightness = np.ravel(self.brightness)                                                             #Save the brightness as a 1-d array of the scaled image
        return
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 3
    def show_original(self):
        """Display the original image."""
        if len(self.scaled.shape) == 3:
            plt.imshow(self.scaled)                                                                             #If there is an RGB color scheme, use plt.imshow. Otherwise, include the argument cmap='gray' to havea grayscale image
        else:
            plt.imshow(self.scaled, cmap='gray')
        plt.axis('off')                                                                                         #Turn of the axis to just show the image.
        plt.show()
        return
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        n = len(self.brightness)
        A = sp.lil_matrix((n,n))                                                                                #Create a Matrix A where we will alter all the values based on the weights
        B = np.zeros((n,n)) 
        for i in range(n):
            neighbors_1, distances_1 = get_neighbors(i, r, self.m, self.n)                                      #Used the predifined function to find the indices that need weighted values, and to find the euclidean distance for the weighted part of the algorithm
            bright_vec = -1*np.abs(self.brightness[i]-self.brightness[neighbors_1])                             #Find the negative ndifference between the brightness of the current pixel and its neighbors
            weights = np.exp(bright_vec/sigma_B2-distances_1/sigma_X2)
            A[i, neighbors_1] = weights                                                                         #Set each row of values to the previously found weights
            B[i, neighbors_1] = weights
        D = np.diag(np.sum(B, axis=1))
        A = sp.csc_matrix(A)
        return(A, D)
            
        raise NotImplementedError("Problem 4 Incomplete")

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        L = sp.csgraph.laplacian(A)                                                                             #Find L based on a matrix A
        D = np.sum(D, axis=1)                                                                                   
        D = D**(-1/2)
        D_o = sp.diags(D)                                                                                       #Create a diagonal matrix by taking the degree matrix and raising it to a power of -1/2
        mult = D_o @ L @ D_o
        eig_val, eig_vec = lo.eigsh(mult, which = 'SM', k = 2)                                                  #Find the eigenvectors corresponding to the smallest eigenvalues
        my_vec = eig_vec[:,1]                                                                                   #Take the second vector, as the first oe corresponds to 0
        my_vec = my_vec.reshape(self.m, self.n)
        mask = my_vec > 0                                                                                       #Return a mask where each value in the vector is greater than 0
        return(mask)
        raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency(r, sigma_B, sigma_X)                                                              #Use the previously defined function to find A and D
        mask = self.cut(A,D)                                                                                    #Use the previously defined function to find the mask
        mask_neg = ~mask                                                                                        #Make a mask for negative values
        fig, axes= plt.subplots(1,3)
        if len(np.shape(self.scaled)) == 3:                                                                     #If the photo is rgb, stack the mask three times and use element-wise multiplication
            axes[0].imshow(self.scaled)
            axes[1].imshow(self.scaled * np.dstack((mask,mask,mask)))
            axes[2].imshow(self.scaled * np.dstack((mask_neg, mask_neg, mask_neg)))
        else:
            axes[0].imshow(self.scaled, cmap = 'gray')                                                          #If the photo is grayscale, plot the mask times the photo with element wise multiplication
            axes[1].imshow(self.scaled * mask, cmap = 'gray')
            axes[2].imshow(self.scaled * mask_neg, cmap = 'gray')
        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')
        plt.show()
        return
        raise NotImplementedError("Problem 6 Incomplete")


#if __name__ == '__main__':
 #    ImageSegmenter("dream_gray.png").segment()
 #    ImageSegmenter("dream.png").segment()
 #    ImageSegmenter("monument_gray.png").segment()
 #    ImageSegmenter("monument.png").segment()
