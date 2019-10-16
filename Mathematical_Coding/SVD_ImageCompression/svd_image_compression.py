# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""
import numpy as np
import scipy.linalg as la
from imageio import imread
import matplotlib.pyplot as plt

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    B = A.conj().T.dot(A)
    eig_val, eig_vec = la.eig(B)                                                            #Calculate Eigenvalues and eigenvectors of A.T.dot(A)
    eig_vec = eig_vec.T
    sig = eig_val**(1/2)                                                                    
    sort_index = np.argsort(sig)                                                            #Sort the values correctly from least to greatest
    sort_index = sort_index[::-1]
    sig = sig[sort_index]
    eig_vec = eig_vec[sort_index]
    r = np.sum(sig > tol)                                                                   #Check to see if some of the singluar values are close to zero
    if len(sig)-r != 0:
        sig = sig[:r]
        eig_vec = eig_vec[:r]
    U = A.dot(eig_vec.T) /sig                                                               #Calculate U
    return(U, sig, eig_vec)                                                                 #Return U, E, and V.T
    
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    E = np.array([[1,0,0],[0,0,1]])                                                         #Initialize the E matrix
    theta = np.linspace(0,2*np.pi, 200)
    x = np.cos(theta)
    y = np.sin(theta)
    S = np.vstack((x,y))
    
    plt.subplot(221)                                                                       #Create the various plots for the function
    plt.plot(x, y, 'b')
    plt.plot(E[0,:], E[1,:], 'g')
    plt.axis("equal")
    
    U, sigma, V_H = la.svd(A)                                                               #Calculate the svd using scipy's function
    
    V_HS = np.dot(V_H, S)
    V_HE = np.dot(V_H, E)
    plt.subplot(222)
    plt.plot(V_HS[0,:], V_HS[1,:], 'b')                                                      #Plot with different colors. 
    plt.plot(V_HE[0,:], V_HE[1,:], 'g')
    plt.axis("equal")

    plt.subplot(223)
    plt.plot(sigma[0]*V_HS[0,:], sigma[1]*V_HS[1,:], 'b')
    plt.plot(sigma[0]*V_HE[0,:], sigma[1]*V_HE[1,:], 'g')
    plt.axis("equal")
    
    sV_HS = np.vstack((sigma[0]*V_HS[0,:], sigma[1]*V_HS[1,:]))
    sV_HE = np.vstack((sigma[0]*V_HE[0,:], sigma[1]*V_HE[1,:]))                             #Stack the various vectors together
    
    UsV_HS = np.dot(U, sV_HS)
    UsV_HE = np.dot(U, sV_HE)
    plt.subplot(224)                                                                        #Plot with different colors.
    plt.plot(UsV_HS[0,:], UsV_HS[1,:], 'b')
    plt.plot(UsV_HE[0,:], UsV_HE[1,:], 'g')
    
    plt.show()
    return
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    U, E, V = compact_svd(A)                                                                        #Get the compact svd using the function previously made
    if s > np.size(E):                                                                              #Check to make sure that S is greater than the amount of singuar values
        raise ValueError("S is larger than rank(A)")
    U = U[:,0:s]                                                                                    #Index into each Matrix to create the truncated form.
    E = E[0:s]
    V = V[0:s]
    values = U.size + E.size + V.size
    return(U.dot(np.diag(E).dot(V)), values)                                                        #Return the new matrix, and the number of entries needed to store the truncated form.
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    U, E, V = compact_svd(A)                                                                     #Calculate the SVD using the previously defined compact svd function
    if (err <= E[len(E)-1]):                                                                    #Make sure that the error is greater than the size of the last singular value.
        raise ValueError('A cannot be approximated within the tolerance by a matrix of lesser rank.')
    s = np.argmax(np.where(E < err))                                                            #Find the S that gives you the lowest rank approximation
    return(svd_approx(A, s))                                                                    #Return the new truncated version of A and the previous S found

    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    fig, axes= plt.subplots(1,2)                                                            #Create two subplots and read in the image
    image = imread(filename)/255
    if len(np.shape(image)) == 2:
        new_image, values = svd_approx(image, s)                                            #Use the previously defined function to get the new A and amount of values neccessary to store A
        new_image = np.real(new_image)                                                      #Take only the real values of A
        axes[0].imshow(image, cmap='gray')                                                  #Show the original image and the approximated image.                      
        axes[1].imshow(new_image, cmap='gray')
    else:
        red_layer = image[:,:,0]                                                            #Split the color image into three different layers, red green and blue
        blue_layer = image[:,:,1]
        green_layer = image[:,:,2]
        red_layer, red_val = svd_approx(red_layer, s)                                       #Calculate the truncated version of each layer and the amount of values necessary to store it
        red_layer = np.real(red_layer)                                                      #Take only the real values of each layer
        red_layer = np.clip(red_layer, 0, 1)                                                #Clip any values that are less than zero and greater than one
        blue_layer, blue_val = svd_approx(blue_layer, s)
        blue_layer = np.real(blue_layer)
        blue_layer = np.clip(blue_layer, 0, 1)
        green_layer, green_val = svd_approx(green_layer, s)
        green_layer = np.real(green_layer)
        green_layer = np.clip(green_layer, 0, 1)
        axes[0].imshow(image)                                                               #Show the original image
        axes[1].imshow(np.dstack((red_layer, blue_layer, green_layer)))                     #Show the new image by stacking the three layers on top of one another
        values = red_val+blue_val+green_val                                                 #Calculate the values needed to store all three layers (the new image)
    axes[0].axis("off") 
    axes[1].axis("off")
    plt.suptitle('Difference in Entries Stored: ' + str(image.size-values))                 #Create a title saying the amount of entries saved by using the approximation
    plt.show()
    return
    raise NotImplementedError("Problem 5 Incomplete")
