import numpy as np
import warnings
import matplotlib.pylab as pyp
import math
from PIL import Image



resultImage = np.array(Image.open('Noisy-Lena-image-noise-is-AWGN-with-standard-deviation-s-35-which-corresponds-to-an.png').convert('L'))  ## Specify the original file path

im_min, im_max = resultImage.min(), resultImage.max()

print("Original image:", resultImage.shape, resultImage.dtype, im_min, im_max)
resultImage = (resultImage - im_min) / (float)(im_max - im_min)   ## Conversion
print("Perona-Malik Anisotropic Diffusion:", resultImage.shape, resultImage.dtype, resultImage.min(), resultImage.max())

pyp.figure('Original Image')
pyp.imshow(resultImage, cmap='gray')
pyp.axis('on')


pyp.show() #Display image

"""
	****Stopping Functions*****
	Alternate Stopping Function
	def f(lam,b): 
		func = 1/(1 + ((lam/b)**2))
		return func
	
"""
def f(lam,b):
    return np.exp(-1* (np.power(lam,2))/(np.power(b,2)))

def anisodiff(im, steps, b, lam = 0.25):  #takes image input, the number of iterations, 
    

    im_new = np.zeros(im.shape, dtype=im.dtype) 
    for t in range(steps): 
        dn = im[:-2,1:-1] - im[1:-1,1:-1] 
        ds = im[2:,1:-1] - im[1:-1,1:-1] 
        de = im[1:-1,2:] - im[1:-1,1:-1] 
        dw = im[1:-1,:-2] - im[1:-1,1:-1] 
        im_new[1:-1,1:-1] = im[1:-1,1:-1] +\
                            lam * (f(dn,b)*dn + f (ds,b)*ds + 
                                    f (de,b)*de + f (dw,b)*dw) 
        im = im_new 
    return im
  

im2 = anisodiff(resultImage, 60, 0.15, 0.15)
pyp.figure('Perona-Malik anisotropic diffusion')
pyp.imshow(im2, cmap='gray')
pyp.axis('on')


pyp.show() #Display image
