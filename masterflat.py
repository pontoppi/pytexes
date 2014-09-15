import os
import astropy.io.fits as pf
import numpy as np

def masterflat(path):
    files = os.listdir(path)
    fitsfiles = [file for file in files if '.fits' in file]
    flats = [file for file in fitsfiles if 'flat' in file]
    flatDataList = []
    for flat in flats:
        flatCube = pf.getdata(path+flat)
        flatPlane = flatCube[0,:,:]+flatCube[1,:,:]-flatCube[2,:,:]-flatCube[3,:,:]
        flatPlane = flatPlane/np.median(flatPlane)
        flatDataList.append(flatPlane)
    
    hdu = pf.PrimaryHDU(flatDataList[0]/flatDataList[1])
    hdu.writeto('flat.fits')
    
path = '/astro/pontoppi/DATA/TEXES/NH3_rawdata/Aug10/'
masterflat(path)