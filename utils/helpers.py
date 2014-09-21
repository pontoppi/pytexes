import numpy as np
import astropy.io.fits as pf

def calc_centroid(cc,cwidth=15):
    maxind = np.argmax(cc)
    mini = max([0,maxind-cwidth])
    maxi = min([maxind+cwidth,cc.shape[0]])
    trunc = cc[mini:maxi]
    centroid = mini+(trunc*np.arange(trunc.shape[0])).sum()/trunc.sum()
    return centroid

def write_fits(array,filename='test.fits'):
    hdu = pf.PrimaryHDU(array)
    hdu.writeto(filename,clobber=True)

def getBaseName(header):
    time   = header['TIME']
    time   = time.replace(':','')
    time   = time[0:4]
    date   = header['DATE-OBS']
    date   = date.replace('-','')
    object = header['OBJECT']
    object = object.replace(' ','')
    basename = object+'_'+date+'_'+time
    return basename
