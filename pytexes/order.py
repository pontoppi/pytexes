import warnings
import json
import os
import ConfigParser as cp

import numpy as np
import numpy.ma as ma
import astropy.io.fits as pf
import scipy.fftpack as fp
from scipy.stats import tmean, tvar
from scipy.ndimage.filters import median_filter
from scipy import constants
import matplotlib.pylab as plt
import inpaint as inpaint
import utils.helpers as helpers

class Order():
    def __init__(self,Nod,onum=1,trace=None,write_path=None):
        self.type = 'order'
        self.headers = Nod.headers
        self.setting = Nod.setting
        self.echelle = Nod.echelle
        self.crossdisp = Nod.crossdisp
        self.airmass = Nod.airmass
        
        self.Envi    = Nod.Envi
        self.onum    = onum

        self.image = Nod.image
        self.uimage = Nod.uimage
        self.sh = self.image.shape
        xrs,trace = self.fitTrace()

        self.xrange = self.Envi.getXRange(self.setting,onum)
        self.image = Nod.image[:,self.xrange[0]:self.xrange[1]]
        self.uimage = Nod.uimage[:,self.xrange[0]:self.xrange[1]]
        self.sky = Nod.sky[:,self.xrange[0]:self.xrange[1]]
        self.usky = Nod.usky[:,self.xrange[0]:self.xrange[1]]
        self.sh = self.image.shape
        
        self._cullEdges(trace)
        import pdb;pdb.set_trace()
        self._subMedian()

        if trace is None:
            yrs,trace = self.fitTrace()

        self.image_rect,self.uimage_rect = self.yRectify(self.image,self.uimage,yrs,trace)
        self.sky_rect,self.usky_rect = self.yRectify(self.sky,self.usky,yrs,trace)

        if write_path:
            self.file = self.writeImage(path=write_path)

    def _cullEdges(self,trace):
        orderw = self.Envi.getOrderWidth(self.setting)
        det_pars = self.Envi.getDetPars()
        xindex = 
        for i in np.arange(det_pars['ny']):
            np.where()

    def fitTrace(self,kwidth=10):
        sh = self.sh
        xr1 = (0,sh[0]-1)
        xrs = [xr1]

        polys = []
        for xr in xrs:
            xindex = np.arange(xr[0],xr[1])
            kernel = np.median(self.image[sh[1]/2-kwidth:sh[1]/2+kwidth,xindex],0)
           
            
            centroids = []
            totals = []
            for i in np.arange(sh[0]):
                col_med = np.median(self.image[i,xindex])
                total = np.abs((self.image[i,xindex]-col_med).sum())
                cc = fp.ifft(fp.fft(kernel)*np.conj(fp.fft(self.image[i,xindex]-col_med)))
                cc_sh = fp.fftshift(cc)
                centroid = helpers.calc_centroid(cc_sh,cwidth=30).real - xindex.shape[0]/2.
                centroids.append(centroid)
                totals.append(total)

            centroids = np.array(centroids)
        
            yindex = np.arange(sh[0])
            gsubs = np.where((np.isnan(centroids)==False))

            centroids[gsubs] = median_filter(centroids[gsubs],size=20)
            coeffs = np.polyfit(yindex[gsubs],centroids[gsubs],3)

            poly = np.poly1d(coeffs)
            polys.append(poly)
            
        return xrs,polys

    def yRectify(self,image,uimage,yrs,traces):
        
        sh = self.sh
        image_rect = np.zeros(sh)
        uimage_rect = np.zeros(sh)
        
        for yr,trace in zip(yrs,traces):
            index = np.arange(yr[0],yr[1])
            for i in np.arange(sh[1]):
                col = np.interp(index-trace(i),index,image[index,i])
                image_rect[index,i] = col
                col = np.interp(index-trace(i),index,uimage[index,i])
                uimage_rect[index,i] = col

        return image_rect,uimage_rect
                
    def _subMedian(self):
        self.image = self.image-np.median(self.image,axis=0)
            
    def writeImage(self,filename=None,path='.'):

        if filename is None:
            time   = self.header['UTC']
            time   = time.replace(':','')
            time   = time[0:4]
            date   = self.header['DATE-OBS']
            date   = date.replace('-','')
            object = self.header['OBJECT']
            object = object.replace(' ','')
            filename = path+'/'+object+'_'+date+'_'+time+'_order'+str(self.onum)+'.fits'

            
        hdu  = pf.PrimaryHDU(self.image_rect)
        uhdu = pf.ImageHDU(self.uimage_rect)
        sky_hdu = pf.ImageHDU(self.sky_rect)
        usky_hdu = pf.ImageHDU(self.usky_rect)
        
        hdu.header['SETNAME'] = (self.setting, 'Setting name')
        hdu.header['ECHLPOS'] = (self.echelle, 'Echelle position')
        hdu.header['DISPPOS'] = (self.crossdisp, 'Cross disperser position')
        hdu.header['ORDER'] = (str(self.onum),'Order number')

        hdulist = pf.HDUList([hdu,uhdu,sky_hdu,usky_hdu])

        hdulist.writeto(filename,clobber=True)

        return filename
