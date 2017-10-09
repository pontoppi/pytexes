import warnings
import json
import os
import configparser as cp

import numpy as np
import numpy.ma as ma
import astropy.io.fits as pf
import scipy.fftpack as fp
from scipy.stats import tmean, tvar
from scipy.ndimage.filters import median_filter
from scipy import constants
from scipy import interpolate as ip
import matplotlib.pylab as plt
import pytexes.inpaint as inpaint
import utils.helpers as helpers

class Order():
    def __init__(self,Nod,onum=1,write_path=None):
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
        xrs,traces = self.fitTrace(porder=1,cwidth=3.)
        trace = traces[0]

        self.xrange = self.Envi.getXRange(self.setting,onum)
        self.image = Nod.image[:,self.xrange[0]:self.xrange[1]]
        self.uimage = Nod.uimage[:,self.xrange[0]:self.xrange[1]]
        self.sky = Nod.sky[:,self.xrange[0]:self.xrange[1]]
        self.usky = Nod.usky[:,self.xrange[0]:self.xrange[1]]
        self.sh = self.image.shape
        
        xrs,traces = self.fitTrace(cwidth=3.,porder=5,pad=False)
        self.image_rect,self.uimage_rect = self.xRectify(self.image,self.uimage,xrs,traces)
#        self.image_rect = np.transpose(np.transpose(self.image_rect)-np.median(self.image_rect[:,10:30],axis=1))
                
        self.sky_rect,self.usky_rect = self.xRectify(self.sky,self.usky,xrs,traces)
        self._cullEdges()


        if write_path:
            self.file = self.writeImage(path=write_path)

#    def _cullEdges(self,trace):
#        orderw = self.Envi.getOrderWidth(self.setting)
#        center = self.Envi.getSpatialCenter(self.setting,self.onum)
#        xindex = np.arange(self.sh[1])
#        for i in np.arange(self.sh[0]):
#            bsubs = np.where(xindex<-trace(i))
#            self.image[i,bsubs] = 0.
#            self.sky[i,bsubs] = 0.
#            bsubs = np.where(xindex>-trace(i)+1.5*orderw)
#            self.image[i,bsubs] = 0.
#            self.sky[i,bsubs] = 0.
    def _cullEdges(self):
        orderw = self.Envi.getOrderWidth(self.setting)
        fullw = self.sh[1]
        self.image_rect[:,:int((fullw-orderw)/2)] = 0.
        self.image_rect[:,-int((fullw-orderw)/2):] = 0.        
        self.sky_rect[:,:int((fullw-orderw)/2)] = 0.
        self.sky_rect[:,-int((fullw-orderw)/2):] = 0.        
            
    def fitTrace(self,kwidth=10,porder=3,cwidth=30,pad=False):
        sh = self.sh
        xr1 = (0,sh[1])
        xrs = [xr1]

        polys = []
        for xr in xrs:
            xindex = np.arange(xr[0],xr[1])
            kernel = np.median(self.image[int(sh[0]/2-kwidth):int(sh[0]/2+kwidth),xindex],0)
                
            centroids = []
            totals = []
            for i in np.arange(sh[0]):
                row = self.image[i,xindex]
                row_med = np.median(row)
                    
                total = np.abs((row-row_med).sum())
                cc = fp.ifft(fp.fft(kernel)*np.conj(fp.fft(row-row_med)))
                cc_sh = fp.fftshift(cc)
                centroid = helpers.calc_centroid(cc_sh,cwidth=cwidth).real - xindex.shape[0]/2.
                centroids.append(centroid)
                totals.append(total)

            centroids = np.array(centroids)
        
            yindex = np.arange(sh[0])
            gsubs = np.where((np.isnan(centroids)==False))

            centroids[gsubs] = median_filter(centroids[gsubs],size=20)
            coeffs = np.polyfit(yindex[gsubs],centroids[gsubs],porder)

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
                col = ip.interp1d(index,image[index,i],bounds_error=False,fill_value=0)
                image_rect[index,i] = col(index-trace(i))
                col = ip.interp1d(index,uimage[index,i],bounds_error=False,fill_value=1e10)
                uimage_rect[index,i] = col(index-trace(i))

        return image_rect,uimage_rect

    def xRectify(self,image,uimage,xrs,traces):
        
        sh = self.sh
        image_rect = np.zeros(sh)
        uimage_rect = np.zeros(sh)
        
        for xr,trace in zip(xrs,traces):
            index = np.arange(xr[0],xr[1])
            for i in np.arange(sh[0]):
                row = ip.interp1d(index,image[i,index],bounds_error=False,fill_value=0)
                image_rect[i,index] = row(index-trace(i))
                row = ip.interp1d(index,uimage[i,index],bounds_error=False,fill_value=1e10)
                uimage_rect[i,index] = row(index-trace(i))

        return image_rect,uimage_rect
 
                
    def _subMedian(self):
        self.image = self.image-np.median(self.image,axis=0)
            
    def writeImage(self,filename=None,path='.'):
        header = self.headers[0]
        if filename is None:
            time   = header['TIME']
            time   = time.replace(':','')
            time   = time[0:4]
            date   = header['DATE-OBS']
            date   = date.replace('-','')
            object = header['OBJECT']
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
