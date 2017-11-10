import numpy as np
import numpy.ma as ma
import astropy.io.fits as pf
from scipy import constants

import matplotlib.pylab as plt

class Spec1D():
    def __init__(self,Order,sa=False,write_path=None):
        self.Order   = Order
        self.setting = Order.setting
        self.echelle = Order.echelle
        self.crossdisp = Order.crossdisp
        self.airmass = Order.airmass
        self.onum    = Order.onum
        self.headers = Order.headers
        self.Envi    = Order.Envi
        self.sh      = Order.sh[0]
        self.sa      = sa


        PSF = self.getPSF()

        self.disp = self.Envi.getDispersion(self.setting,self.onum)
        self.wave = self.waveGuess(self.disp)
        
        self.flux,self.uflux,self.sky,self.usky = self.extract(PSF)
        
        if sa:
            self.sa_pos,self.usa_pos,self.sa_neg,self.usa_neg = self.SpecAst(PSF)

        if write_path:
            self.file = self.writeSpec(path=write_path)

    def getPSF(self,range=None):
        if range is None:
            range = (0,self.Order.sh[1])
        PSF = np.median(self.Order.image_rect[:,range[0]:range[1]],0)
        npsf = PSF.size
        PSF_norm = PSF/np.abs(PSF).sum()
        return PSF_norm
        
    def extract(self,PSF):
        #Placeholder
        pixsig = 1.
        sh   = self.sh
        npsf = PSF.size

        flux  = np.zeros(sh)
        uflux = np.zeros(sh)
        sky1d  = np.zeros(sh)
        usky1d  = np.zeros(sh)
        
        im = self.Order.image_rect
        uim = self.Order.uimage_rect
        sky = self.Order.sky_rect
        usky = self.Order.usky_rect
        
        for i in np.arange(sh):

            flux[i] = (PSF*im[i,:]/uim[i,:]**2).sum() / (PSF**2/uim[i,:]**2).sum()
            uflux[i] = np.sqrt(1.0/(PSF**2/uim[i,:]**2.).sum())
            
            sky1d[i] = (np.abs(PSF)*sky[i,:]).sum() / (PSF**2).sum()
            usky1d[i] = np.sqrt(1.0/(PSF**2/usky[i,:]**2.).sum())

        flux = ma.masked_invalid(flux)
        flux = ma.filled(flux,1.)
        uflux = ma.masked_invalid(uflux)
        uflux = ma.filled(uflux,1000.)
        sky1d = ma.masked_invalid(sky1d)
        sky1d = ma.filled(sky1d,1.)
        usky1d = ma.masked_invalid(usky1d)
        usky1d = ma.filled(usky1d,1000.)
        sky_cont = self._fitCont(self.wave,sky1d)
        return flux,uflux,sky1d-sky_cont,usky1d
        
    def _fitCont(self,wave,spec):
        bg_temp = 210. #K
        
        niter = 2
        
        cont = self.bb(wave*1e-6,bg_temp)
        gsubs = np.where(np.isfinite(spec))
        for i in range(niter):
            norm = np.median(spec[gsubs])
            norm_cont = np.median(cont[gsubs])
            cont *= norm/norm_cont 
            gsubs = np.where(spec<cont)

        return cont
        
    def bb(self,wave,T):
        cc = constants.c
        hh = constants.h
        kk = constants.k
        
        blambda = 2.*hh*cc**2/(wave**5*(np.exp(hh*cc/(wave*kk*T))-1.))
        
        return blambda
        

    def SpecAst(self,PSF,method='centroid',width=5):
        '''
        The uncertainty on the centroid is:

                 SUM_j([j*SUM_i(F_i)-SUM_i(i*F_i)]^2 * s(F_j)^2)
        s(C)^2 = ------------------------------------------------
                                [SUM_i(F_i)]^4

        
        '''
        
        #Guesstimated placeholder
        aper_corr = 1.4
        posloc = np.argmax(PSF)
        negloc = np.argmin(PSF)

        sa_pos = np.zeros(self.sh)
        sa_neg = np.zeros(self.sh)

        usa_pos = np.zeros(self.sh)
        usa_neg = np.zeros(self.sh)

        im = self.Order.image_rect
        uim = self.Order.uimage_rect

        for i in np.arange(self.sh):
            index = np.arange(width*2+1)-width

            # First calculate SUM_i(F_i)
            F_pos = (im[posloc-width:posloc+width+1,i]).sum() 
            F_neg = (im[negloc-width:negloc+width+1,i]).sum()

            # then SUM_i(i*F_i)
            iF_pos = (index*im[posloc-width:posloc+width+1,i]).sum()
            iF_neg = (index*im[negloc-width:negloc+width+1,i]).sum()

            sa_pos[i] = iF_pos/F_pos
            sa_neg[i] = iF_neg/F_neg
       
            # Now propagate the error
            uF_pos = uim[posloc-width:posloc+width+1,i]
            uF_neg = uim[negloc-width:negloc+width+1,i]
            usa_pos[i]  = np.sqrt(((index*F_pos - iF_pos)**2 * uF_pos**2).sum())/F_pos**2
            usa_neg[i]  = np.sqrt(((index*F_neg - iF_neg)**2 * uF_neg**2).sum())/F_neg**2

        #NIRSPEC flips the spectrum on the detector (as all echelles do).
        sa_pos[i] = -sa_pos[i]
        sa_neg[i] = -sa_neg[i]

        return sa_pos*aper_corr,usa_pos*aper_corr,sa_neg*aper_corr,usa_neg*aper_corr

    def plot(self):        
        plt.plot(self.wave,self.flux_pos,drawstyle='steps-mid')
        plt.plot(self.wave,self.flux_neg,drawstyle='steps-mid')
        plt.show()

    def plotSA(self):
        plt.plot(self.wave,self.sa_pos,drawstyle='steps-mid')
        plt.plot(self.wave,self.sa_neg,drawstyle='steps-mid')
        plt.show()

    def waveGuess(self,disp):
        index = np.arange(self.sh)
        wave = disp['w0']+index*disp['dw']
        return wave

    def writeSpec(self,filename=None,path='.'):
        
        header = self.headers[0]
        c1  = pf.Column(name='wave', format='D', array=self.wave)
        c2  = pf.Column(name='flux', format='D', array=self.flux)
        c3  = pf.Column(name='uflux', format='D', array=self.uflux)
        c4  = pf.Column(name='sky', format='D', array=self.sky)
        c5  = pf.Column(name='usky', format='D', array=self.usky)        
        if self.sa:
            c6  = pf.Column(name='sa', format='D', array=self.sa)
            c7  = pf.Column(name='usa', format='D', array=self.usa)

        if self.sa:
            coldefs = pf.ColDefs([c1,c2,c3,c4,c5,c6,c7])
        else:
            coldefs = pf.ColDefs([c1,c2,c3,c4,c5])

        tbhdu = pf.BinTableHDU.from_columns(coldefs)

        header['SETNAME'] = (self.setting, 'Setting name')
        header['ECHLPOS'] = (self.echelle, 'Echelle position')
        header['DISPPOS'] = (self.crossdisp, 'Cross disperser position')
        header['ORDER'] = (str(self.onum),'Order number')        

        hdu = pf.PrimaryHDU(header=header)
        thdulist = pf.HDUList([hdu,tbhdu])

        if filename is None:
            time   = header['TIME']
            time   = time.replace(':','')
            time   = time[0:4]
            date   = header['DATE-OBS']
            date   = date.replace('-','')
            object = header['OBJECT']
            object = object.replace(' ','')
            filename = path+'/'+object+'_'+date+'_'+time+'_spec1d'+str(self.onum)+'.fits'

        
        thdulist.writeto(filename,clobber=True)

        return filename
