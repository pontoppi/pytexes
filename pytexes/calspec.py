import numpy as np
import astropy.io.fits as pf
import utils.helpers as helpers

class CalSpec():
    def __init__(self,scifile,stdfile,shift=0.,dtau=0.0,dowave=True,write_path=None,order=0):

        self.scifile = scifile
        self.stdfile = stdfile

        self.Sci,self.header = helpers.readTEXES(scifile)
        self.Std,self.header_std = helpers.readTEXES(stdfile)

        self._normalize(self.Sci)
        self._normalize(self.Std)
        self._maskLowTrans()
        self._beers(self.Std,dtau)

        self.wave = self.Sci['wave']
        self.flux = self.Sci['flux']/self.Std['flux']
        self.uflux = self.flux*np.sqrt((self.Sci['uflux']/self.Sci['flux'])**2 + (self.Std['uflux']/self.Std['flux'])**2)

        if write_path:
            self.file = self.writeSpec(path=write_path,order=order)

    def _maskLowTrans(self,thres=0.25):
        bsubs = np.where(self.Std['flux']<thres)
        self.Std.field('flux')[bsubs] = np.nan
        self.Sci.field('flux')[bsubs] = np.nan
        
    def _specShift(self,spec,shift):
        nx = spec.size
        index = np.arange(nx)
        return np.interp(index-shift,index,spec)

    def _normalize(self,Spec):
        niter = 3
        
        median = 0.
        gsubs = np.where(Spec['flux']>median)
        for i in np.arange(niter):
            median = np.median(Spec['flux'][gsubs])
            gsubs = np.where(Spec['flux']>median)
        Spec.field('flux')[:] /= median
        Spec.field('uflux')[:] /= median

    def _beers(self,Spec,dtau):
        Spec.field('flux')[:] = np.exp((1.+dtau)*np.log(Spec['flux']))
                
    def writeSpec(self, filename=None, path='.',order=1):
        c1  = pf.Column(name='wave', format='D', array=self.wave)
        c2  = pf.Column(name='flux', format='D', array=self.flux)
        c3  = pf.Column(name='uflux', format='D', array=self.uflux)

        coldefs = pf.ColDefs([c1,c2,c3])

        tbhdu = pf.BinTableHDU.from_columns(coldefs)
        hdu = pf.PrimaryHDU(header=self.header)
        thdulist = pf.HDUList([hdu,tbhdu])

        if filename is None:
            basename = helpers.getBaseName(self.header)
            filename = path+'/'+basename+'_calspec'+str(order)+'.fits'

        thdulist.writeto(filename,clobber=True)

        return filename

