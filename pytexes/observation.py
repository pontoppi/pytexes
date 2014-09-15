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

class Environment():
    '''
    Class to encapsulate global environment parameters.

    Parameters
    ----------
    config_file: string
        ConfigParser compliant file containing global parameters.

    Methods
    -------
    
    Attributes
    ----------
    pars: SafeConfigParser object
        Contains all global parameters

    '''
    def __init__(self,settings_file='texes.ini',detpars_file='detector.ini'):
        sys_dir = self._getSysPath()
        self.settings = cp.SafeConfigParser()
        self.settings.read(sys_dir+'/'+settings_file)
        self.detpars = cp.SafeConfigParser()
        self.detpars.read(sys_dir+'/'+detpars_file)

    def _getSysPath(self):
        sys_dir, this_filename = os.path.split(__file__)
        return sys_dir

    def getItems(self,option):
        return [self.settings.get(section,option) for section in self.settings.sections()]

    def getSections(self):
        return self.settings.sections()

    def getWaveRange(self,setting,onum):
        range_str = self.settings.get(setting,'wrange'+str(int(onum)))
        range = json.loads(range_str)
        return range
        
    def getYRange(self,setting,onum):
        range_str = self.settings.get(setting,'yrange'+str(int(onum)))
        range = json.loads(range_str)
        return range

    def getDispersion(self,setting,onum):
        A_str = self.settings.get(setting,'A'+str(int(onum)))
        B_str = self.settings.get(setting,'B'+str(int(onum)))
        C_str = self.settings.get(setting,'C'+str(int(onum)))
        R_str = self.settings.get(setting,'R'+str(int(onum)))
        As = json.loads(A_str)
        Bs = json.loads(B_str)
        Cs = json.loads(C_str)
        Rs = json.loads(R_str)
        return {'A':As,'B':Bs,'C':Cs,'R':Rs}
        
    def getDetPars(self):
        gain = self.detpars.getfloat('Detector','gain')
        rn   = self.detpars.getfloat('Detector','rn')
        dc   = self.detpars.getfloat('Detector','dc')
        nx   = self.detpars.getfloat('Detector','nx')
        ny   = self.detpars.getfloat('Detector','ny')
        return {'gain':gain,'rn':rn,'dc':dc,'nx':nx,'ny':ny}
        
    def getNOrders(self,setting):
        return self.settings.getint(setting,'norders')

class Observation():
    '''
    Private object containing a TEXES observation - that is, all exposures related to a single type of activity.

    Any specific activity (Darks, Flats, Science, etc.) are modeled as classes derived off the Observation class.

    Parameters
    ----------
    filelist: List of strings
        List of data (.fits) files associated with the observation.
    type: string
        type of observation (e.g., Dark). 

    Attributes
    ----------
    type
    Envi
    flist
    planes
    header

    Methods
    -------
    getSetting
    getNOrders
    getTargetName
    subtractFromStack
    divideInStack
    writeImage
   
    '''
    def __init__(self,filelist,type='image'):
        self.type = type
        self.Envi = Environment()
        self.flist = filelist

        self._openList()        
        self._makeHeader()
        self.stack,self.ustack = self._getStack()
        
        self.nplanes = self.stack.shape[2]
        self.height = self.stack[:,:,0].shape[0]

    def _openList(self):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        import pdb;pdb.set_trace()
        self.planes = []
        for file in self.flist:
            plane = pf.getdata(file,ignore_missing_end=True)
            self.planes.append(plane)
        self._makeHeader

        self.exp_pars = self.getExpPars()
        self.det_pars = self.Envi.getDetPars()

    def getExpPars(self):
        coadds   = self.getKeyword('COADDS')[0]
        sampmode = self.getKeyword('SAMPMODE')[0]
        nreads   = self.getKeyword('MULTISPE')[0]
        itime    = self.getKeyword('ITIME')[0]
        nexp     = len(self.planes)
        return {'coadds':coadds,'sampmode':sampmode,'nreads':nreads,'itime':itime,'nexp':nexp}

    def getSetting(self):
        echelle   = self.getKeyword('ECHLPOS')
        crossdisp = self.getKeyword('DISPPOS')

        assert len([e for e in echelle if e==echelle[0]])==len(echelle), \
            'All exposures must be taken with the same setting!'
        assert len([c for c in crossdisp if c==crossdisp[0]])==len(crossdisp), \
            'All exposures must be taken with the same setting!'

        echelle   = echelle[0]
        crossdisp = crossdisp[0]

        echelles   = self.Envi.getItems('echelle')
        crossdisps = self.Envi.getItems('crossdisp')

        setsub1 = [i for i,v in enumerate(echelles) if float(v)==echelle]
        setsub2 = [i for i,v in enumerate(crossdisps) if float(v)==crossdisp]

        sub = [i for i in setsub1 if i in setsub2]
        return self.Envi.getSections()[sub[0]],echelle,crossdisp

    def getAirmass(self):
        airmasses = self.getKeyword('AIRMASS')
        return airmasses

    def getNOrders(self):
        setting,echelle,crossdisp = self.getSetting()
        return self.Envi.getNOrders(setting)

    def getTargetName(self):
        target_name = self.header['OBJECT']
        return target_name

    def _getStack(self):
        
        nexp = self.exp_pars['nexp']
        nx = self.planes[0][0].header['NAXIS1']
        ny = self.planes[0][0].header['NAXIS2']
        
        stack = np.zeros((nx,ny,nexp))
        ustack = np.zeros((nx,ny,nexp))
        
        for i,plane in enumerate(self.planes):
            stack[:,:,i]  = plane[0].data*self.det_pars['gain'] #convert everything to e-
            ustack[:,:,i] = self._error(plane[0].data)
        return stack,ustack

    def _error(self,data):
        var_data = np.abs(data+self.exp_pars['itime']*self.det_pars['dc']+
                          self.det_pars['rn']**2/self.exp_pars['nreads'])
        return np.sqrt(var_data)
    
    def subtractFromStack(self,Obs):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
        try:
            self.uimage = np.sqrt(self.uimage**2+Obs.uimage**2)
            self.image -= Obs.image
        except:
            print 'Subtraction failed - no image calculated'

    def divideInStack(self,Obs):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)

        for i in np.arange(self.stack.shape[2]):
            plane = self.stack[:,:,i]
            uplane = self.ustack[:,:,i]
            uplane = np.sqrt((uplane/plane)**2+(Obs.uimage/Obs.image)**2)
            plane /= Obs.image
            uplane *= np.abs(plane)

            self.stack[:,:,i]  = plane
            self.ustack[:,:,i] = uplane
    
    def _collapseStack(self,stack=None,ustack=None,method='SigClip',sig=50.):
        '''
        If called without the stack keyword set, this will collapse the entire stack.
        However, the internal stack is overridden if a different stack is passed.
        For instance, this could be a stack of nod pairs.
        '''
        if stack is None:
            stack,ustack = self.stack,self.ustack

        #stack_median = np.median(stack,2)
        #stack_stddev = np.std(stack,2)
        #shape = stack.shape
        #masked_stack = ma.zeros(shape)
        
        masked_stack = ma.masked_invalid(stack)
        masked_ustack = ma.masked_invalid(ustack)
        
        image = ma.average(masked_stack,2,weights=1./masked_ustack**2)
        uimage = np.sqrt(ma.mean(masked_ustack**2,2)/ma.count(masked_ustack,2))
        
        return image, uimage

    def writeImage(self,filename=None):
        if filename is None:
            filename = self.type+'.fits'
            
        hdu = pf.PrimaryHDU(self.image.data)
        uhdu = pf.ImageHDU(self.uimage.data)
        hdulist = pf.HDUList([hdu,uhdu])
        hdulist.writeto(filename,clobber=True)

    def getKeyword(self,keyword):
        try:
            klist = [plane[0].header[keyword] for plane in self.planes]
            return klist
        except ValueError:
            print "Invalid header keyword"
        
class Flat(Observation):
    def __init__(self,filelist,dark=None,norm_thres=5000.,save=False):
        self.type = 'flat'
        self.Envi = Environment()
        self.flist = filelist
        self._openList()
        self._makeHeader()
        self.stack,self.ustack = self._getStack()
        self.nplanes = self.stack.shape[2]
        self.height = self.stack[:,:,0].shape[0]
        self.setting,self.echelle,self.crossdisp = self.getSetting()

        self.image,self.uimage = self._collapseStack()

        if dark:
            self.subtractFromStack(dark)

        self._normalize(norm_thres)

        #Where the flat field is undefined, it's set to 1 to avoid divide by zeros.
        self.image[np.where(self.image<0.1)] = 1
        if save:
            self.writeImage()
        
    def _normalize(self,norm_thres):
        flux = np.median(self.image[np.where(self.image>norm_thres)])
        self.image = self.image/flux
        self.uimage = self.uimage/flux

    def makeMask(self):
        return np.where(self.image<0.1,0,1)

class Dark(Observation):
    def __init__(self,filelist,save=False):
        self.type = 'dark'
        self.Envi = Environment()
        self.flist = filelist
        self._openList()
        self.stack,self.ustack = self._getStack()
        self.nplanes = self.stack.shape[2]
        self.height = self.stack[:,:,0].shape[0]
        
        self.image,self.uimage = self._collapseStack()
        self._badPixMap()
        if save:
            self.writeImage()
            
    def _badPixMap(self,clip=30,filename='badpix.dmp'):
        median = np.median(self.image)
        var  = tvar(self.image,(-100,100))
        self.badpix = ma.masked_greater(self.image-median,clip*np.sqrt(var))
        if filename is not None:
            self.badpix.dump(filename)
        
class Nod(Observation):
    def __init__(self,filelist,dark=None,flat=None,badpix='badpix.dmp'):
        self.type = 'nod'
        self.Envi = Environment()
        self.flist = filelist
        self._openList()
        self._makeHeader()
                
        self.setting,self.echelle,self.crossdisp = self.getSetting()
        self.airmasses = self.getAirmass()

        self.airmass = np.mean(self.airmasses)
        
        RAs  = self.getKeyword('RA')
        DECs = self.getKeyword('DEC')
        
        #A average nod requires a Pair Stack. 
        pairs = self._getPairs(RAs,DECs)
        self.stack,self.ustack = self._makePairStack(pairs)
        self.height = self.stack[:,:,0].shape[0]        
        if flat:
            self.divideInStack(flat)
        if badpix:
            badmask = np.load(badpix)
            self._correctBadPix(badmask)
        offsets = self._findYOffsets()
        self.stack   = self._yShift(offsets,self.stack)
        self.ustack  = self._yShift(offsets,self.ustack)       
        self.image,self.uimage = self._collapseStack()
        
        #An averaged sky frame is constructed using the straight stack collapse.
        self.stack,self.ustack = self._getStack()
        if flat:
            self.divideInStack(flat)
        if badpix:
            badmask = np.load(badpix)
            self._correctBadPix(badmask)
        offsets = [offset for offset in offsets for i in range(2)]
        self.stack   = self._yShift(offsets,self.stack)
        self.ustack  = self._yShift(offsets,self.ustack)

        self.sky,self.usky = self._collapseStack()
        self.sky -= np.absolute(self.image)/2.
        self.usky = np.sqrt(self.uimage**2/2.+self.usky**2)
        
    def _correctBadPix(self,badmask):
        for i in np.arange(self.stack.shape[2]):
            plane = self.stack[:,:,i]            
            maskedImage = np.ma.array(plane, mask=badmask.mask)
            NANMask = maskedImage.filled(np.NaN)
            self.stack[:,:,i] = inpaint.replace_nans(NANMask, 5, 0.5, 1, 'idw')          
            
            plane = self.ustack[:,:,i]            
            maskedImage = np.ma.array(plane, mask=badmask.mask)
            NANMask = maskedImage.filled(np.NaN)            
            self.ustack[:,:,i] = inpaint.replace_nans(NANMask, 5, 0.5, 1, 'idw')          
            

    def _makePairStack(self,pairs):
        npairs = len(pairs)
        nx = self.planes[0][0].header['NAXIS1']
        ny = self.planes[0][0].header['NAXIS2']

        stack,ustack = self._getStack()
        pair_stack  = np.zeros((nx,ny,npairs))
        pair_ustack = np.zeros((nx,ny,npairs))
        
        for i,pair in enumerate(pairs):
            pair_stack[:,:,i] = stack[:,:,pair[0]] - stack[:,:,pair[1]]
            pair_ustack[:,:,i] = np.sqrt(ustack[:,:,pair[0]]**2 + ustack[:,:,pair[1]]**2)
            
        return pair_stack,pair_ustack

    def _findYOffsets(self):
        
        kernel = np.median(self.stack[:,:,0],1)
        offsets = np.empty(0)
        nplanes = self.stack.shape[2]
        for i in np.arange(nplanes):
            profile = np.median(self.stack[:,:,i],1)
            cc = fp.ifft(fp.fft(kernel)*np.conj(fp.fft(profile)))
            cc_sh = fp.fftshift(cc)
            cen = calc_centroid(cc_sh).real - self.height/2.
            offsets = np.append(offsets,cen)
        return offsets

    def _yShift(self,offsets,stack):

        sh = stack.shape
        internal_stack = np.zeros(sh)

        index = np.arange(sh[0])
        for plane in np.arange(sh[2]):
            for i in np.arange(sh[1]):
                col = np.interp(index-offsets[plane],index,stack[:,i,plane])
                internal_stack[:,i,plane] = col

        return internal_stack

    def _getPairs(self,RAs,DECs):
        nexp = len(RAs)
        #        assert nexp&2==0, "There must be an even number of exposures"
        AorB = []
        RA_A  = RAs[0]
        DEC_A = DECs[0]

        dist = np.sqrt(((np.array(RAs)-RA_A)*3600)**2+((np.array(DECs)-DEC_A)*3600)**2)
        max_dist = np.max(dist)
        
        for RA,DEC in zip(RAs,DECs):
            if abs(RA-RA_A)*3600.<max_dist/2. and abs(DEC-DEC_A)*3600.<max_dist/2.:
                AorB.append('A')
            else:
                AorB.append('B')
        ii = 0
        pairs = []
        while ii<nexp:
            if AorB[ii]!=AorB[ii+1]:
                if AorB[ii]=='A':
                    pair = (ii,ii+1)
                else:
                    pair = (ii+1,ii)
                pairs.append(pair)
                ii+=2
            else:
                ii+=1

        return pairs


