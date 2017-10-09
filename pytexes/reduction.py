import os
from pytexes.observation import *
from pytexes.order import *
from pytexes.spec1d import *
from pytexes.calspec import *
from pytexes.combine_orders import *

import utils.helpers as helpers

class Reduction():
    '''
    Top level basic script for reducing an observation, consisting of a science target and a telluric standard, as well
    as associated calibration files. 
    '''
    def __init__(self,flat_names=None, sci_names=None, std_names=None, path=None, level1=True,level2=True,level3=True,
                 level1_path='L1FILES',shift=0.0, dtau=0.0, save_dark=False, save_flat=False, **kwargs):

        self.flat_names = [path+name for name in flat_names]
        self.sci_names = [path+name for name in sci_names]
        self.std_names = [path+name for name in std_names]

        self.save_dark = save_dark
        self.save_flat = save_flat

        self.shift       = shift
        self.dtau        = dtau
        self.level1_path = level1_path

        self.mode  = 'SciStd'
        self.tdict = {'science':self.sci_names,'standard':self.std_names}
        if level1:
            self._level1()

        if level2:
            self._level2()
            
        if level3:
            self._level3()
        
    def _level1(self):
        OFlat = Flat(self.flat_names, save=self.save_flat)
        level1_files = {}
        for key in self.tdict.keys():
            ONod    = Nod(self.tdict[key],flat=OFlat)
            norders = ONod.getNOrders()
            target_files = []
            for i in np.arange(norders):
                OOrder   = Order(ONod,onum=i,write_path='SPEC2D')
                OSpec1D  = Spec1D(OOrder,sa=False,write_path='SPEC1D')
                OOrder_files = {'2d':OOrder.file, '1d':OSpec1D.file}
                target_files.append(OOrder_files)

            level1_files[key] = target_files

        filename = self._getLevel1File()            
        f = open(self.level1_path+'/'+filename, 'w')
        json.dump(level1_files,f)
        f.close()

    def _level2(self):

        filename = self._getLevel1File()
        f = open(self.level1_path+'/'+filename, 'r')
        level1_files = json.load(f)
        f.close()

        norders = len(level1_files['science'])

        for i in np.arange(norders):
            sci_file = level1_files['science'][i]['1d']
            std_file = level1_files['standard'][i]['1d']
            OCalSpec = CalSpec(sci_file,std_file,shift=self.shift,dtau=self.dtau,write_path='CAL1D',order=i+1)

    def _level3(self):
        filename = self._getLevel1File()
        basename = filename[0:-11]
        all_files = os.listdir('CAL1D')
        level2_files = [os.path.join('CAL1D',file) for file in all_files if basename in file]

        OCombSpec = CombSpec(level2_files,write_path='COMBSPEC',micron=True)
        

    def _getLevel1File(self):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        header = pf.open(self.sci_names[0],ignore_missing_end=True)[0].header
        basename = helpers.getBaseName(header)
        filename = basename+'_files.json'
        return filename
