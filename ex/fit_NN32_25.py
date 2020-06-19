
import logging, logzero, os
_lg = logzero.logger
logzero.loglevel(logging.INFO)
basename = os.path.splitext(os.path.basename(__file__))[0]
logfilename = "/home/kkumer/tmp/{}.log".format(basename)
logzero.logfile(logfilename, loglevel=logging.INFO)

import sys, shelve
GEPARD_DIR = '/home/kkumer/gepard'
sys.path.append(GEPARD_DIR+'/pype')
import Model, Approach, Fitter, Data, utils, plots
import numpy
from results import *
from abbrevs import *


CLAS14pts = CLAS14TSApts + CLAS14BTSApts + CLAS14BSApts
CLAS15pts = C_BSDwpts + C_BSSw0pts + C_BSSw1pts
HA15pts = H_BSDwpts + H_BSSw0pts + H_BSSw1pts
HA17pts = H17_BSDwpts + H17_BSSw0pts + H17_BSSw1pts


pts = GLO15new + H17_BSSw0pts + H17_BSSw1pts


numn = 2

numpy.random.seed(95)

m = Model.ModelNN(zeropointpower=False, hidden_layers=[13],
        output_layer=['ImH', 'ReH', 'ImE', 'ReE', 'ImHt', 'ImEt'],
        useDR = ['ReH', 'ReE'],
        )
_lg.info('New model created: {}'.format(m.__class__))

th = Approach.BM10tw2(m)
th.name = "nfNNDR-J15+H17S-p3225"
f = Fitter.FitterBrain(pts, th, nnets=numn, nbatch=30, minprob=0.0000001)
th.fitpoints = f.fitpoints
f.verbose = 1
_lg.info('Start fit to {} data points'.format(len(f.fitpoints)))
f.fitgood(minchi=600.)

_lg.info('{} NNs {} {:.1f}/{} p={:.3g}'.format(numn, th.name, *th.chisq(f.fitpoints)))

th.description = '2x -13-6DR, J15+H17 BSSw'
a32db = shelve.open('a32.db')
th.save(a32db)
a32db.close()

_lg.info('Done. Emailing log file.')
utils.mailfile('kkumer@calculon.phy.hr', 'kkumer@calculon.phy.hr',
        '{} fit is done'.format(basename), logfilename)
