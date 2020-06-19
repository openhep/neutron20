#!/usr/bin/env python

import os, shelve, copy, sys, builtins
import numpy as np

import logging, logzero
_lg = logzero.logger
logzero.loglevel(logging.INFO)
basename = os.path.splitext(os.path.basename(__file__))[0]
logfilename = "/home/kkumer/tmp/{}.log".format(basename)
logzero.logfile(logfilename,
        loglevel=logging.INFO, maxBytes=1000000, backupCount=5)

import Model, Approach, Fitter, Data, utils, plots
from results import *
from utils import listdb
from abbrevs import *

# Datasets in n-space
C_BSA = data[94]
C_TSA = data[95]
C_BTSA = data[96]
C_BSS = data[102]
C_BSD = data[101]

HA15_BSS = data[116]
HA15_BSD = data[117]
HA17_BSS = data[136]
HA17_BSD = data[135]
HA20_BSS = data[140]

# Datasets in phi-space
C_BSA_phi = data[88]
C_TSA_phi = data[89]
C_BTSA_phi = data[90]
C_BSS_phi = data[100]
C_BSD_phi = data[99]

HA15_BSS_phi = data[107]+data[108]
HA15_BSD_phi = data[109]+data[110]+data[111]
HA17_BSS_phi = data[129]+data[130]+data[131]+data[132]+data[133]+data[134]
#HA17_BSS_phiR = utils.select(HA17_BSS_phi, ['Q2 > 1.7'])
HA20_BSS_phi = data[139]

sets = [C_BSA_phi, C_TSA_phi, C_BTSA_phi, C_BSS_phi, C_BSD_phi,
        HA15_BSS_phi, HA15_BSD_phi, HA17_BSS_phi, HA20_BSS_phi]

# Models
nfdb = shelve.open('/home/kkumer/projects/gpd/models/nf.db', flag='r')
fdb = shelve.open('/home/kkumer/projects/gpd/models/f.db', flag='r')

## Non-flavored
# model, KM-like
th20 = nfdb['nf-DR-20c']  # full J15  335.6/253
th20.name = 'KM20'
# NNet (without DR constraints)
thNN20 = nfdb['nfNN-J15+H17-p31']   # full J15   415.2/263   20x
thNN20.name = 'NN20'
# NNet+DR
thNNDR20 = nfdb['nfNNDR-J15+H17S-p32']   # full J15  458.9/263     20x
thNNDR20.name = 'NNDR20'

## Flavored
# model, KM-like
thf20 = fdb['f-DR-p36']
thf20.name = 'fKM20'  # full J15  465.1/279
# NNet (without DR constraints)
thfNN20 = fdb['fNN-J15+H17S+neu-p31']  # full J15  495.6/279     20x
thfNN20.name = 'fNN20'
# NNet+DR
thfNNDR20= fdb['fNNDR-J15+H17S+neu-p30']  # full J15   498.6/279    22x
thfNNDR20.name = 'fNNDR20'

nfdb.close()
fdb.close()

models = [th20, thNN20, thNNDR20, thf20, thfNN20, thfNNDR20]
fmodels = [thf20, thfNN20, thfNNDR20]

def pred(th, pt):
    """Attach a prediction of th to pt as attribute"""
    if not hasattr(pt, 'preds'):
        pt.preds = {}
    if 'nnet' not in th.m.parameters:
        # we have a normal model
        pt.preds[th.name] = th.predict(pt)
    else:
        # we have NN model
        p = []
        for k in range(len(th.m.nets)):
            th.m.parameters['nnet'] = k
            p.append(th.predict(pt))
        th.m.parameters['nnet'] = 'ALL'
        pt.preds[th.name] = p


for pt in C_BSA:
    for th in models:
        pred(th, pt)

db = shelve.open('fits20.db')
db['C_BSA'] = C_BSA
db.close()

for pt in C_TSA:
    for th in models:
        pred(th, pt)

db = shelve.open('fits20.db')
db['C_TSA'] = C_TSA
db.close()

