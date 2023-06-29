
import sys
import math
import time
import pickle
import pyBigWig

def get_bigWigSummary(file_bigWig, resolution=10000):
    '''
    given a resolution, get genome-wide bigWig summary using pyBigWig
    '''
    bw = pyBigWig.open(file_bigWig)
    # get the list of chrs and their lengths
    chrlens = bw.chroms()
    bws = {}
    for chrid in chrlens:
        numBins = math.floor(chrlens[chrid] / resolution)
        maxBind = numBins * resolution
        bws[chrid] = bw.stats(chrid, 0, maxBind, nBins=numBins)
        if chrlens[chrid] > maxBind: # for the last smaller bin
            bwLast = bw.stats(chrid, maxBind, chrlens[chrid])
            bws[chrid].append(bwLast[0])
        # replace None with zero
        bws[chrid] = [0 if b is None else b for b in bws[chrid]]

    return bws, chrlens


file_chipseq = sys.argv[1]
dirout = sys.argv[2]

res = 10000

##################################################
# get ChIP-seq data and chromosome lengths

f_bw = dirout+"/bw.pkl"
f_chrlens = dirout+"/chrlens.pkl"

bws, chrlens = get_bigWigSummary(file_chipseq, resolution=res)
with open(f_bw, 'wb') as f:
    pickle.dump(bws, f)
with open(f_chrlens, 'wb') as f:
    pickle.dump(chrlens, f)


