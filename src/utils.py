
import math
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import minmax_scale
import pyBigWig
import straw

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

def norm_minmax(mat):
    # given a 3D mat, minmax along the last two dimensions
    B, H, W = mat.shape
    mat2 = np.reshape(mat, (B, -1))
    mat3 = minmax_scale(mat2, axis=1)
        
    return np.reshape(mat3, (B,H,W))

def get_hic(file_hic, chrid, chrlen, resolution=10000, norm="KR"):
    '''
    given a .hic file, get sparse matrices of intra-chr.
    '''
    result = straw.straw('observed', norm, file_hic, chrid, chrid, 'BP', resolution)

    # convert genomic position to a relative bin position
    extract = lambda x: (x.binX, x.binY, x.counts)
    result2 = np.array(list(map(extract, result)), dtype = np.float64)
    row_inds = result2[:,0] // resolution
    col_inds = result2[:,1] // resolution

    # put into a sparse matrix format
    nBins = math.ceil(chrlen / resolution)
    mat_csr = sp.csr_matrix((result2[:,2], (row_inds.astype(int), col_inds.astype(int))), shape=(nBins, nBins))
    # make it symmetric and prevent doubling of diagonal
    mat_csr = mat_csr + mat_csr.T - sp.diags(mat_csr.diagonal())

    return mat_csr, nBins


def sigmoid(x):
	return 1./(1. + np.exp(-x, dtype=np.float128))


def diag_indices(n, k):
	rows, cols = np.diag_indices(n)
	rows, cols = list(rows), list(cols)
	rows2, cols2 = rows.copy(), cols.copy()
	for i in range(1, (k+1)):
		rows2 += rows[:-i]
		cols2 += cols[i:]
	return np.array(rows2), np.array(cols2)


def sparse_divide_nonzero(a, b):
	inv_b = b.copy()
	inv_b.data = 1 / inv_b.data
	return a.multiply(inv_b)

