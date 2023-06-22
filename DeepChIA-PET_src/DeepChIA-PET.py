import os
import sys
import math
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import minmax_scale
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data

from utils import *
from model import *


##### global parameters
np.seterr(divide='ignore', invalid='ignore')
res = 10000
image_stride = 50
image_size = 250
mesha, meshb = np.meshgrid(np.arange(0, image_size), np.arange(0, image_size))
batch_size = 2
model_url = "http://dna.cs.miami.edu/DeepChIA-PET/"


##### input arguments
file_hic = sys.argv[1]
dir_chipseq = sys.argv[2]
dir_out = sys.argv[3]
human = sys.argv[4]
chrid = sys.argv[5]

chrid2 = chrid.replace("chr","")

##### determine the best model
#dir_work = os.path.abspath(os.getcwd())
dir_work = model_url

if human == "1":
	best_model  = dir_work+"models/bm_"+chrid2
	if chrid == 'chrX':
		best_model  = dir_work+"models/bm_23"
else:
	best_model  = dir_work+"models/bm_1"

print("Model used: ", best_model, flush=True)

##################################################
# step 1: get ChIP-seq data and chromosome lengths
print("step 1: getting ChIP-seq data and chromosome length information...", flush=True)

f_bw = dir_chipseq+"/bw.pkl"
f_chrlens = dir_chipseq+"/chrlens.pkl"

if os.path.isfile(f_bw) and os.path.isfile(f_chrlens):
	with open(f_bw, 'rb') as f:
		bws = pickle.load(f)
	with open(f_chrlens, 'rb') as f:
		chrlens = pickle.load(f)
else:
	print("Can't get processed ChIP-seq data!", flush=True)
	exit()

print("step 1: done.", flush=True)


####################################################################
# step 2: get Hi-C data, combine Hi-C and ChIP-seq, do normalization
print("step 2: extracting Hi-C data and generating input data...", flush=True)

chr_len = chrlens[chrid]
bins = math.ceil(chr_len / res)
print(chrid, chr_len, bins, flush=True)

hics, nBins = get_hic(file_hic, chrid2, chr_len)
L = hics.shape[0]

if L != bins:
	L = bins

allInds = np.arange(0, L-image_size, image_stride)
lastInd = allInds[len(allInds)-1]
if (lastInd + image_size) < L:
	allInds = np.append(allInds, L-image_size)

fin = dir_out+"/"+chrid+"_input.npy"
if os.path.isfile(fin):
	dat_test = np.load(fin)
else:
	all_hics, all_chipseqa, all_chipseqb = [], [], []
	for i in allInds:
		l1, l2 = i, i + image_size
		# Hi-C data
		submat = hics[l1:l2, l1:l2].toarray()
		# set NaNs to zero
		submat[np.isnan(submat)] = 0
		mati = np.log2(submat + 1)
		all_hics.append(mati)
		# ChIP-seq data
		chipseqi = minmax_scale(np.array(bws[chrid])[l1:l2])
		mati_chipseqa, mati_chipseqb = chipseqi[mesha], chipseqi[meshb]
		all_chipseqa.append(mati_chipseqa)
		all_chipseqb.append(mati_chipseqb)

	# normalize using mean and std
	all_hics = norm_minmax(np.array(all_hics))
	all_hics = (all_hics - 0.28948) / 0.15136
	all_chipseqa = (np.array(all_chipseqa) - 0.1105) / 0.1241
	all_chipseqb = (np.array(all_chipseqb) - 0.1105) / 0.1241
	dat_test = np.array([all_hics, all_chipseqa, all_chipseqb])
	dat_test = np.transpose(dat_test, (1,0,2,3))
	#np.save(fin, dat_test)
print("Input data: ", dat_test.shape, flush=True)
print("step 2: done.", flush=True)


##########################
# step 3: make predictions

print("step 3: start predicting...", flush=True)
test_loader = torch.utils.data.DataLoader(data.TensorDataset(torch.from_numpy(dat_test)), batch_size=batch_size, shuffle=False)

### load model
print("loading model...", flush=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet(3)

#model_load = torch.utils.model_zoo(best_model, model_dir=dir_out)
state_dict = torch.hub.load_state_dict_from_url(best_model, model_dir=dir_out)

if torch.cuda.device_count() > 1:
	model = nn.DataParallel(model)
model.to(device).eval()
if torch.cuda.device_count() > 1:
	model.module.load_state_dict(state_dict)
else:
	model.load_state_dict(state_dict)
print("loading model done.", flush=True)

### predicting
print("predicting...", flush=True)
result = np.zeros((dat_test.shape[0], dat_test.shape[2], dat_test.shape[3]))
for i, data in enumerate(test_loader):
	data2 = Variable(data[0]).to(device, dtype=torch.float)
	output = model(data2)
	resulti = sigmoid(output.cpu().data.numpy())
	resulti = np.squeeze(resulti)
	i1 = i * batch_size
	i2 = i1 + batch_size
	if i == int(dat_test.shape[0] / batch_size):
		i2 = dat_test.shape[0]
	result[i1:i2,:,:] = resulti
print("prediction done.", flush=True)

print("getting the final predicted matrix...", flush=True)
rows, cols = diag_indices(bins, image_size - 1)
# matrix for predicted probabilities
mp = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(bins, bins))
mp = mp + mp.T - sp.diags(mp.diagonal())
# matrix for overlapping numbers
mn = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(bins, bins))
mn = mn + mn.T - sp.diags(mn.diagonal())

for i in range(result.shape[0]):
	i1 = allInds[i]
	i2 = i1 + image_size
	mp[i1:i2, i1:i2] += result[i,:,:]
	mn[i1:i2, i1:i2] += np.ones((image_size, image_size))

mp.data -= 1
mn.data -= 1
mp.eliminate_zeros()
mn.eliminate_zeros()
mpn = sparse_divide_nonzero(mp, mn)
mpn = (mpn + mpn.T) / 2.0

### sort 
print("sorting...", flush=True)
rows, cols = mpn.nonzero()
upper_tri = rows < cols
rows2, cols2 = rows[upper_tri], cols[upper_tri]
dat = np.array(np.vstack((rows2, cols2, mpn[rows2, cols2])).T)
dat_sorted = dat[np.argsort(dat[:,2])] # sort based on predicted probability
dat_sorted = np.flip(dat_sorted, 0) # from max to min

### save
print("saving...", flush=True)

#command = "rm "+fin
#os.system(command)

fout = dir_out+"/"+chrid+"_sorted_predictions.npy"
np.save(fout, dat_sorted)
print("step 3: done.", flush=True)
print("Done! Please check the output file: ", fout, flush=True)


