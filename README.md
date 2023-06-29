# DeepChIA-PET

DeepChIA-PET is a supervised deep-learning method for predicting ChIA-PET from Hi-C and ChIP-seq. 
It used a deep dilated residual network to learn the mapping relationships between ChIA-PET and two widely-used methods (Hi-C and ChIP-seq).


## Required Python packages (versions we used) 
	1. pyBigWig (0.3.18) for parsing .bigwig ChIP-seq data
	2. straw (1.0.0.1) for extracting .hic Hi-C data
	3. numpy (1.21.5)
	4. pytorch (1.10.1)
	5. sklearn (1.0.2)
	6. scipy (1.7.3)

For installing packages 1-2: 
* pip install pyBigWig
+ pip install hic-straw==1.0.0.1

For installing packages 3-6, we highly recommend using conda. 

## Processing ChIP-seq

Before making predictions, we need to process ChIP-seq data (bigwig). 

`python process_ChIP-seq.py url/path_to_bigWig output_directory`

In the output folder, you can find two .pkl files: bw.pkl and chrlens.pkl.

## Running DeepChIA-PET

For prediction/testing, please run the following command:

`python DeepChIA-PET.py hic_file bigwig_output_directory output_directory human_or_not test_chr`

There are five required arguments:
	
	1. path to the .hic file for Hi-C data (may be a URL or path to a local .hic file)
	2. bigwig_output_directory for processing the bigwig file for ChIP-seq data
	3. output_directory for storing the final output files.
	4. human genome or not (1 for human and 0 for the others)
	5. test_chr (e.g., chr1)

In the output folder, you can find "chrid_sorted_predictions.npy", which is the final output file, containing three columns (i, j, p) where i and j are bin indexes (0-based at 10-kb resolution) and p is the predicted probability, which has already been sorted from high to low.


## A specific example for predicting RNAPII ChIA-PET of GM12878 

For a specific example, please see the Jupyter notebook document "DeepChIA-PET.ipynb". 

## Citation
Tong Liu and Zheng Wang. <a href="https://www.biorxiv.org/content/10.1101/2022.10.19.512935v1">DeepChIA-PET: Accurately predicting ChIA-PET from Hi-C and ChIP-seq with deep dilated networks.</a> <i>bioRxiv</i>, p. 2022.10.19.512935, 2022, doi: 10.1101/2022.10.19.512935.
