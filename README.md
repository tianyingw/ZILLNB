# ZILLNB
ZILLNB: Denoising Single-cell RNA-Seq Data with a Deep Learning-embedded Statistical Framework

*Dec 10, 2024, with ZILLNB version 1.0.0*

## Abstract

Single-cell RNA sequencing (scRNA-seq) data has multiple sources of heterogeneity, like sequencing platform, sequencing strategy, experimental environment, biological materials, etc. We want to design a statistical model using the ZINB model to handle the excessive zeros, aiming to remove heterogeneity. The ZILLNB model is designed to effectively impute single-cell RNA sequencing (scRNA-seq) data by leveraging a zero-inflated negative binomial framework. Its architecture integrates cell-level (V matrix) and gene-level (U matrix) information, allowing for the modeling of complex biological processes and reducing technical noise.

## Package
* R packages: gsl, turner,pscl, doParallel, optimParallel,dplyr, Matrix
* Python packages: numpy, pytorch, pandas, sklearn

## Core idea
![outline_fig1_v2](https://github.com/user-attachments/assets/b25e4460-138b-4625-a993-b77f7bd63561)

Structural illustration of link function in ZINB model fitting. The green square is the raw data. Blue squares are parameters to be learned. Orange squares are fixed. Red dashed squares have initializations learned from neural networks.

## 1 Introduction to ZILLNB
![fig1_14_7](https://github.com/user-attachments/assets/8ec08e71-8d38-4128-950d-ca00ece089df)
The proposed ZILLNB model consists of three key components. First, we employ a deep learning approach that integrates InfoVAE and GAN to extract latent cell-wise and gene-wise information simultaneously. Second, these learned latent factor matrices are used to fit a zero-inflated negative binomial (ZINB) model, where the latent factors and the coefficients of interest are iteratively updated using the expectation-maximization (EM) algorithm. This process removes noise introduced by cell-specific sampling effects (e.g., uneven library sizes) and gene-specific variation (e.g., heterogeneity across genes). The adjusted mean parameters are then used to generate a dense denoised and imputed matrix. Additionally, we provide a method to recover the count matrix based on the denoised distribution. For more information on COMSE, we recommend the user to check the following article:
> ZILLNB: Denoising Single-cell RNA-Seq Data with a Deep Learning-embedded Statistical Framework(https://doi.org/xxxxx)

## 2 Requirements

### 2.1 Package installation

For the R platform, the core dependent packages are as follows: 

``` R
packages <- c("gsl", "pscl", "turner", "dplyr", "Matrix", "doParallel", "optimParallel")
installed_packages <- installed.packages()[, "Package"]
packages_to_install <- setdiff(packages, installed_packages)
if(length(packages_to_install) > 0) {
  install.packages(packages_to_install)
} else {
  message("All packages are installed!")
}
```
For the Python platform, it is highly recommended that users install packages via conda.
``` Python
conda create -n denoise python=3.9
conda activate denoise
conda install pytorch scikit-learn pandas
```


### 2.2 Input: expression matrix
The input of ZILLNB is the **expression matrix** of scRNA-seq RNA-seq data:

* Each column represents a cell sample and each row represents a gene. 
* The row name of the matrix should be the gene symbol of gene ID.
* Expression units: The preferred expression values are raw counts. Since we will normalize the data by default.


## 3 ZILLNB Running demo
Here we used the scRNA-seq dataset from mouse brain as test data[1], which can be downloaded via GEO Accession (GSE60361) or scRNAseq package(R).
### Case1: Without Covariates
```R
  source("/home/ZILLNB/Function_ZILLNB.R")
  cores_num = 10
  path = "/home/ZILLNB/test_data/"
  file_cell = "data2CE.csv"
  file_gene = "data2GE.csv"
  sceMouseBrain = scRNAseq::ZeiselBrainData()
  counts = sceMouseBrain@assays@data$counts
  #load(paste0(path,"sceMouseBrain.RData"))
  #counts = sceMouseBrain@assays$data@listData$counts
  counts = counts[which(rowSums(counts!=0)>=10),]
  cores_num = 10


  ## Data Preparation
  data1 = log1p(sweep(counts,2,colSums(counts),FUN = "/")*1e4)
  write.csv(data1,paste(path,file_cell,sep = "/"),quote = FALSE)
  data2 = log1p(log1p(sweep(counts,1,rowSums(counts),FUN = "/")*1e4))
  pca = prcomp(data2, scale=TRUE)
  pca.var <- pca$sdev^2 
  pca.var.per <- pca.var/sum(pca.var)
  data_pca = pca$x[,1:500]
  write.csv(data_pca,paste(path,file_gene,sep = "/"),quote = FALSE)
  print("Data Trasformation Completed!")
  ## path2 is the path of your conda environment(check by "conda info --envs" in terminal)
  ## Calling for latent factor estimation step
  system(paste("path2/.conda/envs/denoise/bin/python3 ZILLNB_model.py --cell_data_name",
               file_cell,"--wdir",path,"--gene_data_name",file_gene,"--cell_model_name","CellEmbedding.pkl",
               "--gene_model_name","GeneEmbedding.pkl","--out_cell_name","CellEmbedding.csv",
               "--out_gene_name","GeneEmbedding.csv",sep = " "))

  ## Calling for fitting step
  parameters = ZILLNB_Fitting(counts,cores_num = cores_num,data_path = path,record_path = paste0(path,"record/"),record = T)
  
```


### Case2: With Covariates
```R
  source("/home/ZILLNB/Function_ZILLNB.R")
  cores_num = 10
  path = "/home/ZILLNB/test_data"
  file_cell = "data2CE.csv"
  file_gene = "data2GE.csv"
  counts =  read.delim(paste(path,"yourdata.txt",sep = "/),row.names = 1)
  counts = counts[which(rowSums(counts!=0)>=10),]
  meta = read.delim(paste(path,"yourmetadata.txt",sep = "/),row.names = 1)

  ## Data Preparation
  data1 = log1p(sweep(counts,2,colSums(counts),FUN = "/")*1e4)
  write.csv(data1,paste(path,file_cell,sep = "/"),quote = FALSE)
  data2 = log1p(log1p(sweep(counts,1,rowSums(counts),FUN = "/")*1e4))
  pca = prcomp(data2, scale=TRUE)
  pca.var <- pca$sdev^2 
  pca.var.per <- pca.var/sum(pca.var)
  data_pca = pca$x[,1:500]
  write.csv(data_pca,paste(path,file_gene,sep = "/"),quote = FALSE)
  print("Data Trasformation Completed!")

  ## Calling for latent factor estimation step
  system(paste("path2/.conda/envs/denoise/bin/python3 ZILLNB_model.py --cell_data_name",
               file_cell,"--wdir",path,"--gene_data_name",file_gene,"--cell_model_name","CellEmbedding.pkl",
               "--gene_model_name","GeneEmbedding.pkl","--out_cell_name","CellEmbedding.csv",
               "--out_gene_name","GeneEmbedding.csv",sep = " "))

  ## Calling for fitting step
  parameters = ZILLNB_Fitting(counts,cellmeta = meta,cores_num = cores_num,data_path = path,record_path = "/home/ZILLNB/test_data/record/",record = T)
  
```

## Reference
[1] # ZILLNB
ZILLNB: Denoising Single-cell RNA-Seq Data with a Deep Learning-embedded Statistical Framework


