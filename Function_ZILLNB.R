library(gsl)
library(turner)
library(pscl)
library(doParallel)
library(optimParallel)
ToOneHot <- function(meta){
  level_meta = levels(as.factor(meta))
  onehot <- matrix(0, nrow = length(meta), ncol = length(level_meta))
  names(onehot) = level_meta
  for (i in 1:length(meta)) {
    onehot[i, which(level_meta %in% meta[i])] = 1
  }
  return(onehot)
}
UpdateZ <- function(parameters,xdata){
  ### Compute intermediate terms 
  inter1 =  lgamma(xdata + 1/parameters$thelta) - lgamma(xdata + 1) - lgamma(1/parameters$thelta)
  inter2 = (-1/parameters$thelta)* log( 1 +  parameters$Mu*parameters$thelta)
  #inter3 = xdata * (log(parameters_fixed$Mu) + log(parameters_fixed$thelta) - log(1 +  parameters_fixed$Mu*parameters_fixed$thelta))
  #inter3 = xdata * (log(Mu * parameters_fixed$thelta) - log(1 +  Mu * parameters_fixed$thelta))
  inter3 = xdata * (log(parameters$Mu * parameters$thelta) - log(1 + parameters$Mu * parameters$thelta))
  p_y =   inter1 + inter2 + inter3 
  p_y = exp(p_y)
  inter = (counts == 0) * (parameters$phi / ((counts == 0) * parameters$phi + p_y * (1 - parameters$phi)))
  inter[is.na(inter)] = 0
  return(inter)
}
UpdateThelta <- function(parameters,counts,ncores = cores_num){
  Theltaj_calculation <- function(j,counts,parameters){
    thelta_f <- function(x){
      inter1 =  Rfast::Lgamma(as.numeric(counts[j,] + 1/x)) - Rfast::Lgamma(as.numeric(counts[j,] + 1))- Rfast::Lgamma(1/x)
      inter2 = (-1/x)* log( 1 +  parameters$Mu[j,]*x)
      
      #inter3 = xdata * (log(parameters_fixed$Mu) + log(parameters_fixed$thelta) - log(1 +  parameters_fixed$Mu*parameters_fixed$thelta))
      #inter3 = xdata * (log(Mu * parameters_fixed$thelta) - log(1 +  Mu * parameters_fixed$thelta))
      inter3 = counts[j,] * (log(parameters$Mu[j,] * x) - log(1 + parameters$Mu[j,] * x))
      p_y =   inter1 + inter2 + inter3 
      #print(p_z[1:5,1:5])
      inter =   p_y * (1-parameters$Z[j,])
      print(x)
      ## phi is close to 1
      #print(inter[1:5,1:5])
      return(-sum(inter))
    }
    thelta_gr <- function(x){
      inter1 = (1-parameters$Z[j,]) # n dim
      
      inter2 = gsl::psi(1/x) - gsl::psi(counts[j,]+1/x) + log(1 + x*parameters$Mu[j,]) + x *(counts[j,]-parameters$Mu[j,])/(1 + x*parameters$Mu[j,])
      inter = sum(inter1 * inter2)
      inter = inter / (x^2)
      return(-inter)
    }
    #theltaj_new =  optim(par = parameters$thelta[j],fn = thelta_f, gr = thelta_gr,method = "L-BFGS-B",lower = 1e-4,upper = 1e4)$par
    theltaj_new =  optim(par = parameters$thelta[j], fn = thelta_f, gr = thelta_gr, method = "L-BFGS-B",lower = 1e-4,upper = 1e4,control=list(maxit=50000))$par
    return(theltaj_new)
  }
  cores=detectCores()
  cl <- makeCluster(min(cores-1,ncores))
  registerDoParallel(cl)
  result = foreach(j = 1:parameters$nfeatures,.combine = c)%dopar% Theltaj_calculation(j,counts,parameters)
  stopCluster(cl)
  return(result)
}
UpdatePhi <- function(parameters){
  return(rowMeans(parameters$Z))
}
UpdateZeta <- function(parameters, counts,ncores = cores_num){
  Zetaj_calculation <- function(j,counts, parameters){
    Q_zetaj <- function(x){
      if(sum(is.na(parameters$gamma))==0){
        mu_j = exp(parameters$ksi + x + as.vector(parameters$beta %*% parameters$U[j,]) + 
                     as.vector(parameters$alpha[j,] %*% t(parameters$V)) + as.vector(parameters$gamma[j,] %*% t(parameters$C)))
      }else{
        mu_j = exp(parameters$ksi + x + as.vector(parameters$beta %*% parameters$U[j,]) + 
                     as.vector(parameters$alpha[j,] %*% t(parameters$V)))
      }

      mu_j[which(mu_j > 1e6)] = 1e6
      mu_j[which(mu_j < 1e-40)] = 1e-40
      inter2 = -(1/parameters$thelta[j]) * log(1 + parameters$thelta[j] * mu_j) + 
        counts[j,] * log((parameters$thelta[j] * mu_j)/(1 + parameters$thelta[j] * mu_j))
      inter2 = inter2 * (1 - parameters$Z[j,])
      if(parameters$sigmaZeta !=0){
        return(sum(-inter2) +  1/(2*parameters$sigmaZeta) * x^2) 
      }else{
        return(sum(-inter2)) 
      }
    }
    Q_zetaj_gr <- function(x){
      if(sum(is.na(parameters$gamma))==0){
        mu_j = exp(parameters$ksi + x + as.vector(parameters$beta %*% parameters$U[j,]) + 
                     as.vector(parameters$alpha[j,] %*% t(parameters$V)) + as.vector(parameters$gamma[j,] %*% t(parameters$C)))
      }else{
        mu_j = exp(parameters$ksi + x + as.vector(parameters$beta %*% parameters$U[j,]) + 
                     as.vector(parameters$alpha[j,] %*% t(parameters$V)))
      }
      mu_j[which(mu_j > 1e6)] = 1e6
      mu_j[which(mu_j < 1e-40)] = 1e-40
      dQ_mu = (1 - parameters$Z[j,]) * (counts[j,]- mu_j) / (1 + parameters$thelta[j] * mu_j)
      if(parameters$sigmaZeta !=0){
        return(-sum(dQ_mu)+ 1/parameters$sigmaZeta * x)
      }else{
        return(-sum(dQ_mu))
      }
    }
    return(optim(par = parameters$zeta[j],fn = Q_zetaj, gr = Q_zetaj_gr, method = "L-BFGS-B")$par)
  }
  cores=detectCores()
  cl <- makeCluster(min(cores-1,ncores))
  registerDoParallel(cl)
  result = foreach(j = 1:parameters$nfeatures,.combine = c)%dopar% Zetaj_calculation(j,counts,parameters)
  stopCluster(cl)
  #rownames(result) = rownames(parameters$U)
  return(result)
}
UpdateKsi <-function(parameters , counts,ncores = cores_num){
  Ksii_calculation <- function(i,counts, parameters){
    Q_ksii <- function(x){
      if(sum(is.na(parameters$gamma))==0){
        mu_i = exp(x + parameters$zeta + as.vector(parameters$beta[i,] %*% t(parameters$U)) + 
                     as.vector(parameters$alpha %*% parameters$V[i,]) + as.vector(parameters$gamma %*% parameters$C[i,]))
      }else{
        mu_i = exp(x + parameters$zeta + as.vector(parameters$beta[i,] %*% t(parameters$U)) + 
                     as.vector(parameters$alpha %*% parameters$V[i,]))
      }

      mu_i[which(mu_i > 1e6)] = 1e6
      mu_i[which(mu_i < 1e-40)] = 1e-40
      inter2 = -(1/parameters$thelta) * log(1 + parameters$thelta * mu_i) + 
        counts[,i] * log((parameters$thelta * mu_i)/(1 + parameters$thelta * mu_i))
      inter2 = inter2 * (1 - parameters$Z[,i])
      #inter4 = - as.vector(t(x)%*%x)/parameters$sigmak
      if(parameters$sigmaKsi!=0){
        return(sum(-(inter2)) +  1/(2*parameters$sigmaKsi) * x^2 )
      }else{
        return(sum(-(inter2)))
      }
    }
    Q_ksii_gr <- function(x){
      if(sum(is.na(parameters$gamma))==0){
        mu_i = exp(x + parameters$zeta + as.vector(parameters$beta[i,] %*% t(parameters$U)) + 
                     as.vector(parameters$alpha %*% parameters$V[i,]) + as.vector(parameters$gamma %*% parameters$C[i,]))
      }else{
        mu_i = exp(x + parameters$zeta + as.vector(parameters$beta[i,] %*% t(parameters$U)) + 
                     as.vector(parameters$alpha %*% parameters$V[i,]))
      }
      mu_i[which(mu_i > 1e6)] = 1e6
      mu_i[which(mu_i < 1e-40)] = 1e-40
      dQ_mu = (1 - parameters$Z[,i]) * (counts[,i]- mu_i) / (1 + parameters$thelta * mu_i) 
      if(parameters$sigmaKsi!=0){
        return(-sum(dQ_mu) + 1/parameters$sigmaKsi * x)
      }else{
        return(-sum(dQ_mu))
      }
    }
    return(optim(par = parameters$ksi[i],fn = Q_ksii, gr = Q_ksii_gr,method = "L-BFGS-B")$par)
  }
  cores=detectCores()
  cl <- makeCluster(min(cores-1,ncores))
  registerDoParallel(cl)
  result = foreach(i = 1:parameters$nsamples,.combine = c)%dopar% Ksii_calculation(i,counts,parameters)
  stopCluster(cl)
  return(result)
}
CalculationMu <- function(parameters){
  # Precompute reused terms
  Mu = matrix(0,nrow = parameters$nfeatures, ncol = parameters$nsamples)
  #U_H_V = parameters$U%*%parameters$H%*%t(parameters$V)
  beta_U = t(parameters$beta %*% t(parameters$U))
  alpha_V = parameters$alpha %*% t(parameters$V)
  if(sum(is.na(parameters$gamma))==0){
    gamma_C = parameters$gamma %*% t(parameters$C) 
  }else{
    gamma_C = 0
  }
  # Compute Mu in matrix form
  #Mu = U_H_V  + beta_U + parameters$zeta
  Mu =   beta_U + parameters$zeta + Mu + alpha_V + gamma_C
  Mu = t(t(Mu) +  parameters$ksi)
  Mu = exp(Mu) 
  Mu[which(Mu > 1e6)] = 1e6
  Mu[which(Mu < 1e-40)] = 1e-40
  return(Mu)
}
UpdateAlpha <- function(parameters,counts,ncores = cores_num){
  Alphaj_calculation <- function(j,counts,parameters){
    Alphaj_f <- function(x){
      #print(x)
      #max_mu = max(counts)
      test = as.vector(exp((x - parameters$alpha[j,])%*%t(parameters$V)))
      mu_j = parameters$Mu[j,] * test
      mu_j[which(mu_j > 1e6)] = 1e6
      mu_j[which(mu_j < 1e-40)] = 1e-40
      inter1 =  lgamma(counts[j,] + 1/parameters$thelta[j]) - lgamma(counts[j,] + 1) - lgamma(1/parameters$thelta[j])
      if(length(which(is.na(inter1)))){
        print(x)
        print("inter1 here")
      }
      inter2 = (-1/parameters$thelta[j])* log( 1 + mu_j*parameters$thelta[j])
      if(length(which(is.na(inter2)))){
        print(x)
        print("inter2 here")
      }
      inter3 = counts[j,] * (log(mu_j * parameters$thelta[j]) - log(1 + mu_j * parameters$thelta[j]))
      if(length(which(is.na(inter3)))){
        print(head(mu_j))
        print(x)
        print("inter3 here")
        return(x)
      }
      p_y =   inter1 + inter2 + inter3 
      #print(p_z[1:5,1:5])
      inter =   p_y * (1-parameters$Z[j,]) 
      ## phi is close to 1
      #print(inter[1:5,1:5])
      return(-sum(inter))
    }
    Alphaj_gr <- function(x){
      test = as.vector(exp((x - parameters$alpha[j,])%*%t(parameters$V)))
      mu_j = parameters$Mu[j,] * test
      mu_j[which(mu_j > 1e6)] = 1e6
      mu_j[which(mu_j < 1e-40)] = 1e-40
      dQ_mu = (1 - parameters$Z[j,]) * (counts[j,] - mu_j) / (1 + parameters$thelta[j] * mu_j)
      dQ_alpha = colSums(dQ_mu * parameters$V)
      return(-as.vector(dQ_alpha))
    }
    alpha_new =  optim(par = parameters$alpha[j,], fn = Alphaj_f, gr = Alphaj_gr,method = "L-BFGS-B" )$par
    return(alpha_new)
  }
  cores=detectCores()
  cl <- makeCluster(min(cores-1,ncores))
  registerDoParallel(cl)
  result = foreach(j = 1:parameters$nfeatures,.combine = rbind)%dopar% Alphaj_calculation(j,counts,parameters)
  stopCluster(cl)
  return(result)
}
UpdateBeta <- function(parameters,counts,ncores = cores_num){
  Betai_calculation <- function(i,counts,parameters){
    Betai_f <- function(x){
      test = as.vector(exp((x - parameters$beta[i,])%*%t(parameters$U)))
      mu_i = parameters$Mu[,i] * test
      mu_i[which(mu_i > 1e6)] = 1e6
      mu_i[which(mu_i < 1e-40)] = 1e-40
      inter1 =  lgamma(counts[,i] + 1/parameters$thelta) - lgamma(counts[,i] + 1) - lgamma(1/parameters$thelta)
      inter2 = (-1/parameters$thelta)* log( 1 + mu_i*parameters$thelta)
      inter3 = counts[,i] * (log(mu_i * parameters$thelta) - log(1 + mu_i * parameters$thelta))
      p_y =   inter1 + inter2 + inter3 
      #print(p_z[1:5,1:5])
      inter =   p_y * (1-parameters$Z[,i]) 
      ## phi is close to 1
      #print(inter[1:5,1:5])
      return(-sum(inter))
    }
    Betai_gr <- function(x){
      test = as.vector(exp((x - parameters$beta[i,])%*%t(parameters$U)))
      mu_i = parameters$Mu[,i] * test
      mu_i[which(mu_i > 1e6)] = 1e6
      mu_i[which(mu_i < 1e-40)] = 1e-40
      dQ_mu = (1 - parameters$Z[,i]) * (counts[,i] - mu_i) / (1 + parameters$thelta * mu_i)
      dQ_beta = colSums(dQ_mu * parameters$U)
      return(-as.vector(dQ_beta))
    }
    beta_new =  optim(par = parameters$beta[i,], fn = Betai_f, gr = Betai_gr, method = "L-BFGS-B")$par
    return(beta_new)
  }
  
  cores=detectCores()
  cl <- makeCluster(min(cores-1,ncores))
  registerDoParallel(cl)
  result = foreach(i = 1:parameters$nsamples,.combine = rbind)%dopar% Betai_calculation(i,counts,parameters)
  stopCluster(cl)
  return(result)
}
UpdateGamma <- function(parameters,counts,ncores = cores_num){
  Gammaj_calculation <- function(j,counts,parameters){
    Gammaj_f <- function(x){
      #print(x)
      #max_mu = max(counts)
      test = as.vector(exp((x - parameters$gamma[j,])%*%t(parameters$C)))
      mu_j = parameters$Mu[j,] * test
      mu_j[which(mu_j > 1e6)] = 1e6
      mu_j[which(mu_j < 1e-40)] = 1e-40
      inter1 =  lgamma(counts[j,] + 1/parameters$thelta[j]) - lgamma(counts[j,] + 1) - lgamma(1/parameters$thelta[j])
      if(length(which(is.na(inter1)))){
        print(x)
        print("inter1 here")
      }
      inter2 = (-1/parameters$thelta[j])* log( 1 + mu_j*parameters$thelta[j])
      if(length(which(is.na(inter2)))){
        print(x)
        print("inter2 here")
      }
      inter3 = counts[j,] * (log(mu_j * parameters$thelta[j]) - log(1 + mu_j * parameters$thelta[j]))
      if(length(which(is.na(inter3)))){
        print(head(mu_j))
        print(x)
        print("inter3 here")
        return(x)
      }
      p_y =   inter1 + inter2 + inter3 
      #print(p_z[1:5,1:5])
      inter =   p_y * (1-parameters$Z[j,]) 
      ## phi is close to 1
      #print(inter[1:5,1:5])
      return(sum(-inter + 1/(2*parameters$sigmaGamma) * x^2 ))
    }
    Gammaj_gr <- function(x){
      test = as.vector(exp((x - parameters$gamma[j,])%*%t(parameters$C)))
      mu_j = parameters$Mu[j,] * test
      mu_j[which(mu_j > 1e6)] = 1e6
      mu_j[which(mu_j < 1e-40)] = 1e-40
      dQ_mu = (1 - parameters$Z[j,]) * (counts[j,] - mu_j) / (1 + parameters$thelta[j] * mu_j)
      dQ_gamma = colSums(dQ_mu * parameters$C)
      return(-as.vector(dQ_gamma) + 1/parameters$sigmaGamma * x)
    }
    gamma_new =  optim(par = parameters$gamma[j,], fn = Gammaj_f, gr = Gammaj_gr,method = "L-BFGS-B" )$par
    return(gamma_new)
  }
  cores=detectCores()
  cl <- makeCluster(min(cores-1,ncores))
  registerDoParallel(cl)
  result = foreach(j = 1:parameters$nfeatures,.combine = rbind)%dopar% Gammaj_calculation(j,counts,parameters)
  stopCluster(cl)
  return(result)
}
UpdateU<-function(parameters,counts,ncores = 10){
  Uj_calculation <- function(j,counts, parameters){
    Q_uj <- function(x){
      if(sum(is.na(parameters$gamma))==0){
        mu_j = exp(parameters$ksi + parameters$zeta[j] + as.vector(parameters$beta %*% x) + 
                     as.vector(parameters$alpha[j,] %*% t(parameters$V)) + 
                     as.vector(parameters$gamma[j,] %*% t(parameters$C)))
      }else{
        mu_j = exp(parameters$ksi + parameters$zeta[j] + as.vector(parameters$beta %*% x) + 
                     as.vector(parameters$alpha[j,] %*% t(parameters$V))) 
      }
      mu_j[which(mu_j > 1e6)] = 1e6
      mu_j[which(mu_j < 1e-40)] = 1e-40
      inter2 = -(1/parameters$thelta[j]) * log(1 + parameters$thelta[j] * mu_j) + 
        counts[j,] * log((parameters$thelta[j] * mu_j)/(1 + parameters$thelta[j] * mu_j))
      inter2 = (1 - parameters$Z[j,]) * inter2
      inter3 = as.vector(x%*%x)/(2*parameters$sigmaU)
      return(inter3 - sum(inter2))
    }
    Q_uj_gr <- function(x){
      if(sum(is.na(parameters$gamma))==0){
        mu_j = exp(parameters$ksi + parameters$zeta[j] + as.vector(parameters$beta %*% x) + 
                     as.vector(parameters$alpha[j,] %*% t(parameters$V)) + 
                     as.vector(parameters$gamma[j,] %*% t(parameters$C)))
      }else{
        mu_j = exp(parameters$ksi + parameters$zeta[j] + as.vector(parameters$beta %*% x) + 
                     as.vector(parameters$alpha[j,] %*% t(parameters$V))) 
      }
      mu_j[which(mu_j > 1e6)] = 1e6
      mu_j[which(mu_j < 1e-40)] = 1e-40
      dQ_mu = (1 - parameters$Z[j,]) * (counts[j,]- mu_j) / (1 + parameters$thelta[j] * mu_j)
      x_gr = colSums(dQ_mu * parameters$beta) - x/parameters$sigmaU
      if(length(which(is.na(x_gr)))){
        print(1)
      }
      return(-x_gr)
    }
    return(optim(par = parameters$U[j,],fn = Q_uj,gr=Q_uj_gr,method = "L-BFGS-B")$par)
  }
  cl <- makeCluster(ncores)
  registerDoParallel(cl)
  result = foreach(j = 1:dim(parameters$U)[1],.combine = rbind)%dopar% Uj_calculation(j,counts,parameters)
  stopCluster(cl)
  rownames(result) = rownames(parameters$U)
  return(result)
}
QfunctionCalculation <- function(parameters,counts){
  inter1 =  lgamma(counts + 1/parameters$thelta) - lgamma(counts + 1) - lgamma(1/parameters$thelta)
  
  inter2 = (-1/parameters$thelta)* log( 1 +  parameters$Mu*parameters$thelta)
  
  #inter3 = xdata * (log(parameters_fixed$Mu) + log(parameters_fixed$thelta) - log(1 +  parameters_fixed$Mu*parameters_fixed$thelta))
  #inter3 = xdata * (log(Mu * parameters_fixed$thelta) - log(1 +  Mu * parameters_fixed$thelta))
  inter3 = counts * (log(parameters$Mu * parameters$thelta) - log(1 + parameters$Mu * parameters$thelta))
  p_y =   inter1 + inter2 + inter3 
  #print(p_z[1:5,1:5])
  Q =   sum(p_y * (1-parameters$Z))
  
  Q =  Q + sum(parameters$Z * log(parameters$phi + 1e-12) + (1-parameters$Z) * log(1 - parameters$phi))
  if(parameters$sigmaKsi!=0){
    Q = Q - sum(parameters$ksi * parameters$ksi)/(2*parameters$sigmaKsi)
  }
  if(parameters$sigmaZeta!=0){
    Q = Q - sum(parameters$zeta * parameters$zeta)/(2*parameters$sigmaZeta)
  }

  if(parameters$sigmaGamma!=0){
    Q = Q - sum(parameters$gamma * parameters$gamma)/(2*parameters$sigmaGamma)
  }
  if(parameters$flag_U  & parameters$sigmaU!=0){
    Q = Q - sum(diag(parameters$U%*%t(parameters$U)))/(2*parameters$sigmaU)
  }
  return(Q)
}
ZILLNB_Fitting<-function(counts,cellmeta=NA,max_iter = 5,cores_num = 10,file_name = "",data_path = "/home/ZILLNB/test_data/",
                         sigmaKsi = 100,sigmaZeta = 100,sigmaU = 100,flag_U = F,sigmaGamma = 100, record_path = "/home/ZILLNB/",record = F){
  parameters = list()
  parameters$flag_U = flag_U
  parameters$nfeatures = dim(counts)[1]
  parameters$nsamples = dim(counts)[2]
  parameters$phi = rowSums(counts==0)/parameters$nsamples
  parameters$thelta = rep(5,parameters$nfeatures)
  parameters$U = as.matrix(read.csv(paste(data_path,"GeneEmbedding.csv",sep = "/"), header=FALSE))
  parameters$V = as.matrix(read.csv(paste(data_path,"CellEmbedding.csv",sep = "/"), header=FALSE))
  parameters$K = dim(parameters$U)[2]
  parameters$L = dim(parameters$V)[2]
  parameters$Z = matrix(0, nrow = parameters$nfeatures,ncol = parameters$nsamples)
  parameters$ksi = log(colSums(counts))
  parameters$ksi = parameters$ksi - mean(parameters$ksi)
  parameters$zeta = log(rowSums(counts))
  parameters$zeta = parameters$zeta - mean(parameters$zeta)
  parameters$beta = matrix(0,parameters$nsamples ,parameters$K)
  parameters$alpha = matrix(0,parameters$nfeatures,parameters$L)
  if(!is.na(cellmeta)){
    parameters$C = cell_meta
    parameters$Cd = dim(parameters$C)[2]
    parameters$gamma = matrix(0,parameters$nfeatures,parameters$Cd)
    parameters$sigmaGamma = sigmaGamma
  }else{
    parameters$gamma = NA
    parameters$sigmaGamma = 0
  }
  parameters$Mu = CalculationMu(parameters)
  parameters$sigmaKsi = sigmaKsi
  parameters$sigmaZeta = sigmaZeta
  parameters$sigmaU = sigmaU
  #parameters$sigmaGamma = sigmaGamma
  print("Parameters Initiation Completed!")
  Q_loss = QfunctionCalculation(parameters,counts)
  if(!dir.exists(record_path) & record){
    system(paste("mkdir ",record_path,sep = ""))
  }
  for(j in 1:max_iter){
    parameters$Z = UpdateZ(parameters,counts)
    print("Z Updated!")
    print("E Step Completed!")
    parameters$phi = UpdatePhi(parameters)
    print(head(parameters$phi))
    parameters$thelta = UpdateThelta(parameters,counts,ncores = cores_num)
    parameters$zeta = UpdateZeta(parameters,counts,ncores = cores_num)
    parameters$ksi = UpdateKsi(parameters,counts)
    parameters$Mu = CalculationMu(parameters)
    Q_loss = c(Q_loss,QfunctionCalculation(parameters,counts))
    print("Phi, Thelta, Zeta, Ksi Updated!")
    
    parameters$alpha = UpdateAlpha(parameters,counts)
    parameters$Mu = CalculationMu(parameters)
    print("Coefficient Alpha Updated!")
    if(!is.na(parameters$gamma)){
      parameters$gamma = UpdateGamma(parameters,counts)
      parameters$Mu = CalculationMu(parameters)
      print("Coefficient Gamma Updated!")
    }

    
    parameters$beta = UpdateBeta(parameters,counts)
    parameters$Mu = CalculationMu(parameters)
    print("Coefficient Beta Updated!")
    
    if(parameters$flag_U){
      parameters$U = UpdateU(parameters,counts)
      parameters$Mu = CalculationMu(parameters)
      print("Coefficient U Updated!")
    }
    
    Q_loss = c(Q_loss,QfunctionCalculation(parameters,counts))
    print("M Step Completed!")
    print(paste("iteration:",j,sep = ""))
    print(Q_loss)
    #print(Q_loss[i])
    parameters$Q_loss = Q_loss
    if(record){
      saveRDS(parameters,paste(record_path,"/parameters_iter",j,".rds",sep = "")) 
    }
  }
  return(parameters)
}
