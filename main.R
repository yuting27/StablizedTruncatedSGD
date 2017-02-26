### Stabilized Truncated Stochastic Gradient Descent
### Example Main.R: an example of implementing Stabilized Truncated SGD algorithm on Dexter data 
### with comparison with other methods discussed in the paper.
### Author: Yuting Ma 
### Date: 08/2016

setwd("~/StabilizedTruncatedSGD")

library(Matrix)
data_name <- "dexter"
load(paste0("./data/", data_name, "/dat_", data_name, ".RData"))
dat <- dat_dexter
X <- dat$X_train 
y <- dat$y_train
X_test <- dat$X_valid
y_test <- dat$y_validffob
p <- ncol(X) # p=7751
n <- nrow(X) # n=300

sum_X <- summary(X)
XX <- sparseMatrix(i=sum_X$i, j=sum_X$j, x=1)

sparsity <- colSums(XX)
seed <- 1
save(sparsity, file="./results/sparsity_dexter.RData")

num_seed <- 50

source("./codes/summary_stat.R")

#################################################################################
### hinge loss ###
#################################################################################

#####################################################################
### Standard Stochatic Gradient

source("./codes/standard_sgd_hinge.R") 

test_err_std_hinge <- rep(NA,num_seed)
num_var_std_hinge <- rep(NA, num_seed)
indSel_std_hinge <- NULL

for(seed in 1:num_seed){
  cat("s=", seed, "\n")
  result_std_hinge <- standard_sgd_hinge(X, y, X_test, y_test, seed)
  test_err_std_hinge[seed] <- result_std_hinge$test_err_final
  num_var_std_hinge[seed] <- result_std_hinge$num_var_final
  indSel_std_hinge <- c(indSel_std_hinge, list(which(result_std_hinge$w != 0)))
}

results_std_hinge <- list(test_err=test_err_std_hinge, 
                          num_var=num_var_std_hinge,
                          indSel=indSel_std_hinge)
save(results_std_hinge, file=paste0("./results/results_",data_name ,"_std_hinge.RData"))
summary_stat(results_std_hinge)

###############################################################
### Truncated Gradient Algorithm by Langford et al.

source("./codes/truncated_grad_hinge.R")
test_err_trunc_hinge <- rep(NA,num_seed)
num_var_trunc_hinge <- rep(NA, num_seed)
indSel_trunc_hinge <- NULL

par_list_trunc <- list(n_epoch=60,
                       K = 5,
                       eta = 0.01,
                       gravity=0.001)
result_trunc_hinge <- truncated_grad_hinge(X, y, X_test, y_test, seed, par_list_trunc)


for(seed in 1:num_seed){
  cat("s=", seed, "\n")
  result_truncated_hinge <- truncated_grad_hinge(X, y, X_test, y_test, seed, par_list_trunc)
  test_err_trunc_hinge[seed] <- result_truncated_hinge$test_err_final
  num_var_trunc_hinge[seed] <- result_truncated_hinge$num_var_final
  indSel_trunc_hinge <- c(indSel_trunc_hinge, list(which(result_truncated_hinge$w != 0)))
}

results_trunc_hinge <- list(test_err=test_err_trunc_hinge, 
                            num_var=num_var_trunc_hinge,
                            indSel=indSel_trunc_hinge)
save(results_trunc_hinge, file=paste0("./results/results_",data_name ,"_trunc_hinge.RData")
summary_stat(results_trunc_hinge)


#############################################################################
### Stabilized Truncated SGD algorithm
source("./codes/stsgd_algorithm_hinge.R")

par_list_ssgd <- list(nK = 5,  # number of bursts in one stage,
                      n_epoch = 20,  # number of epochs
                      beta_0 = 0.8,  # initial maximum rejection rate
                      gamma = -9,   # annealing rate of rejection rate
                      theta = 0.02, # shrinkage threshold
                      K_0 = 20,  # size of burst
                      pi_0 = 0.85, # purging threshold
                      n_chain=4)  # number of parallel chains

test_err_ssgd_hinge <- rep(NA,num_seed)
num_var_ssgd_hinge <- rep(NA, num_seed)
indSel_ssgd_hinge <- NULL

for(seed in 1:num_seed){
  cat("s=", seed, "\n")
  result_ssgd_hinge<- stabilizedTruncatedSGD_hinge(X, y, X_test, y_test, seed, par_list_ssgd)
  test_err_ssgd_hinge[seed] <- result_ssgd_hinge$test_err_final
  num_var_ssgd_hinge[seed] <- result_ssgd_hinge$num_var_final
  indSel_ssgd_hinge <- c(indSel_ssgd_hinge, list(which(result_ssgd_hinge$w != 0)))
}

results_ssgd_hinge <- list(test_err=test_err_ssgd_hinge, 
                           num_var=num_var_ssgd_hinge,
                           indSel=indSel_ssgd_hinge)
save(results_ssgd_hinge, file=paste0("./results/results_",data_name ,"_ssgd_hinge.RData"))
summary_stat(results_ssgd_hinge)


#################################################################################
### logistic loss ###
#################################################################################

#####################################################################
### Standard Stochatic Gradient
source("./codes/standard_sgd_logistic.R") 

par_list_std <- list(n_epoch=20, eta=0.1)

test_err_std_logistic <- rep(NA,num_seed)
num_var_std_logistic <- rep(NA, num_seed)
indSel_std_logistic <- NULL

for(seed in 1:num_seed){
  cat("s=", seed, "\n")
  result_std_logistic <- standard_sgd_logistic(X, y, X_test, y_test, seed, par_list_std)
  test_err_std_logistic[seed] <- result_std_logistic$test_err_final
  num_var_std_logistic[seed] <- result_std_logistic$num_var_final
  indSel_std_logistic <- c(indSel_std_logistic, list(which(result_std_logistic$w != 0)))
}

results_std_logistic <- list(test_err=test_err_std_logistic, 
                             num_var=num_var_std_logistic,
                             indSel=indSel_std_logistic)
save(results_std_logistic, file=paste0("./results/results_",data_name ,"_std_logistic.RData"))
summary_stat(results_std_logistic)

###############################################################
### Truncated Gradient Algorithm by Langford et al.

source("./codes/truncated_grad_logistic.R")
test_err_trunc_logistic <- rep(NA,num_seed)
num_var_trunc_logistic <- rep(NA, num_seed)
indSel_trunc_logistic <- NULL

par_list_trunc <- list(n_epoch=60,
                       K = 5,
                       eta = 0.01,
                       gravity=0.001)
result_trunc_logistic <- truncated_grad_logistic(X, y, X_test, y_test, seed, par_list_trunc)


for(seed in 1:num_seed){
  cat("s=", seed, "\n")
  result_truncated_logistic <- truncated_grad_logistic(X, y, X_test, y_test, seed, par_list_trunc)
  test_err_trunc_logistic[seed] <- result_truncated_logistic$test_err_final
  num_var_trunc_logistic[seed] <- result_truncated_logistic$num_var_final
  indSel_trunc_logistic <- c(indSel_trunc_logistic, list(which(result_truncated_logistic$w != 0)))
}

results_trunc_logistic <- list(test_err=test_err_trunc_logistic, 
                               num_var=num_var_trunc_logistic,
                               indSel=indSel_trunc_logistic)
save(results_trunc_logistic, file=paste0("./results/results_",data_name ,"_trunc_logistic.RData")
summary_stat(results_trunc_logistic)

#############################################################################
### Stabilized Truncated SGD algorithm
source("./codes/stsgd_algorithm_logistic.R")

par_list_ssgd <- list(nK = 5,  # number of bursts in one stage,
                      n_epoch = 20,  # number of epochs
                      beta_0 = 0.8,  # initial maximum rejection rate
                      gamma = -9,   # annealing rate of rejection rate
                      theta = 0.02, # shrinkage threshold
                      K_0 = 20,  # size of burst
                      pi_0 = 0.85, # purging threshold
                      n_chain=4)  # number of parallel chains

test_err_ssgd_logistic <- rep(NA,num_seed)
num_var_ssgd_logistic <- rep(NA, num_seed)
indSel_ssgd_logistic <- NULL

for(seed in 1:num_seed){
  cat("s=", seed, "\n")
  result_ssgd_logistic<- stabilizedTruncatedSGD_logistic(X, y, X_test, y_test, seed, par_list_ssgd)
  test_err_ssgd_logistic[seed] <- result_ssgd_logistic$test_err_final
  num_var_ssgd_logistic[seed] <- result_ssgd_logistic$num_var_final
  indSel_ssgd_logistic <- c(indSel_ssgd_logistic, list(which(result_ssgd_logistic$w != 0)))
}

results_ssgd_logistic <- list(test_err=test_err_ssgd_logistic, 
                              num_var=num_var_ssgd_logistic,
                              indSel=indSel_ssgd_logistic)
save(results_ssgd_logistic, file=paste0("./results/results_",data_name ,"_ssgd_logistic.RData"))
summary_stat(results_ssgd_logistic)


#############################################################################
### Regularized Dual Averaging algorithm
### Logistic loss & hinge loss

par_list_rda <- list(gamma = 1, rho = 0.005, lambda = 0.001, n_epoch=20)
result_rda_logistic <- rda(X, y, X_test, y_test, seed, par_list_rda, 'logistic', trace=TRUE)
result_rda_hinge <- rda(X, y, X_test, y_test, seed, par_list_rda, 'hinge', trace=TRUE)
plot_result(result_rda_logistic)

# search for parameters
lambda_list <- c(0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10 )

result <- array(dim=c(length(lambda_list),4))
for(i in 1:length(lambda_list)){
  print(lambda_list[i])
  par_list_rda <- list(gamma = 1, rho = 0.005, lambda = lambda_list[i], n_epoch=20)
  result_rda_logistic <- rda(X, y, X_test, y_test, seed, par_list_rda, 'logistic')
  result_rda_hinge <- rda(X, y, X_test, y_test, seed, par_list_rda, 'hinge')
  result[i,1] <- result_rda_logistic$test_err_final
  result[i,2] <- result_rda_logistic$num_var_final
  result[i,3] <- result_rda_hinge$test_err_final
  result[i,4] <- result_rda_hinge$num_var_final
}

### 20-repeats

par_list_rda <- list(gamma = 1, rho = 0.005, lambda = 0.001, n_epoch=5)
test_err_rda_logistic <- rep(NA,num_seed)
num_var_rda_logistic <- rep(NA, num_seed)
indSel_rda_logistic <- NULL

for(seed in 1:num_seed){
  cat("s=", seed, "\n")
  result_rda_logistic <- rda(X, y, X_test, y_test, seed, par_list_rda, 'logistic')
  test_err_rda_logistic[seed] <- result_rda_logistic$test_err_final
  num_var_rda_logistic[seed] <- result_rda_logistic$num_var_final
  indSel_rda_logistic <- c(indSel_rda_logistic, list(which(result_rda_logistic$w != 0)))
}

results_rda_logistic <- list(test_err=test_err_rda_logistic, 
                             num_var=num_var_rda_logistic,
                             indSel=indSel_rda_logistic)
save(results_rda_logistic, file=paste0("./results/results_",data_name ,"_rda_logistic.RData"))
summary_stat(results_rda_logistic)


test_err_rda_hinge <- rep(NA,num_seed)
num_var_rda_hinge <- rep(NA, num_seed)
indSel_rda_hinge <- NULL

for(seed in 1:num_seed){
  cat("s=", seed, "\n")
  result_rda_hinge <- rda(X, y, X_test, y_test, seed, par_list_rda, 'hinge')
  test_err_rda_hinge[seed] <- result_rda_hinge$test_err_final
  num_var_rda_hinge[seed] <- result_rda_hinge$num_var_final
  indSel_rda_hinge <- c(indSel_rda_hinge, list(which(result_rda_hinge$w != 0)))
}

results_rda_hinge <- list(test_err=test_err_rda_hinge, 
                          num_var=num_var_rda_hinge,
                          indSel=indSel_rda_hinge)
save(results_rda_hinge, file=paste0("./results/results_",data_name ,"_rda_hinge.RData"))
summary_stat(results_rda_hinge)

###############################################################
### FOBOS
### Logistic loss & hinge loss
par_list_fobos <- list(lambda = 0.00005, n_epoch=20)
result_fobos_logistic <- fobos(X, y, X_test, y_test, seed, par_list_fobos, 'logistic', trace=TRUE)
result_fobos_hinge <- fobos(X, y, X_test, y_test, seed, par_list_fobos, 'hinge')


# search for parameters
lambda_list <- c(0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10 )

for(i in 1:length(lambda_list)){
  print(lambda_list[i])
  par_list_fobos <- list(lambda = lambda_list[i], n_epoch=20)
  result_fobos_logistic <- fobos(X, y, X_test, y_test, seed, par_list_fobos, 'logistic')
  result_fobos_hinge <- fobos(X, y, X_test, y_test, seed, par_list_fobos, 'hinge')
  result[i,1] <- result_fobos_logistic$test_err_final
  result[i,2] <- result_fobos_logistic$num_var_final
  result[i,3] <- result_fobos_hinge$test_err_final
  result[i,4] <- result_fobos_hinge$num_var_final
}


# 20-repeats
par_list_fobos <- list(lambda = 0.001, n_epoch=20)
test_err_fobos_logistic <- rep(NA,num_seed)
num_var_fobos_logistic <- rep(NA, num_seed)
indSel_fobos_logistic <- NULL

for(seed in 1:num_seed){
  cat("s=", seed, "\n")
  result_fobos_logistic <- fobos(X, y, X_test, y_test, seed, par_list_fobos, 'logistic')
  test_err_fobos_logistic[seed] <- result_fobos_logistic$test_err_final
  num_var_fobos_logistic[seed] <- result_fobos_logistic$num_var_final
  indSel_fobos_logistic <- c(indSel_fobos_logistic, list(which(result_fobos_logistic$w != 0)))
}

results_fobos_logistic <- list(test_err=test_err_fobos_logistic, 
                               num_var=num_var_fobos_logistic,
                               indSel=indSel_fobos_logistic)
save(results_fobos_logistic, file=paste0("./results/results_",data_name ,"_fobos_logistic.fobosta"))
summary_stat(results_fobos_logistic)


test_err_fobos_hinge <- rep(NA,num_seed)
num_var_fobos_hinge <- rep(NA, num_seed)
indSel_fobos_hinge <- NULL

for(seed in 1:num_seed){
  cat("s=", seed, "\n")
  result_fobos_hinge <- fobos(X, y, X_test, y_test, seed, par_list_fobos, 'hinge')
  test_err_fobos_hinge[seed] <- result_fobos_hinge$test_err_final
  num_var_fobos_hinge[seed] <- result_fobos_hinge$num_var_final
  indSel_fobos_hinge <- c(indSel_fobos_hinge, list(which(result_fobos_hinge$w != 0)))
}

results_fobos_hinge <- list(test_err=test_err_fobos_hinge, 
                            num_var=num_var_fobos_hinge,
                            indSel=indSel_fobos_hinge)
save(results_fobos_hinge, file=paste0("./results/results_",data_name ,"_fobos_hinge.fobosta"))
summary_stat(results_fobos_hinge)

