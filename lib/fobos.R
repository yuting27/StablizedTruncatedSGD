# FOBOS
# Yuting Ma

source('./codes/rda.R')

fobos <- function(X, y, X_test, y_test, sample_seed, par_list, loss, trace=FALSE){
  
  library(Matrix)
  
  ### Parameters
  n_epoch <- par_list$n_epoch  # number of epochs
  lambda <- par_list$lambda
  optTol <- 1e-4  # optimal tolerance
  
  p <- ncol(X)
  n <- nrow(X)
  maxIter <- n_epoch*n
  
  num_var <- rep(NA, maxIter) 
  train_err <- rep(NA, maxIter)
  test_err <- rep(NA, maxIter)
  
  set.seed(sample_seed)
  idx_seq <- sample(1:n, maxIter, replace=T) 
  
  t <- 1
  v_0 <- rep(0, p)
  
  v_t <- v_0
  #maxIter <- 2
  while(t < maxIter){
    idx <- idx_seq[t]
    x_t <- X[idx,]
    y_t <- y[idx]
    
    if(loss == 'logistic'){
      g_t <- logistic_subg(x_t, y_t, v_t)
    } else if(loss == 'hinge'){
      g_t <- hinge_subg(x_t, y_t, v_t)
    } else{
      print('wrong specification of loss function')
      break
    }
    eta_t <- 1/sqrt(t)
    v_demi <- v_t - eta_t*g_t
    v_new <- sign(v_demi) * as.numeric(abs(v_demi) - lambda > 0) * (abs(v_demi) - lambda)
    num_var[t] <- sum(v_new != 0)
    u <- X %*% v_new
    train_err[t] <- mean(u*y <= 0)
    u_test <- X_test %*% v_new
    test_err[t] <- mean(u_test*y_test <= 0)
    
    if(trace & t %% 100 == 0){
      cat("t=", t, "\n")
      cat("\ttrain_err=", train_err[t], "\n")
      cat("\ttest_err=", test_err[t], "\n")
      cat("\tnum_var=", num_var[t], "\n")
    }
    v_t <- v_new
    t <- t + 1
  }
  # end of algorithm
  i_last <- sum(!is.na(num_var))
  return(list(w=v_t, train_err = train_err, test_err=test_err,
              test_err_final = test_err[i_last],
              num_var = num_var, 
              num_var_final = num_var[i_last]))
}


