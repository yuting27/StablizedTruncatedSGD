### truncated gradient algorithm in Langford et al. (2009)

library(Matrix)
library(methods)

soft_truncate_operator2 <- function(v, a, theta=NULL){
  ### a = gravity parameter
  ### theta = threshold
  v <- as.vector(v)
  p <- length(v)
  if(is.null(theta)){ # theta = inf
    v_new <- as.numeric(v > 0)*pmax(v-a*rep(1,p), 0) + as.numeric(v < 0)*pmin(v+a*rep(1,p), 0)
  } else{
    v_new <- (as.numeric(v > 0 & v<= theta)*pmax(v-a*rep(1,p), 0) 
              + as.numeric(v < 0 & v>= -theta)*pmin(v+a*rep(1,p), 0)
              + as.numeric(v> theta | v < -theta)*v )
  }
  
  v_new <- as(v_new, "sparseMatrix")
  return(list(v_new=v_new))
}


normalizer <- function(x){
  x/sqrt(sum(x^2))
}


truncated_grad_hinge <- function(X, y, X_test, y_test, sample_seed, par_list){
  library(Matrix)
  
  ### Parameters
  K <- par_list$K   # number of bursts in one stage
  n_epoch <- par_list$n_epoch  # number of epochs
  eta <- par_list$eta # learning rate
  gravity <- par_list$gravity # gravity
  optTol <- 1e-4  # optimal tolerance
 
  do_plot <- FALSE
  trace <- FALSE
  
  p <- ncol(X)
  n <- nrow(X)
  maxIter <- n_epoch*n
  
  set.seed(sample_seed)
  idx_seq <- sample(1:n, maxIter, replace=T) 
  
  sum_X <- summary(X)
  XX <- sparseMatrix(i=sum_X$i, j=sum_X$j, x=1)
  v_temp <- colSums(diag(y) %*% XX)
  #v_temp <- colSums(diag(y) %*% X)
  #v_0 <- normalizer(v_temp)
  v_0 <- rep(0, p)
  
  v <- v_0
  
  num_var <- rep(NA, maxIter) 
  train_err <- rep(NA, maxIter)
  test_err <- rep(NA, maxIter)
  mark <- rep(1, maxIter)
  t <- 1
  i <- 1
  while(t < maxIter){
    idx <- idx_seq[t]
    x_t <- X[idx,]
    y_t <- y[idx]
    f_t <- as.numeric(x_t %*% v)
    if(y_t*f_t <= 1){
      v_new <- v +  eta * y_t* x_t
      if(i %% K == 0){
        v_new <- soft_truncate_operator2(v_new, gravity)$v_new
        mark[t] <- 2
      }
      
      num_var[t] <- sum(v_new != 0)
      u <- X %*% v_new
      train_err[t] <- mean(u*y <= 0)
      u_test <- X_test %*% v_new
      test_err[t] <- mean(u_test*y_test <= 0)
      #if(trace){
      i <- i + 1
      
      if(i %% 100 == 0){
        cat("\tt=", t, "\n")
        cat("\ttrain_err=", train_err[t], "\n")
        cat("\ttest_err=", test_err[t], "\n")
        cat("\tnum_var=", num_var[t], "\n")
      }
    }  
    
#     diff <- sqrt(sum((v - v_new)^2))
#     if(diff < optTol){
#       break
#     }
#     
    v <- v_new
    t <- t + 1
  } ### end of algorithm 
  
  
  if(do_plot){
    ind_na <- which(!is.na(num_var))
    par(mfrow=c(2,1))
    plot(num_var[ind_na],
         ylim=c(0,p), 
         ylab="d", cex=0.7,
         main="Number of Nonzero Variables")  
    
    plot(train_err[ind_na], ylim=c(0, 0.4), cex=0.7,
         ylab="Error Rate")
    points(test_err[ind_na], , cex=0.7,col="blue")
    legend(x=5300, y=0.4, legend=c("Train Error", "Test Error"), 
           pch=c(1,1), col=c("black", "blue"), cex=0.7, bty="n",
           text.width =NULL, y.intersp = 0.5)
    
    par(mfrow=c(1,1))
  }
  i_last <- sum(!is.na(num_var))
  ind_var <- which(v != 0)
  return(list(w=v, train_err = train_err, test_err=test_err,
              test_err_final = test_err[i_last],
              num_var = num_var, 
              mark=mark,
              num_var_final = num_var[i_last],
              ind_var=ind_var))
}