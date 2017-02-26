# Regularized Dual Averaging with l-1 penalty
# Enhanced l1_RDA (Algorithm 2)
# Yuting Ma


rda <- function(X, y, X_test, y_test, sample_seed, par_list, loss, trace=FALSE){
  
  library(Matrix)
  
  ### Parameters
  n_epoch <- par_list$n_epoch  # number of epochs
  lambda <- par_list$lambda
  rho <- par_list$rho
  gamma <- par_list$gamma # learning rate
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
  g_0 <- rep(0, p)
  v_0 <- rep(0, p)
  
  v_t <- v_0
  g_bar <- g_0
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
    
    g_bar <- (t-1)*g_bar/t  + g_t/t
    lambda_t <- lambda + gamma*rho/sqrt(t)
    v_new <- - as.numeric(abs(g_bar) - lambda_t > 0) * sqrt(t)*(g_bar - lambda_t*sign(g_bar))/gamma
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

sign <- function(v){
  return(1*as.numeric(v>0) -1*as.numeric(v < 0))
}

logistic_subg <- function(x_t, y_t, v_t){
  f_t <- as.numeric(x_t %*% v_t)
  u_t <- 1/(1+exp(f_t*y_t))
  g <- -y_t * (x_t* u_t)
  return(g)
}


hinge_subg <- function(x_t, y_t, v_t){
  f_t <- as.numeric(x_t %*% v_t)
  g <- -y_t*as.numeric(y_t*f_t <= 1)* x_t
  return(g)
}

plot_result <- function(result){
  num_var <- result$num_var
  train_err <- result$train_err
  test_err <- result$test_err
  ind_na <- which(!is.na(num_var))
  par(mfrow=c(2,1))
  plot(num_var[ind_na],
       ylim=c(0,p), 
       ylab="d", cex=0.7,
       main="Number of Nonzero Variables")  
  
  plot(train_err[ind_na], ylim=c(0, 0.7), cex=0.7,
       ylab="Error Rate")
  points(test_err[ind_na], , cex=0.7,col="blue")
  legend(x=5300, y=0.4, legend=c("Train Error", "Test Error"), 
         pch=c(1,1), col=c("black", "blue"), cex=0.7, bty="n",
         text.width =NULL, y.intersp = 0.5)
  
  par(mfrow=c(1,1))

}