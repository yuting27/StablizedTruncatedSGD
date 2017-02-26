### Stabilized Truncated Stochastic Gradient Descent
### Main algorithm with hinge loss
### Author: Yuting Ma
### Date: 04/21/2016


soft_truncate_operator <- function(v, rej_rate){
  
  ### rej_rate = rejection rate at each soft thresholding
  
  v <- as.vector(v)
  p <- length(v)
  a <- quantile(abs(v), rej_rate)
  v_new <- (as.numeric(v > 0)*pmax(v-a*rep(1,p), 0) 
            + as.numeric(v < 0)*pmin(v+a*rep(1,p), 0))  
  return(v_new)
}


normalizer <- function(x){
  x/sqrt(sum(x^2))
}

sample_idx <- function(h, seed, n, maxIter){
  ### h = chain index
  set.seed(h*20+seed)
  idx_seq <- sample(1:n, maxIter, replace=T) 
  return(idx_seq)
}

update_idx_list <- function(tt_list, idx_list, maxIter, K, n, seed){
  h_idx <- which(tt_list >= maxIter-K)
  for(h in h_idx){
    set.seed(h+100+seed*10)
    idx_temp <- sample(1:n, n, replace=F)
    idx_list[[h]] <- c(idx_list[[h]], idx_temp) 
  }
  return(idx_list)
}


anneal_beta <- function(d, p, beta_0, gamma){
  dd <- d/p
  if(gamma > 0){
    beta <- beta_0*exp(gamma*(dd-1)) - (1- dd)*exp(-gamma)
  } else{
    beta <- beta_0* log(-gamma*dd+1)/log(1-gamma)
  }
  return(beta)
}

stabilizedTruncatedSGD_hinge <- function(X, y, X_test, y_test, seed, par_list){
  
  ### Stabilized SGD with hinge loss for binary classification
  ### Input:
  ###  - X = training features
  ###  - y = training class labels
  ###  - X_test = testing features
  ###  - y_test = testing class labels
  ###  - seed = the seed for randomized data ordering
  ###  - par_list = parameter list
  ###     - nK = the number of bursts in one stage
  ###     - n_epoch = the maximum number of epochs
  ###     - beta_0 = the inital rejection rate
  ###     - gamma = annealing rate for rejection rate
  ###     - K_0 = initial burst size
  ###     - pi_0 = purging threshold
  ###     - n_chain = number of parallel threads
  
  library(parallel)
  library(Matrix)
  
  ### Parameters
  nK <- par_list$nK
  n_epoch <- par_list$n_epoch
  beta_0 <- par_list$beta_0
  gamma <- par_list$gamma
  K_0 <- par_list$K_0
  pi_0 <- par_list$pi_0
  n_chain <- par_list$n_chain
  
  optTol <- 0.01  # optimal tolerance
  alpha <- 0.0003
  Theta <- 0.5
  
  do_validation <- TRUE
  do_plot <- FALSE
  
  p <- ncol(X)
  n <- nrow(X)
  maxIter <- n_epoch*n
  
  # randomized sampling order for SGD updates
  idx_list <- lapply(1:n_chain, function(h) sample_idx(h, seed, n, maxIter))
  
  ###################################################################
  ### Initialization
  v_0 <- rep(0, p)
  v <- v_0
  # initialize all weights as zero 
  v_list <- rep(list(v), n_chain)
  X_new <- X
  if(!is.null(X_test)){
    X_test_new <- X_test
  }
  
  ind_var <- 1:p
  K <- K_0
  d_var <- p
  ind_trunc <- 1:p
  
  beta_t <- beta_0  ### rejection rate
  
  tt_list <- rep(1, n_chain)
  train_error <- NULL
  test_error <- NULL
  num_var <- NULL
  N_var_nz <- NULL
  run_time <- NULL
  
  b_rej <- rep(0, p)
  while(min(tt_list) < maxIter){
    s <- 1
    b_Sel <- rep(0, ncol(X_new))
    b_nz_sum <- rep(0, ncol(X_new))
    while(s <= nK){
      
      burst_hinge_trunc <- function(h){
        ptm <- proc.time()
        v_00 <- v_list[[h]]
        v <- v_00
        tt <- tt_list[h]  ### starting index
        idx_seq <- idx_list[[h]][tt:length(idx_list[[h]])]
        b_nz_val <- rep(0, ncol(X_new))
        i <- 1
        t <- 1
        n_var_nz <- rep(0, K)
        b_rej <- rep(0, ncol(X_new))
        while(i <= K & t <= length(idx_seq)){
          idx <- idx_seq[t]
          x_t <- X_new[idx,]
          y_t <- y[idx]
          f_t <- as.numeric(x_t %*% v)
          nz_t <- sum(abs(x_t))
         
          ### when sampled (x_t, y_t) is within the margin 
          if(y_t*f_t <= 1 & nz_t != 0){
            b_nz_val <- b_nz_val + abs(x_t)  # check whether feature j is updated
            eta_t <- 1/sqrt(t + tt)  ### learning rate
            h_t <- v +  eta_t * y_t* x_t
            h_temp <- soft_truncate_operator(h_t, beta_t)
            b_rej <- b_rej + as.numeric(h_temp == 0)
            v <- h_t
            n_var_nz[i] <- sum(v != 0)
            i <- i + 1
          }
          t <- t + 1
        }
        trunc <- soft_truncate_operator(v, beta_t)
        v_trunc <- as.vector(trunc$v_new)
        v_new <- normalizer(v_trunc)
        
        b_nz <- as.numeric(b_nz_val > 0)
        b_sel <- as.numeric(v_new > 0)
        
        if(do_validation){
          u <- X_new %*% v_new
          train_err <- mean(u*y <= 0)
          d_var <- sum(v_new != 0)
          u_test <- X_test_new %*% v_new
          test_err<- mean(u_test*y_test <= 0)
          run_time <- (proc.time() - ptm)[1]
          return(list(v=v_new, b_sel = b_sel, 
                      b_nz = b_nz, t=t, b_rej=b_rej,
                      n_var_nz = n_var_nz,
                      train_err=train_err, test_err=test_err, 
                      d_var= d_var, run_time=run_time))
        } else{
          return(list(v=v_new, b_sel= b_sel, b_nz = b_nz, b_rej=b_rej,
                      n_var_nz = n_var_nz, t=t, d_var=d_var))
        }
      } # end of function burst_hinge_trunc
      
      sgd_chain <- mclapply(1:n_chain, burst_hinge_trunc, mc.cores=n_chain)
      v_list <- lapply(sgd_chain, function(x) x$v)
      b_Sel <- b_Sel + rowSums(do.call(cbind,lapply(sgd_chain, function(x) x$b_sel)))
      b_nz_sum <- b_nz_sum + rowSums(do.call(cbind,lapply(sgd_chain, function(x) x$b_nz)))
      N_var_nz <- rbind(N_var_nz, do.call(cbind,lapply(sgd_chain, function(x) x$n_var_nz)))
      tt_list <- sapply(sgd_chain, function(x) x$t) + tt_list
      idx_list <- update_idx_list(tt_list, idx_list, maxIter, K, n, seed)
      b_rej <- b_rej + rowSums(do.call(cbind, lapply(sgd_chain, function(x) x$b_rej)))
      
      if(do_validation){
        tt_h <- sapply(sgd_chain, function(x) x$t)
        train_err_h <- sapply(sgd_chain, function(x) x$train_err)
        test_err_h <- sapply(sgd_chain, function(x) x$test_err)
        num_var_h <- sapply(sgd_chain, function(x) x$d_var)
        run_time_h <- sapply(sgd_chain, function(x) x$run_time)
        
        run_time <- c(run_time, list(run_time_h))
        train_error <- c(train_error, list(train_err_h))
        test_error <- c(test_error, list(test_err_h))
        num_var <- c(num_var, list(num_var_h))
        
        cat("t=", min(tt_list), "\n")
        cat("\tmean train error=",mean(train_err_h), "\n")
        cat("\tmean test error=", mean(test_err_h), "\n")
        cat("\tmean d_var=", mean(num_var_h), "\n")
      }  
      
      s <- s + 1 # update burst index 
    }# end of a stage
    
    ### Stability selection
    # selection probability
    p_rej <- b_rej/(nK*n_chain*K)
    
    ### deterministically purge features
    ind_sel <- which(p_rej < 1- pi_0)
    
    d_rej <- ncol(X_new) -length(ind_sel)
    if(d_rej > 0){
      ind_var_temp <- ind_var[ind_sel]
      X_new <- X_new[,ind_sel]
      if(!is.null(X_test)){
        X_test_new <- X_test_new[,ind_sel]
      }
      v_list_temp <- lapply(v_list, function(v) v[ind_sel])
      d_var <- ncol(X_new)
      b_Sel <- b_nz_sum <- rep(0, d_var)
      b_rej <- rep(0, d_var)
      
      beta_t <- anneal_beta(d_var, p, beta_0, gamma)
      K <- ceiling(K_0*log(p/(Theta*d_var)))
      s <- 1
      do_purge <- TRUE
    }
    
    if(min(tt_list) < maxIter){
      v_list <- v_list_temp
      ind_var <- ind_var_temp
    }
  } # end of outer loop: convergence of algorithm
  
  ###########################################################################################
  
  test_err_tbl <- do.call(rbind, test_error)
  train_err_tbl <- do.call(rbind, train_error)
  num_var_tbl <- do.call(rbind, num_var)
  if(do_plot){
    par(mfrow=c(2,1))
    idxx <- 1:length(test_error)
    plot(idxx, type="n", ylim=c(0, 0.6), 
         main="Test Error and Train Error", ylab="Error Rate", xlab="Burst")
    for(h in 1:n_chain){
      lines(idxx, test_err_tbl[,h], col=h)
      lines(idxx, train_err_tbl[,h], col=h, lty="dashed")
    }
    
    plot(1:nrow(N_var_nz), type="n", ylim=c(0, p), 
         main="Number of Nonzero Variables", ylab="d",
         xlab="Iterations")
    for(h in 1:n_chain){
      points(1:nrow(N_var_nz), N_var_nz[,h], col=h, pch=16, cex=0.5)
    }
  }
  
  test_error_final <- min(test_err_tbl)
  num_var_final <- min(num_var_tbl)
  
  return(list(w=v, train_err = test_err_tbl, test_err=test_err_tbl,
              test_err_final = test_error_final,
              num_var = num_var_tbl, 
              num_var_final = num_var_final,
              ind_var=ind_var))
}


