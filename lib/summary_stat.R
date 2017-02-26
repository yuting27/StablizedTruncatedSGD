summary_stat <- function(result){
  err_mean <- mean(result$test_err)*100
  err_sd <- sd(result$test_err)*100
  d_per <- result$num_var/p*100
  d_mean <- mean(d_per)
  d_sd <- sd(d_per)
  var_count <- rep(0, p)
  for(s in 1:num_seed){
	ind_var <- result$indSel[[s]]
    var_count[ind_var] <- var_count[ind_var]+1
  }
  avg_sel_prob <- mean(var_count[which(var_count > 0)])/num_seed*100
  cat("err_mean=", round(err_mean, digits=2), "\n")
  cat("err_sd = ", round(err_sd, digits=2),"\n")
  cat("d_mean = ",round(d_mean, digits=2) ,"\n")
  cat("d_sd = ", round(d_sd, digits=2),"\n")
  cat("avg_sel_prob = ", round(avg_sel_prob, digits=2),"\n")
}

kappa_coef <- function(result){
  
}