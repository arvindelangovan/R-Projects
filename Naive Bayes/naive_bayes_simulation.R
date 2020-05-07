# Understanding Naive Bayes with randomly generated training and test data

library(mvtnorm)

means <- matrix(NA,2,2)
means[1,] <- c(-1.5,0)
means[2,] <- c(1.5,0)

plot(means)


cov_mat1 <- diag(2)  
cov_mat2 <- matrix(NA,2,2)
cov_mat2[1,] <- c(1,0.5)
cov_mat2[2,] <- c(0.5,1)
cov_mats <- list()
cov_mats[[1]] <- cov_mat1
cov_mats[[2]] <- cov_mat2

n <- 100
m <- 2
data_sep <- list()
for (k in 1:m){
  data_sep[[k]] <- rmvnorm(n,means[k,],cov_mats[[k]])
}
data <- rbind(data_sep[[1]],data_sep[[2]])
class <- c(rep(1,n),rep(2,n))
plot(data)

library(naivebayes)

nb_out <- naive_bayes(data,class)

# Plot decision boundary
mu1 <- c(nb_out$tables$V1[1,1],nb_out$tables$V2[1,1])
sds1 <- c(nb_out$tables$V1[2,1],nb_out$tables$V2[2,1])
mu2 <- c(nb_out$tables$V1[1,2],nb_out$tables$V2[1,2])
sds2 <- c(nb_out$tables$V1[2,2],nb_out$tables$V2[2,2])

delta_k_diff <- function(x1){
  value <- (prod(dnorm(c(x1,x2),mu1,sds1))-prod(dnorm(c(x1,x2),mu2,sds2)))^2
  return(value)
}

m <- 100
vec <- seq(-3,3,length.out = m)
out <- numeric(m)
for (i in 1:m){
  x2 <- vec[i]
  out[i] <- optimize(delta_k_diff,interval=c(-2,2))$minimum
}
points(out,vec,type="l",col=4)


# Test data 
n.test <- 100
test.data <- matrix(NA,n.test,2)
test.class <- sample(1:2,n.test,replace=TRUE)
for (k in 1:n.test){
  test.data[k,] <- rmvnorm(1,means[test.class[k],],cov_mats[[test.class[k]]])
}
plot(test.data,col=test.class+2,pch=16)
points(out,vec,type="l",col=2)

# Prediction
nb.pred=predict(nb_out,test.data)
nb.class=nb.pred
table(nb.class,test.class)

