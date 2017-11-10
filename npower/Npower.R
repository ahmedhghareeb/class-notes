rattle()



#============================================================

# Rattle is Copyright (c) 2006-2017 Togaware Pty Ltd.
# It is open source software and is freely available.
# It is licensed under the GNU General Public License,
# Version 2. Rattle comes with ABSOLUTELY NO WARRANTY.
# Rattle was written by Graham Williams with contributions
# from others as acknowledged in 'library(help=rattle)'.
# Visit https://rattle.togaware.com/ for details.

#============================================================
# Rattle timestamp: 2017-11-09 02:37:12 x86_64-w64-mingw32 

# Rattle version 5.1.0 user 'Ahmed'

# This log captures Rattle interactions as an R script. 

# For repeatability export this log of all activity to a 
# file using the Export button or the Tools menu. This 
# script can serve as a starting point for developing your 
# own scripts. Exporting to a file called 'model.R' will 
# allow you to type into a new R Console the command 
#"source('model.R')" and so repeat all actions. Generally, 
# you will want to edit the file to suit your own needs. 
# You can also edit this log in place to record additional 
# information before exporting the script. 

# Note that saving/loading projects retains this log.

# We begin most scripts by loading the required packages.
# Here are some initial packages to load and others will be
# identified as we proceed through the script. When writing
# our own scripts we often collect together the library
# commands at the beginning of the script here.

library(rattle)   # Access weather dataset and utilities.
library(magrittr) # For the %>% and %<>% pipeline operators.

# This log generally records the process of building a model. 
# However, with very little effort the log can also be used 
# to score a new dataset. The logical variable 'building' 
# is used to toggle between generating transformations, 
# when building a model and using the transformations, 
# when scoring a dataset.

building <- TRUE
scoring  <- ! building

# A pre-defined value is used to reset the random seed 
# so that results are repeatable.

crv$seed <- 42 

#============================================================
# Rattle timestamp: 2017-11-09 02:43:20 x86_64-w64-mingw32 

# Load an R data frame.

crs$dataset <- train

# Display a simple summary (structure) of the dataset.

str(crs$dataset)

#============================================================
# Rattle timestamp: 2017-11-09 02:43:21 x86_64-w64-mingw32 

# Note the user selections. 

# Build the train/validate/test datasets.

# nobs=731 train=511 validate=109 test=111

set.seed(crv$seed)

crs$nobs     <- nrow(crs$dataset)
crs$train    <- crs$sample <- sample(crs$nobs, 0.7*crs$nobs)
crs$validate <- sample(setdiff(seq_len(crs$nobs), crs$train), 0.15*crs$nobs)
crs$test     <- setdiff(setdiff(seq_len(crs$nobs), crs$train), crs$validate)

# The following variable selections have been noted.

crs$input     <- c("Temp", "Wind_Speed", "Precip_Amount", "Chill",
                   "Solar_Rad", "Humidity", "Cloud_Cover", "Demand")

crs$numeric   <- c("Temp", "Wind_Speed", "Precip_Amount", "Chill",
                   "Solar_Rad", "Humidity", "Cloud_Cover", "Demand")

crs$categoric <- NULL

crs$target    <- NULL
crs$risk      <- NULL
crs$ident     <- "Date"
crs$ignore    <- NULL
crs$weights   <- NULL

#============================================================
# Rattle timestamp: 2017-11-09 02:43:27 x86_64-w64-mingw32 

# Note the user selections. 

# Build the train/validate/test datasets.

# nobs=731 train=511 validate=109 test=111

set.seed(crv$seed)

crs$nobs     <- nrow(crs$dataset)
crs$train    <- crs$sample <- sample(crs$nobs, 0.7*crs$nobs)
crs$validate <- sample(setdiff(seq_len(crs$nobs), crs$train), 0.15*crs$nobs)
crs$test     <- setdiff(setdiff(seq_len(crs$nobs), crs$train), crs$validate)

# The following variable selections have been noted.

crs$input     <- c("Temp", "Wind_Speed", "Precip_Amount", "Chill",
                   "Solar_Rad", "Humidity", "Cloud_Cover")

crs$numeric   <- c("Temp", "Wind_Speed", "Precip_Amount", "Chill",
                   "Solar_Rad", "Humidity", "Cloud_Cover")

crs$categoric <- NULL

crs$target    <- "Demand"
crs$risk      <- NULL
crs$ident     <- "Date"
crs$ignore    <- NULL
crs$weights   <- NULL

#============================================================
# Rattle timestamp: 2017-11-09 02:43:39 x86_64-w64-mingw32 

# Decision Tree 

# The 'rpart' package provides the 'rpart' function.

library(rpart, quietly=TRUE)

# Reset the random number seed to obtain the same results each time.

set.seed(crv$seed)

# Build the Decision Tree model.

crs$rpart <- rpart(Demand ~ .,
                   data=crs$dataset[crs$train, c(crs$input, crs$target)],
                   method="anova",
                   parms=list(split="information"),
                   control=rpart.control(usesurrogate=0, 
                                         maxsurrogate=0))

# Generate a textual view of the Decision Tree model.

print(crs$rpart)
printcp(crs$rpart)
cat("\n")

# Time taken: 0.02 secs

#============================================================
# Rattle timestamp: 2017-11-09 02:43:39 x86_64-w64-mingw32 

# Build a Random Forest model using the traditional approach.

set.seed(crv$seed)

crs$rf <- randomForest::randomForest(Demand ~ .,
                                     data=crs$dataset[crs$sample, c(crs$input, crs$target)], 
                                     ntree=500,
                                     mtry=2,
                                     importance=TRUE,
                                     na.action=randomForest::na.roughfix,
                                     replace=FALSE)

# Generate textual output of the 'Random Forest' model.

crs$rf

# List the importance of the variables.

rn <- crs$rf %>%
  randomForest::importance() %>%
  round(2)
rn[order(rn[,1], decreasing=TRUE),]

# Time taken: 0.67 secs

#============================================================
# Rattle timestamp: 2017-11-09 02:43:40 x86_64-w64-mingw32 

# Regression model 

# Build a Regression model.

crs$glm <- lm(Demand ~ ., data=crs$dataset[crs$train,c(crs$input, crs$target)])

# Generate a textual view of the Linear model.

print(summary(crs$glm))
cat('==== ANOVA ====
    
    ')
print(anova(crs$glm))
print("
      ")

# Time taken: 0.06 secs

#============================================================
# Rattle timestamp: 2017-11-09 02:43:40 x86_64-w64-mingw32 

# Neural Network 

# Build a neural network model using the nnet package.

library(nnet, quietly=TRUE)

# Build the NNet model.

set.seed(199)
crs$nnet <- nnet(Demand ~ .,
                 data=crs$dataset[crs$sample,c(crs$input, crs$target)],
                 size=10, linout=TRUE, skip=TRUE, MaxNWts=10000, trace=FALSE, maxit=100)

# Print the results of the modelling.

cat(sprintf("A %s network with %d weights.\n",
            paste(crs$nnet$n, collapse="-"),
            length(crs$nnet$wts)))
cat(sprintf("Inputs: %s.\n",
            paste(crs$nnet$coefnames, collapse=", ")))
cat(sprintf("Output: %s.\n",
            names(attr(crs$nnet$terms, "dataClasses"))[1]))
cat(sprintf("Sum of Squares Residuals: %.4f.\n",
            sum(residuals(crs$nnet) ^ 2)))
cat("\n")
print(summary(crs$nnet))
cat('\n')

# Time taken: 0.03 secs

#============================================================
# Rattle timestamp: 2017-11-09 02:43:59 x86_64-w64-mingw32 

# Evaluate model performance on the validation dataset. 

# RPART: Generate a Predicted v Observed plot for rpart model on train [validate].

crs$pr <- predict(crs$rpart, newdata=crs$dataset[crs$validate, c(crs$input, crs$target)])

# Obtain the observed output for the dataset.

obs <- subset(crs$dataset[crs$validate, c(crs$input, crs$target)], select=crs$target)

# Handle in case categoric target treated as numeric.

obs.rownames <- rownames(obs)
obs <- as.numeric(obs[[1]])
obs <- data.frame(Demand=obs)
rownames(obs) <- obs.rownames

# Combine the observed values with the predicted.

fitpoints <- na.omit(cbind(obs, Predicted=crs$pr))

# Obtain the pseudo R2 - a correlation.

fitcorr <- format(cor(fitpoints[,1], fitpoints[,2])^2, digits=4)

# Plot settings for the true points and best fit.

op <- par(c(lty="solid", col="blue"))

# Display the observed (X) versus predicted (Y) points.

plot(fitpoints[[1]], fitpoints[[2]], asp=1, xlab="Demand", ylab="Predicted")

# Generate a simple linear fit between predicted and observed.

prline <- lm(fitpoints[,2] ~ fitpoints[,1])

# Add the linear fit to the plot.

abline(prline)

# Add a diagonal representing perfect correlation.

par(c(lty="dashed", col="black"))
abline(0, 1)

# Include a pseudo R-square on the plot

legend("bottomright",  sprintf(" Pseudo R-square=%s ", fitcorr),  bty="n")

# Add a title and grid to the plot.

title(main="Predicted vs. Observed
      Decision Tree Model
      train [validate]",
      sub=paste("Rattle", format(Sys.time(), "%Y-%b-%d %H:%M:%S"), Sys.info()["user"]))
grid()

# RF: Generate a Predicted v Observed plot for rf model on train [validate].

crs$pr <- predict(crs$rf, newdata=na.omit(crs$dataset[crs$validate, c(crs$input, crs$target)]))

# Obtain the observed output for the dataset.

obs <- subset(na.omit(crs$dataset[crs$validate, c(crs$input, crs$target)]), select=crs$target)

# Handle in case categoric target treated as numeric.

obs.rownames <- rownames(obs)
obs <- as.numeric(obs[[1]])
obs <- data.frame(Demand=obs)
rownames(obs) <- obs.rownames

# Combine the observed values with the predicted.

fitpoints <- na.omit(cbind(obs, Predicted=crs$pr))

# Obtain the pseudo R2 - a correlation.

fitcorr <- format(cor(fitpoints[,1], fitpoints[,2])^2, digits=4)

# Plot settings for the true points and best fit.

op <- par(c(lty="solid", col="blue"))

# Display the observed (X) versus predicted (Y) points.

plot(fitpoints[[1]], fitpoints[[2]], asp=1, xlab="Demand", ylab="Predicted")

# Generate a simple linear fit between predicted and observed.

prline <- lm(fitpoints[,2] ~ fitpoints[,1])

# Add the linear fit to the plot.

abline(prline)

# Add a diagonal representing perfect correlation.

par(c(lty="dashed", col="black"))
abline(0, 1)

# Include a pseudo R-square on the plot

legend("bottomright",  sprintf(" Pseudo R-square=%s ", fitcorr),  bty="n")

# Add a title and grid to the plot.

title(main="Predicted vs. Observed
      Random Forest Model
      train [validate]",
      sub=paste("Rattle", format(Sys.time(), "%Y-%b-%d %H:%M:%S"), Sys.info()["user"]))
grid()

# GLM: Generate a Predicted v Observed plot for glm model on train [validate].

crs$pr <- predict(crs$glm, 
                  type    = "response",
                  newdata = crs$dataset[crs$validate, c(crs$input, crs$target)])

# Obtain the observed output for the dataset.

obs <- subset(crs$dataset[crs$validate, c(crs$input, crs$target)], select=crs$target)

# Handle in case categoric target treated as numeric.

obs.rownames <- rownames(obs)
obs <- as.numeric(obs[[1]])
obs <- data.frame(Demand=obs)
rownames(obs) <- obs.rownames

# Combine the observed values with the predicted.

fitpoints <- na.omit(cbind(obs, Predicted=crs$pr))

# Obtain the pseudo R2 - a correlation.

fitcorr <- format(cor(fitpoints[,1], fitpoints[,2])^2, digits=4)

# Plot settings for the true points and best fit.

op <- par(c(lty="solid", col="blue"))

# Display the observed (X) versus predicted (Y) points.

plot(fitpoints[[1]], fitpoints[[2]], asp=1, xlab="Demand", ylab="Predicted")

# Generate a simple linear fit between predicted and observed.

prline <- lm(fitpoints[,2] ~ fitpoints[,1])

# Add the linear fit to the plot.

abline(prline)

# Add a diagonal representing perfect correlation.

par(c(lty="dashed", col="black"))
abline(0, 1)

# Include a pseudo R-square on the plot

legend("bottomright",  sprintf(" Pseudo R-square=%s ", fitcorr),  bty="n")

# Add a title and grid to the plot.

title(main="Predicted vs. Observed
      Linear Model
      train [validate]",
      sub=paste("Rattle", format(Sys.time(), "%Y-%b-%d %H:%M:%S"), Sys.info()["user"]))
grid()

# NNET: Generate a Predicted v Observed plot for nnet model on train [validate].

crs$pr <- predict(crs$nnet, newdata=crs$dataset[crs$validate, c(crs$input, crs$target)])

# Obtain the observed output for the dataset.

obs <- subset(crs$dataset[crs$validate, c(crs$input, crs$target)], select=crs$target)

# Handle in case categoric target treated as numeric.

obs.rownames <- rownames(obs)
obs <- as.numeric(obs[[1]])
obs <- data.frame(Demand=obs)
rownames(obs) <- obs.rownames

# Combine the observed values with the predicted.

fitpoints <- na.omit(cbind(obs, Predicted=crs$pr))

# Obtain the pseudo R2 - a correlation.

fitcorr <- format(cor(fitpoints[,1], fitpoints[,2])^2, digits=4)

# Plot settings for the true points and best fit.

op <- par(c(lty="solid", col="blue"))

# Display the observed (X) versus predicted (Y) points.

plot(fitpoints[[1]], fitpoints[[2]], asp=1, xlab="Demand", ylab="Predicted")

# Generate a simple linear fit between predicted and observed.

prline <- lm(fitpoints[,2] ~ fitpoints[,1])

# Add the linear fit to the plot.

abline(prline)

# Add a diagonal representing perfect correlation.

par(c(lty="dashed", col="black"))
abline(0, 1)

# Include a pseudo R-square on the plot

legend("bottomright",  sprintf(" Pseudo R-square=%s ", fitcorr),  bty="n")

# Add a title and grid to the plot.

title(main="Predicted vs. Observed
      Neural Net Model
      train [validate]",
      sub=paste("Rattle", format(Sys.time(), "%Y-%b-%d %H:%M:%S"), Sys.info()["user"]))
grid()



################################

#Prediction
results<- predict(crs$rf,test)
print(results)
write.csv(results, file = "npower_test01.csv",row.names=T)


attach(train)

plot(Temp,Demand,col=c("red","green"))
plot(Cloud_Cover,Demand,col=c("red","green"))
plot(Humidity,Demand,col=c("red","green"))
plot(Chill,Demand,col=c("red","green"))
plot(Precip_Amount,Demand,col=c("red","green"))
plot(Solar_Rad,Demand,col=c("red","green"))
plot(Wind_Speed,Demand,col=c("red","green"))

#################################
#Multiple Linear Regression

mylm01<- lm(Demand~ Temp+Solar_Rad+Chill, data = train)
summary(mylm01)

mylm02<- lm(Demand~ Temp+Solar_Rad+Humidity, data = train)
summary(mylm02)

mylm03<- lm(Demand~ Temp+Solar_Rad+Humidity+Chill, data = train)
summary(mylm03)


##################################






