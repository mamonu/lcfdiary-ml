library(nnet)
library(ggplot2)
library(reshape2)
library(maxent)
library(RTextTools)
library(xgboost)

#clean up the r environment.
rm(list = ls())
#@home working directory
setwd("/Users/thorosm2002/Dropbox/Rcode/MLinR")

Afile <- 'DEFdry.csv'



rd <- read.csv2(Afile, sep=",", stringsAsFactors=FALSE,header = TRUE)
rd <- rd[,c("ITEMDESC","CCPNODOT")]

keptdata <- rd[! (is.na(rd$ITEMDESC) | 
                    rd$ITEMDESC=="" | 
                    is.na(as.numeric(rd$CCPNODOT)) )   , ]


keptdata <- head(keptdata,10000)
testdata <-  tail(keptdata,8500)
traindata<- head(keptdata,1500)

dtMatrix <- create_matrix(keptdata,language="english",removeNumbers=FALSE,stemWords=FALSE,removeSparseTerms = .9999)

# Configure the training data
container <- create_container(dtMatrix, keptdata$CCPNODOT, trainSize=1:1500, testSize = 1501:10000, virgin=FALSE)


svmmodel <- train_model(container,"SVM")
maxentmodel <- train_model(container,"MAXENT")


results <- classify_model(container,svmmodel)
results <- classify_model(container,maxentmodel)

#analytics<- create_analytics(container , results )



# how did we go? 


total <- cbind(results,testdata)
total$TF <- ifelse(total$CCPNODOT==total$SVM_LABEL,1,0)
hist(total$TF)



MEtotal <- cbind(results,testdata)
MEtotal$TF <- ifelse(MEtotal$CCPNODOT==MEtotal$MAXENTROPY_LABEL,1,0)
hist(MEtotal$TF)


#cross validation
SVM_CROSS <- cross_validate(container,4,algorithm="SVM")
#GLMNET_CROSS <- cross_validate(container,4,algorithm="GLMNET")
MAXENT_CROSS <- cross_validate(container,4,algorithm="MAXENT")
#TREE_CROSS <- cross_validate(container,4,algorithm="TREE")


xgbmodel <- xgboost(data = dtMatrix,
                 nrounds = 2, objective = "multi:softmax")


