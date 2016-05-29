library(nnet)
library(ggplot2)
library(reshape2)
library(maxent)
library(RTextTools)
library(xgboost)
library(randomForest)
library(e1071)
library(tm)

#clean up the r environment.
rm(list = ls())
#@home working directory
setwd("/Users/thorosm2002/Dropbox/Rcode/MLinR")

Afile <- 'DEFdry.csv'



rd <- read.csv2(Afile, sep=",", stringsAsFactors=FALSE,header = TRUE)
rd <- rd[,c("ITEMDESC","CCPNODOT","AMTPAID","STNDQUAN","STNDUNIT")]

rd$WEIGHT <- paste0(rd$STNDQUAN," " , as.character(rd$STNDUNIT))

rd <- na.omit(rd)


unique(rd$WEIGHT)

keptdata <- rd[! (is.na(rd$ITEMDESC) | rd$WEIGHT=="" | 
                    rd$ITEMDESC==""  | is.na(rd$WEIGHT) |
                    is.na(as.numeric(rd$CCPNODOT)) )   , ]


keptdata <- head(keptdata,8000)
testdata <-  tail(keptdata,2000)
traindata<- head(keptdata,6000)

dtMatrix <- create_matrix(cbind(keptdata["ITEMDESC"],keptdata["WEIGHT"],keptdata["CCPNODOT"],keptdata["AMTPAID"]),
                          language="english",removeNumbers=FALSE,stemWords=FALSE,removeSparseTerms = .99999, weighting=tm::weightTfIdf)

# Configure the training data
container <- create_container(dtMatrix, keptdata$WEIGHT, trainSize=1:6000, testSize = 6001:8000, virgin=FALSE)


svmmodel <- train_model(container,"SVM")

write.svm(svmmodel, svm.file = "svm-classifier.svm", scale.file = "svm-classifier.scale")
#maxentmodel <- train_model(container,"MAXENT")


results <- classify_model(container,svmmodel)
#results <- classify_model(container,maxentmodel)

#analytics<- create_analytics(container , results )

# how did we go? 
#cross validation
SVM_CROSS <- cross_validate(container,4,algorithm="SVM")

#GLMNET_CROSS <- cross_validate(container,4,algorithm="GLMNET")
MAXENT_CROSS <- cross_validate(container,4,algorithm="MAXENT")
#TREE_CROSS <- cross_validate(container,4,algorithm="TREE")


total <- cbind(results,testdata)
total$TF <- ifelse(total$WEIGHT==total$SVM_LABEL,1,0)
total$TFsure <-ifelse((total$WEIGHT==total$SVM_LABEL) && (total$SVM_PROB>0.1) ,1,0)
hist(total$TF)





# 
# total <- cbind(results,testdata)
# total$TF <- ifelse(total$WEIGHT==total$SVM_LABEL,1,0)
# hist(total$TF)
# 
# MEtotal <- cbind(results,testdata)
# MEtotal$TF <- ifelse(MEtotal$CCPNODOT==MEtotal$MAXENTROPY_LABEL,1,0)
# hist(MEtotal$TF)




other <- tail(rd,100)




#### error in Rtexttools code! need to run this once!
trace("create_matrix", edit=T)
# Configure the training data
predMatrix <- create_matrix(cbind(other["ITEMDESC"],other["CCPNODOT"],other["AMTPAID"]), originalMatrix=dtMatrix)
# create the corresponding container

predictionContainer <- create_container(predMatrix, labels=rep(0,100), testSize=1:100, virgin=FALSE)
mresults <- classify_model(predictionContainer, svmmodel)
mresults


total <- cbind(mresults,other)
total$TF <- ifelse(total$WEIGHT==total$SVM_LABEL,1,0)
hist(total$TF)




