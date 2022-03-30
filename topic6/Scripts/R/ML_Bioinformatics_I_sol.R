############# ML for Bioinformatics session ###################
###############################################################

# Aim and data for the practical ####

# This tutorial will build a XGBoost classification model to distinguish 
# Alzheimer’s patients from healthy controls using blood-based gene 
# expression profiling.

# The data we are going to work with is an Alzheimer’s Disease (AD) data set,
# where blood samples from 145 AD and 104 healthy patients were expression 
# profiled on the Illumina platform using the HumanHT-12 v3.0 expression 
# chip. We will analyse 5364 genes. The data you will need for this workshop are:

# “expression_data.RDS” - This contains the expression data
# “phenotype_data.RDS” - This contains the phenotype data
# “RFE_results.RDS” - This contains the output from the Recursive Feature 
#                     Elimination (RFE) process that we will not run today 
#                     but will use the results from.

# Packages installation ####

install.packages("ggplot2", dependencies = T)

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("sva")

BiocManager::install("massiR")

install.packages("caret", dependencies = T)

install.packages("xgboost", dependencies = T)

install.packages("reshape2", dependencies = T)

# Packages loading ####

library(ggplot2)
library(sva)
library(massiR)
library(caret)
library(xgboost)
library(reshape2)

# DATA LOADING ####
# Set the working directory if you want, and read data

setwd("/Users/raquel/Downloads/")
raw_expression_data<-readRDS("/Users/raquel/Downloads/expression_data.RDS")
phenotype_data<-readRDS("/Users/raquel/Downloads/phenotype_data.RDS")


# DATA EXPLORATION ####
# Exploring phenotype data 

class(phenotype_data)
dim(phenotype_data)
colnames(phenotype_data)
rownames(phenotype_data)
head(phenotype_data)
table(phenotype_data$Diagnosis)
pie(table(phenotype_data$Diagnosis))

# exploring gender by diagnosis
table(phenotype_data$Gender, phenotype_data$Diagnosis)
# specify table to look at in "data". followed by what columns you want to evaluated in "aes".
ggplot(data=phenotype_data, aes(x=Diagnosis, fill=Gender)) +
  # This tells ggplot to do a bar plot
  geom_bar() +
  # add title to plot
  ggtitle("Distribution of gender in AD and healthy control subjects") +
  # centre title 
  theme(plot.title = element_text(hjust = 0.5)) 

# Exploring expression data

# The expression data has 249 columns, we can view the first 5 columns

head(raw_expression_data)[,1:5]

# Basic plots of the expression data.
# Plot of the distribution of gene expression across samples using a simple 
# boxplot. This does a boxplot of every gene (column), however, we need to 
# do a boxplot of every sample (row). Therefore we transpose the data 
# frame using the t() function, before providing it to the boxplot function.

boxplot(t(raw_expression_data))

# interpretation: The X-axis represents samples and the Y-axis is the summary of all 
# the genes for that sample. From the boxplot it can be clearly seen all samples
# have been normalised, as all the boxplots for all samples are fairly equal.

# Check if there is phenotype information for all samples in the 
# expression table. The sample IDs are the row names in both the 
# phenotype_data and raw_expression_data. We can use the R == operator to 
# see if they are the same.

# check if rownames are the same

rownames(phenotype_data)==rownames(raw_expression_data)

# check if all are the same

all(rownames(phenotype_data)==rownames(raw_expression_data))==T

# see if we have any duplicated sample IDs

anyDuplicated(rownames(phenotype_data))




# DATA QUALITY CONTROL (QC) ####

# 1.Exploring batch effects using unsupervised learning ####

# Principal Component Analysis (PCA) is an unsupervised learning technique 
# that is extremely useful in reducing high dimensional datasets while 
# preserving the original data structure and relationships. 

# To apply PCA analysis to our data, we simply use the prcomp() function. 
# The principal components are calculated for each column, therefore, we 
# need to transpose the data once again. We can then apply the summary 
# function to look a the PCA object.

# run PCA analysis
pca_data<-prcomp(t(raw_expression_data))

# summary
summary(pca_data)

#Q1: How many PCAs there are? How much variation is explained by PC1 and PC2?

#AQ1: We have 249 principal components, which we will call PC1-249. 
# The first PC always captures the most variation, followed by the next PC, 
# and so on. The summary says PC1 explains a massive 93% "Proportion of Variance" of 
# the total variation and PC2 explains only 1.6%.

# Inspecting PCs

names(pca_data)
head(pca_data$rotation)[,1:5]

# plot the PC’s against one another to visualise the data. 
# First extract the PC1 and PC2 column for each sample. 
# Together, they capture over 94% of the variation in the data, 
# and can indicate which samples are similar.
# We then check the sample names are still intact with the phenotype data, 
# and then we plot using the ggplot() function. 
# We can colour each sample according to any variable in the phenotype data.

# extract PC1 and PC2
PC1_plot<-pca_data$rotation[,1:2]

# check sample names are the same 
all(rownames(PC1_plot)==rownames(phenotype_data))==T

# add phenotype information to PC information
PC1_plot<-cbind(PC1_plot, phenotype_data)

# use PC1 and PC2 to inspect batch effects.
# plot the PC1 vs PC2 and colour by batch

ggplot(data=PC1_plot, aes(x=PC1, y=PC2, col=as.character(Batch))) +
  # scatterplot
  geom_point() +
  # add title to plot
  ggtitle("PC1 vs PC2 by gender") +
  # centre title 
  theme(plot.title = element_text(hjust = 0.5)) 


# Q2: From the PCA plots, do you think there is unwanted variation (batch effects)?
  
# AQ2: Yes, there exist batch effects in the data. It is prominent when the data 
# is coloured by the batch from which the data has been assayed. 

# Remove these batch effects 

# This can be done by using an R package called SVA that uses empirical 
# Bayes to adjust the data.

# create a model matrix of batch
combat_model = model.matrix(~1, data=phenotype_data[4])

# apply the ComBat function to the data, using parametric empirical Bayesian adjustments.
# ComBat will adjust the data for the 3 known batches, and return the data as an expression matrix, 
# with the same dimensions as your original dataset.
expr_adjusted = ComBat(dat=t(raw_expression_data), batch=phenotype_data$Batch, mod=combat_model, par.prior=TRUE, prior.plots=FALSE)

# convert matrix to data.frame and transpose
expr_adjusted<-as.data.frame(t(expr_adjusted))


# 2.Exploring gender mismatch using unsupervised learning ####

# Females have two X chromosomes (XX) while males have an X and Y chromosome (XY). 
# Therefore, we can use the expression of genes on the Y chromosome to identify 
# female and male samples. 

# We can extract these Y-chromosome genes from the expression 
# data, perform PCA and colour by gender. This should cluster samples according to 
# expression differences between female and male samples based on the Y chromosome. 
# We can then identify any sample clustering with the wrong sex.

# the package massiR is specific to gene expression data and predcits gender based 
# on genetics 

# we will supply a handful of Y chromosome genes.

# create list of Y probes
y_chromosome_genes<-c("5616", "5687", "8284", "9086", "100133941", "6192")

# extract these from gene expresion data
y_expr<-expr_adjusted[,colnames(expr_adjusted) %in% y_chromosome_genes]

#transpose
y_expr<-t(y_expr)

#change to dataframe
y_expr<-as.data.frame(y_expr)

#check 
head(y_expr)[1:5]
dim(y_expr)

# analyse with massiR
massiR_results <- massi_cluster(y_expr)

# get predicted genders
predicted_gender<-(massiR_results$massi.results)[c(1,5)]

# check
head(predicted_gender$ID)

# move ID column to row names and remove ID column
rownames(predicted_gender)<-predicted_gender$ID
predicted_gender$ID<-NULL

# check and merge with our phenotype gender
all(rownames(phenotype_data)==rownames(predicted_gender))==T
predicted_gender$clinical_gender<-phenotype_data$Gender

# check
head(predicted_gender)

# extract gender mismatches
gender_mismatches<-subset(predicted_gender, sex!=clinical_gender)

#check
gender_mismatches

# Q3: There are any gender mismatches?
  
# AQ3: There are four gender mismatches, which can easily be samples that were 
# unintentionally swapped during laboratory work. 

# Remove these samples from our adjusted gene expression dataframe and our 
# phenotype table to create a clean dataset to produce our classifcation model from. 
# Save these clean datasets using the saveRDS() function.

# clean phenotype data
phenotype_data_clean<-subset(phenotype_data, !(rownames(phenotype_data) %in% rownames(gender_mismatches)))

#clean expression data
expr_adjusted_clean<-subset(expr_adjusted, !(rownames(expr_adjusted) %in% rownames(gender_mismatches)))

# check dim before and after - phenotype
dim(phenotype_data)
dim(phenotype_data_clean)

# check dim before and after - phenotype
dim(expr_adjusted)
dim(expr_adjusted_clean)

# save as RDS file - will save in the working directory we first set

saveRDS(phenotype_data_clean, file="phenotype_data_clean.RDS")
saveRDS(expr_adjusted_clean, file="expr_adjusted_clean.RDS")




# MODEL DEVELOPMENT ####

# merge the phenotype data with the expression data
#merge the data by rownames using the merge function
expr_pheno_data<-merge(phenotype_data_clean, expr_adjusted_clean, by="row.names")

# check
head(expr_pheno_data)[1:10]

# move rowname column to rownames and remove unwanted columns(Tissue_source, Batch)
rownames(expr_pheno_data)<-expr_pheno_data$Row.names
expr_pheno_data$Row.names<-NULL
expr_pheno_data$Tissue_source<-NULL
expr_pheno_data$Batch<-NULL

#check
head(head(expr_pheno_data)[1:10])

# 1. Split the data ####

# split for training (75%) and testing (25%) purposes

# assign a seed number to an object and set seed number - please make sure you use the same seed number as specified below so that we all get the same result
seed_number<-1234
set.seed(seed_number)

#create partition:
trainIndex <- createDataPartition(expr_pheno_data$Diagnosis, p = .75, 
                                  list = FALSE, 
                                  times = 1)

#extract index from data
training_data<-expr_pheno_data[trainIndex,]
testing_data<-expr_pheno_data[-trainIndex,]


# check size
dim(training_data)
dim(testing_data)

# check diagnosis distribution
table(training_data$Diagnosis)
table(testing_data$Diagnosis)

# plot
pie(table(training_data$Diagnosis))
pie(table(testing_data$Diagnosis))

# check diagnosis by gender distribution
table(training_data$Diagnosis, training_data$Gender)
table(testing_data$Diagnosis, testing_data$Gender)

# Q4: Is the distribution of cases vs controls balanced in both 
# the training and testing set? How you thing this problem can be approached?

# AQ4: No, diagnosis is unbalanced in both training and testing sets. 
# This is a very popular problem. One way to tackle this is adding more weights 
# to the smaller class during the model building process. This can be easily 
# addressed by the XGBoost scale_pos_weight function. We will incorporate 
# this function into the model building process.

# 2. Format the data for XGBoost ####
# XGBoost needs the outcome (case/control) to be numeric numbers (0/1). 
# Recode the outcome ("Diagnosis") and gender (males to 1 and females to 2).

# recode diagnosis to training and testing set
training_data[training_data$Diagnosis=="case",1]<-0
training_data[training_data$Diagnosis=="control",1]<-1
testing_data[testing_data$Diagnosis=="case",1]<-0
testing_data[testing_data$Diagnosis=="control",1]<-1

# check
table(training_data$Diagnosis)
table(testing_data$Diagnosis)

# recode gender
training_data[training_data$Gender=="male",2]<-1
training_data[training_data$Gender=="female",2]<-2
testing_data[testing_data$Gender=="male",2]<-1
testing_data[testing_data$Gender=="female",2]<-2

#check
table(training_data$Gender)
table(testing_data$Gender)

# We can explore the data structure to check that all variables are numeric,
# as needed fpr XGBoost, using the str() function and change if necessary.

# current structure of first 5 columns
str(training_data, list.len=5)
str(testing_data, list.len=5)

# change diagnosis column to numeric - XGBoost doesn't like character variables
training_data$Diagnosis<-as.numeric(training_data$Diagnosis)
testing_data$Diagnosis<-as.numeric(testing_data$Diagnosis)

# change gender column to numeric - XGBoost doesn't like character variables
training_data$Gender<-as.numeric(training_data$Gender)
testing_data$Gender<-as.numeric(testing_data$Gender)

# check
str(training_data, list.len=5)
str(testing_data, list.len=5)

# convert both the training and testing set to XGBoost format known as DMatrix. 
# This is an internal data structure that is used by XGBoost which is 
# optimized for both memory efficiency and training speed. 
# This function requires two arguments:
  
# 1. Data - all columns in our data (minus the outcome column)
# 2. Label - outcome - which is our "Diagnosis" column

training_data_xgb<-xgb.DMatrix(data = as.matrix(training_data[2:ncol(training_data)]), 
                               label=training_data$Diagnosis)

testing_data_xgb<-xgb.DMatrix(as.matrix(testing_data[2:ncol(testing_data)]), 
                              label=testing_data$Diagnosis)

# 3. Initial parameter settings ####

# The overall XGBoost parameters have been divided into 3 categories by XGBoost authors:
  
# a) General Parameters: Guide the overall functioning
# b) Booster Parameters: Guide the individual booster (tree/regression) at each step
# c) Learning Task Parameters: Guide the optimization performed

# a) General Parameters: These define the overall functionality of XGBoost.

#booster [default=gbtree]:
  #Select the type of model to run at each iteration. It has 2 options:
  #gbtree: tree-based models (This is what we will be using)
#gblinear: linear models
#silent [default=0]:
  #Silent mode is activated is set to 1, i.e. no running messages will be printed.
  #It’s generally good to keep it 0 as the messages might help in understanding the model.
#nthread [default to maximum number of threads available if not set]
  #This is used for parallel processing and number of cores in the system should be entered
  #If you wish to run on all cores, value should not be entered and algorithm will detect automatically

# b) Booster Parameters: These are often referred to hyperparameters. 
# Although there are 2 types of boosters, we will consider only tree boosters 
# here because it always outperforms the linear booster and thus the later is 
# rarely used.

#eta [default=0.3]
 #Analogous to learning rate in GBM
 #Makes the model more robust by shrinking the weights on each step
#min_child_weight [default=1]
 #Defines the minimum sum of weights of all observations required in a child
 #Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree
 #Too high values can lead to under-fitting
#max_depth [default=6]
 #The maximum depth of a tree
 #Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample
#gamma [default=0]
 #A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split
 #Makes the algorithm conservative. The values can vary depending on the loss function
#subsample [default=1]
 #Denotes the fraction of observations to be randomly samples for each tree
 #Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting
#colsample_bytree [default=1]
 #Denotes the fraction of columns to be randomly samples for each tree
#lambda [default=1]
 #L2 regularization term on weights (analogous to Ridge regression)
 #This used to handle the regularization part of XGBoost.
#alpha [default=0]
 #L1 regularization term on weight (analogous to Lasso regression)
 #Can be used in case of very high dimensionality so that the algorithm runs faster when implemented
#scale_pos_weight [default=1]
 #A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence
 #These parameters should all be tuned using cross-validation. For this tutorial we will only tune a hand full of parameters.

# c) Learning Task Parameters

#These parameters are used to define the optimization objective the metric to be calculated at each step.

#objective [default=reg:linear]. This defines the loss function to be minimized. Mostly used values are:
  #binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
  #multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
  #you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
  #multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.
#eval_metric [ default according to objective ]
 #The metric to be used for validation data.
 #The default values are rmse for regression and error for classification.
 #Typical values are:
  #rmse – root mean square error
  #mae – mean absolute error
  #logloss – negative log-likelihood
  #error – Binary classification error rate (0.5 threshold)
  #merror – Multiclass classification error rate
  #mlogloss – Multiclass logloss
  #auc: Area under the curve
#seed [default=0]
  #The random number seed.
  #Can be used for generating reproducible results and also for parameter tuning.

# parameters (a) and (b) should all be tuned using cross-validation. 
# For this tutorial we will only tune a hand full of parameters.

# name of param list
params <- list(
  booster = "gbtree", # This is the default tree algorithm
  objective = "binary:logistic", # This is telling the model we are doing a logistic regression for binary classification, returns predicted probability (not class)
  eta=0.2,  # learning rate
  max_depth=6, # number of trees
  min_child_weight=1, # requires sum of weight in child
  subsample=1, # number of samples
  colsample_bytree=1, # number of genes
  scale_pos_weight=table(training_data$Diagnosis)[1]/table(training_data$Diagnosis)[2] # weight - # Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative cases) / sum(positive cases)
)

# 4. Create an initial model ####

# use the xgb.cv() to create an initial model. 
# xgb.cv function is used to identify the optimum parameter settings.
# This function uses a cross validation (cv) process at each iteration. 
# This allows us to observe and choose the best tuning parameters based on the 
# CV output.

# All the tuning parameters are adjusted until achieving the optimum test-auc score.

#set the model to stop building once an improvement to the test-auc score is not achieved in 20 
#iterations using the early_stopping_rounds function. 

# set the seed number
set.seed(seed_number)

# create model
xgb_model_1_cv <- xgb.cv(params = params, # This is the default parameter list we made earlier
                         data = training_data_xgb, # This is our training data in XGBoost format
                         nfold= 10, # number of cv folds
                         nrounds=1000, # increase if plot does not converge
                         verbose=T, # lets us see results in real-time
                         showsd=T, # show SD of error
                         eval_metric="auc",
                         stratified=T, # Stratification is the process of rearranging the data to ensure each fold is a good representative of the whole. For example in a binary classification problem where each class comprises 50% of the data, it is best to arrange the data such that in every fold, each class comprises around half the instances.stratification is generally a better scheme, both in terms of bias and variance, when compared to regular cross-validation.
                         early_stopping_rounds = 20 # stop if no improvement to model seen
)

# Q4: Once the model stops building, what is the optimum number of iterations returned?, 
# What does happen after this iteration?

# AQ4: The model stops building at iteration 26. Any more than this and the test-auc starts to decrease, 
# which is a sign of over-fit. 

# optimum iteration is stored in - we will need to use this later
xgb_model_1_cv$best_iteration

# Build the model using the best iteration identified

# set the seed number
set.seed(seed_number)

# build model using default param
xgb_model_1 <- xgboost(
  params = params,
  data = training_data_xgb,
  nrounds = xgb_model_1_cv$best_iteration # set to optimum number identifed earlier
)

# When building the actual model, we will only have the internal train-error auc 
# scores. We can explore the predictive features using the xgb.importance() 
# function.

# xgb.importance returns: 
# (i) Features - names of the features used in the model
# (ii) Gain - represents the fractional contribution of each feature to the model based on the total gain of this feature’s splits. 
#             A Higher percentage means a more important predictive feature.
# (iii) Cover - metric of the number of observation related to this feature
# (iv) Frequency - percentage representing the relative number of times a feature have been used in trees.

model_1_importance <- xgb.importance(feature_names = names(training_data[2:ncol(training_data)]), model = xgb_model_1)

# check
head(model_1_importance)

#Q5: How many predictive features is our mode using from the total number of features available? is gender included?
#AQ5: 163 predictive features.

#We can plot the predictive features to see which ones are more important based on their Gain value:

ggplot(model_1_importance, aes(x=reorder(Feature,Gain), y=Gain, fill=Gain)) + 
  geom_bar(stat="identity", position="dodge") + 
  coord_flip() +
  ylab("Relative importance")+
  xlab("Gene") +
  ggtitle("XGboost Variable Importance") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_gradient(low="red", high="blue") +
  theme(axis.text.y=element_blank())

#Q6: On how many genes depend the model?

#AQ6: We can see the model is heavily dependent on one gene, but the remaining genes 
# also contribute to the model. However, as we go down the list, the genes give 
# minimal improvement to the model. These are extremely weak predictive. 

# We can apply recursive feature elimination to build a model (using xgb.cv) 
# on the 163 predictive features only, then we remove the least predictive 
# feature and rebuild the model and re-evaluate the test-auc. 
# This is an extremely compute/time intensive process as we will 
# be effectively building and evaluating 163 models!

# The output of this analysis was saved as RFE_results. 
# We will load this in the next chunk and continue to find the optimum number 
# of features to use and the optimum number of nrounds.

# load - choose the location where you have this data stored on your laptop
RFE_results<-readRDS("RFE_results.RDS")

# check
class(RFE_results)
names(RFE_results)

#The RFE_results is a list object, which contains multiple xgb.cv objects saved in 
#different slots. Each slot has been named as the number of features it contains.

#We can access each xgb output by using the dollar sign, and can further access the 
#xgb log using another dollar sign. We can search each evaluation log and find the 
#one which contains the highest test-auc score. Then we find the optimum number of 
#features that was used to achieve this test-auc score and what the optimum number 
#of nrounds was used.

# find highest auc
RFE_results_highest_auc<-as.data.frame(unlist(lapply(RFE_results, function(x) max(x$evaluation_log[,4]))))
head(RFE_results_highest_auc)

# nrounds
RFE_results_highest_auc_nr<-as.data.frame(unlist(lapply(RFE_results, function(x) x$best_iteration)))
head(RFE_results_highest_auc_nr)

# merge with nrounds
RFE_results_highest_auc<-merge(RFE_results_highest_auc, RFE_results_highest_auc_nr, by="row.names")

# add name
names(RFE_results_highest_auc)<-c("Number_of_features", "test_auc_mean", "nrounds")

# order - highest auc score 1st
RFE_results_highest_auc<-RFE_results_highest_auc[order(-RFE_results_highest_auc$test_auc_mean),]
head(RFE_results_highest_auc)

# extract best auc log
RFE_results_highest_auc_log<-RFE_results[[eval(RFE_results_highest_auc[1,1])]]$evaluation_log

# plot optimum feature
plot(RFE_results_highest_auc_log$test_auc_mean, type = 'l', col="red", main="Test-auc scores")

#Q7: what was the optimum number of features, the test-auc score and nrounds?
#AQ7: This can be check at head(RFE_results_highest_auc). The optimum number of features is 120, with a test-auc score of 0.9686 and nrounds of 94.
#There is a big improvement from our initial model. Initial test AUC can be checked at 
#xgb_model_1_cv.

# create new training + testing data with optimum feature
RFE_training_data<-training_data[head(model_1_importance, 120)$Feature]
RFE_training_data_xgb<-xgb.DMatrix(as.matrix(RFE_training_data), label=training_data$Diagnosis)

RFE_testing_data<-testing_data[head(model_1_importance, 120)$Feature]
RFE_testing_data_xgb<-xgb.DMatrix(as.matrix(RFE_testing_data), label=testing_data$Diagnosis)

# 5. Hyperparamteter tuning ####

#We will tune the following hyperparameters:
#max_depth
#min_child_weight

# The same process would apply to all the possible hyperparameters (gamma, subsample, 
# colsample, colsample_bytree, regularizations, eta, etc..). You could even create 
# a massive grid with every possible combination of hyperparameter thresholds and 
# build a model and tune. However you will end up building 1000’s of models and 
# this will be extremely time/compute consuming.

# (1) we will tune the hyperparameter max_depth. The default value we used was 6. 
# We will try different ranges from 2:10, stepping in intervals of 1. i.e will test 2,
# 3,4,5,6,7,8,10 as the max_depth threshold. This value can be infinite, but the 
# optimum number is generally around 6.

# copy default param - we will make changes to the copy in the loop
max_depth_param<-params

# parameters for loop
minimum_max_depth<-2 # minimum number
maximum_max_depth<-10 # maximum number
interval_max_depth<-1 # interval to change by
max_depth_counter=1 # counter

# empty list for results
max_depth_tune_results<-list()

for (x in seq(minimum_max_depth, maximum_max_depth, interval_max_depth)) {
  # counter for interation
  y=max_depth_counter
  # print round
  print(paste("round: ", y))
  # change reg param to x
  max_depth_param$max_depth<-x
  # run xgboost cv
  seeds = set.seed(seed_number)
  max_depth_tune_results[[y]] <- xgb.cv(params = max_depth_param,
                                        data = RFE_training_data_xgb,
                                        nfold= 10,
                                        nrounds=1000, 
                                        verbose=T,
                                        showsd=T,
                                        eval_metric="auc", 
                                        stratified=T,
                                        early_stopping_rounds = 20)
  # add name
  names(max_depth_tune_results)[y]<-x
  # change y
  max_depth_counter=max_depth_counter+1
  
}




# find highest auc
max_depth_tune_highest_auc<-as.data.frame(unlist(lapply(max_depth_tune_results, function(x) max(x$evaluation_log[,4]))))
max_depth_tune_highest_auc

# nrounds
max_depth_tune_highest_auc_nr<-as.data.frame(unlist(lapply(max_depth_tune_results, function(x) x$best_iteration)))
max_depth_tune_highest_auc_nr

# merge with nrounds
max_depth_tune_highest_auc<-merge(max_depth_tune_highest_auc, max_depth_tune_highest_auc_nr, by="row.names")

# add name
names(max_depth_tune_highest_auc)<-c("Number_of_features", "test_auc_mean", "nrounds")

# order - highest auc score 1st
max_depth_tune_highest_auc<-max_depth_tune_highest_auc[order(-max_depth_tune_highest_auc$test_auc_mean),]
max_depth_tune_highest_auc

# extract best auc log
max_depth_tune_highest_auc_log<-max_depth_tune_results[[eval(max_depth_tune_highest_auc[1,1])]]$evaluation_log

# plot optimum feature
plot(max_depth_tune_highest_auc_log$test_auc_mean, type = 'l', col="red", main="Test-auc scores")

# compare to previous tuning

# max test-auc in initial model
max(xgb_model_1_cv$evaluation_log$test_auc_mean)

# max test-auc in RFE model
max(RFE_results_highest_auc_log$test_auc_mean)

# max test-auc in max_depth model
max(max_depth_tune_highest_auc_log$test_auc_mean)


# plot
ggplot() + 
  geom_line(data = xgb_model_1_cv$evaluation_log, aes(x = iter , y = test_auc_mean), color = "blue") +
  geom_line(data = RFE_results_highest_auc_log, aes(x = iter, y = test_auc_mean), color = "blue") +
  geom_line(data = max_depth_tune_highest_auc_log, aes(x = iter, y = test_auc_mean), color = "red") +
  ggtitle("Comparison of test-auc scores")


# Q8: On how much have we marginally improved the test-auc score?
# AQ8: It improved by 0.0018, by changing the max_depth from 6 to 2. 
# We can now permanently change our max_depth to 2 in ur default parameter object:
  
# check default parameter for max depth
params

# change to 2
params$max_depth<-2

# check change has occurred
params

# (2) we will tune min_child_weight

# We can use the new max_depth threshold and then tune for min_child_weight 
# from 2:10, stepping in intervals of 1.

# copy default param- we will make changes to the copy in the loop
min_child_weight_param<-params

# parameters for loop
minimum_min_child_weight<-1 # minimum number
maximum_min_child_weight<-10 # maximum number
interval_min_child_weight<-1 # interval to change by
min_child_weight_counter=1 # counter

# empty list for results
min_child_weight_tune_results<-list()

for (x in seq(minimum_min_child_weight, maximum_min_child_weight, interval_min_child_weight)) {
  # counter for interation
  y=min_child_weight_counter
  # print round
  print(paste("round: ", y))
  # change reg param to x
  min_child_weight_param$min_child_weight<-x
  # run xgboost cv
  seeds = set.seed(seed_number)
  min_child_weight_tune_results[[y]] <- xgb.cv(params = min_child_weight_param,
                                               data = RFE_training_data_xgb,
                                               nfold= 10,
                                               nrounds=1000, 
                                               verbose=T,
                                               showsd=T,
                                               eval_metric="auc", 
                                               stratified=T,
                                               early_stopping_rounds = 20)
  # add name
  names(min_child_weight_tune_results)[y]<-x
  # change y
  min_child_weight_counter=min_child_weight_counter+1
  
}




# find highest auc
min_child_weight_tune_highest_auc<-as.data.frame(unlist(lapply(min_child_weight_tune_results, function(x) max(x$evaluation_log[,4]))))
min_child_weight_tune_highest_auc

# nrounds
min_child_weight_tune_highest_auc_nr<-as.data.frame(unlist(lapply(min_child_weight_tune_results, function(x) x$best_iteration)))
min_child_weight_tune_highest_auc_nr

# merge with nrounds
min_child_weight_tune_highest_auc<-merge(min_child_weight_tune_highest_auc, min_child_weight_tune_highest_auc_nr, by="row.names")

# add name
names(min_child_weight_tune_highest_auc)<-c("Number_of_features", "test_auc_mean", "nrounds")

# order - highest auc score 1st
min_child_weight_tune_highest_auc<-min_child_weight_tune_highest_auc[order(-min_child_weight_tune_highest_auc$test_auc_mean),]
min_child_weight_tune_highest_auc

# extract best auc log
min_child_weight_tune_highest_auc_log<-min_child_weight_tune_results[[eval(min_child_weight_tune_highest_auc[1,1])]]$evaluation_log

# plot optimum feature
plot(min_child_weight_tune_highest_auc_log$test_auc_mean, type = 'l', col="red", main="Test-auc scores")

# compare to previous tuning

# plot
ggplot() + 
  geom_line(data = xgb_model_1_cv$evaluation_log, aes(x = iter , y = test_auc_mean), color = "blue") +
  geom_line(data = RFE_results_highest_auc_log, aes(x = iter, y = test_auc_mean), color = "blue") +
  geom_line(data = max_depth_tune_highest_auc_log, aes(x = iter, y = test_auc_mean), color = "blue") +
  geom_line(data = min_child_weight_tune_highest_auc_log, aes(x = iter, y = test_auc_mean), color = "red") +
  ggtitle("Comparison of test-auc scores")

# max test-auc in initial model
max(xgb_model_1_cv$evaluation_log$test_auc_mean)

# max test-auc in RFE model
max(RFE_results_highest_auc_log$test_auc_mean)

# max test-auc in max_depth model
max(max_depth_tune_highest_auc_log$test_auc_mean)

# max test-auc in min_child_weight model
max(min_child_weight_tune_highest_auc_log$test_auc_mean)


#Q9: Was there any improvement to the test-auc score when tuning the min_child_weight?
#AQ9: There was no improvement to the test-auc score when tuning the min_child_weight 
#parameter, as the optimum min_child_weight is 1, which was our default parameter. 
# This can be very common with many tuning parameters.

#With this data, the colsample_bytree, subsample, and eta are all optimum values. 
#Any changes to the default value result in the test-auc decreasing.

#Hyperparamteter FINE tuning - eta

# tuned from range 0.01-0.3, in steps of 0.01 (equivalent to 30 models) gave 
# 0.2 as the optimum number. eta is one of those parameters which we can fine-tune, 
# i.e we can set the threshold to an unlimited decimal place. 
# We can further fine-tune eta by using an even smaller number. 
# The etas that I had tried ranged from 0.01-0.3, in steps of 0.01. 
# Therefore, it included 0.19, 0.2 and 0.21. We can try smaller ranges 
# between these values, i.e range from 0.19-0.21 in steps of 0.001

# copy default param
eta_param<-params

# parameters for loop
minimum_eta<-0.19 # minimum number
maximum_eta<-0.21 # maximum number
interval_eta<-0.001 # interval to change by
eta_counter=1 # counter

# empty list for results
eta_tune_results<-list()

for (x in seq(minimum_eta, maximum_eta, interval_eta)) {
  # counter for interation
  y=eta_counter
  # print round
  print(paste("round: ", y))
  # change reg param to x
  eta_param$eta<-x
  # run xgboost cv
  seeds = set.seed(seed_number)
  eta_tune_results[[y]] <- xgb.cv(params = eta_param,
                                  data = RFE_training_data_xgb,
                                  nfold= 10,
                                  nrounds=1000, 
                                  verbose=T,
                                  showsd=T,
                                  eval_metric="auc", 
                                  stratified=T,
                                  early_stopping_rounds = 20)
  # add name
  names(eta_tune_results)[y]<-x
  # change y
  eta_counter=eta_counter+1
  
}




# find highest auc
eta_tune_highest_auc<-as.data.frame(unlist(lapply(eta_tune_results, function(x) max(x$evaluation_log[,4]))))
eta_tune_highest_auc

# nrounds
eta_tune_highest_auc_nr<-as.data.frame(unlist(lapply(eta_tune_results, function(x) x$best_iteration)))
eta_tune_highest_auc_nr

# merge with nrounds
eta_tune_highest_auc<-merge(eta_tune_highest_auc, eta_tune_highest_auc_nr, by="row.names")

# add name
names(eta_tune_highest_auc)<-c("Number_of_features", "test_auc_mean", "nrounds")

# order - highest auc score 1st
eta_tune_highest_auc<-eta_tune_highest_auc[order(-eta_tune_highest_auc$test_auc_mean),]
eta_tune_highest_auc

# extract best auc log
eta_tune_highest_auc_log<-eta_tune_results[[eval(eta_tune_highest_auc[1,1])]]$evaluation_log

# plot optimum feature
plot(eta_tune_highest_auc_log$test_auc_mean, type = 'l', col="red", main="Test-auc scores")

# compare to previous tuning

# plot
ggplot() + 
  geom_line(data = xgb_model_1_cv$evaluation_log, aes(x = iter , y = test_auc_mean), color = "blue") +
  geom_line(data = RFE_results_highest_auc_log, aes(x = iter, y = test_auc_mean), color = "blue") +
  geom_line(data = max_depth_tune_highest_auc_log, aes(x = iter, y = test_auc_mean), color = "blue") +
  geom_line(data = eta_tune_highest_auc_log, aes(x = iter, y = test_auc_mean), color = "red") +
  ggtitle("Comparison of test-auc scores")

# max test-auc in initial model
max(xgb_model_1_cv$evaluation_log$test_auc_mean)

# max test-auc in RFE model
max(RFE_results_highest_auc_log$test_auc_mean)

# max test-auc in max_depth model
max(max_depth_tune_highest_auc_log$test_auc_mean)

# max test-auc in eta model
max(eta_tune_highest_auc_log$test_auc_mean)

#Q10: Is there an improvement in test-auc after finr-tuning eta?
#AQ10: We have found an improvement to the test-auc by fine-tuning 
#the eta parameter to 0.207 and using a nround of 87. 

#We can further fine-tune this parameter to see if we can improve the test-auc. 
#However, for this tutorial we will stop here.
#Change the eta in our default parameter settings.

# check default parameter for max depth
params

# change to 0.207
params$eta<-0.207

# check change has occurred
params


# 6. Final model ####

# Use the updated params list along with the optimum number of nround (87) we 
# found in the eta step to build our final XGBoost model.

# set the seed number
set.seed(seed_number)

# build model using default param
xgb_model_final <- xgboost(
  params = params,
  data = RFE_training_data_xgb,
  nrounds = 87 # set to optimum number identifed earlier
)


# MODEL EVALUATION ####
# Test our model on the holdout dataset (testing set) which has not been seen 
# by the model.

# predict training data
training_set_pred<-as.numeric(predict(xgb_model_final, RFE_training_data_xgb) > 0.5)

# check
training_set_pred

# make sense of the data using caret package - does all the confusion matrix for you
training_set_pred_results<-caret::confusionMatrix(as.factor(training_set_pred), as.factor(training_data$Diagnosis))

# check
training_set_pred_results

# Unsurprisingly, we have predicted the diagnosis of all our training data correctly. 
# But let’s see how the model works on the hold out data. 
# The testing set which the model hasn’t seen yet.

# predict testing data  and provide predcition as 0 or 1
raw_testing_set_pred<-predict(xgb_model_final, RFE_testing_data_xgb)

# change to 0 or 1
testing_set_pred<-as.numeric(raw_testing_set_pred>0.5)

# check
testing_set_pred

# make sense of the data using caret package - does all the confusion matrix for you
testing_set_pred_results<-caret::confusionMatrix(as.factor(testing_set_pred), as.factor(testing_data$Diagnosis))

# check
testing_set_pred_results

#Q11: How does the model works? What performance measures would be better to look at? 
#AQ11: The model works fairly well. As we have an unbalanced dataset, we should 
#not concentrate on accuracy alone. Useful metrics are:
  
#sensitivity - portion of true positives (TP/(TP + FN))
#specificity - portion of true negatives (TN/(TN + FP))
#Positive Predictive Value (PPV) - the probability that following a positive test result, that individual will truly have that specific disease. (TP/(TP + FP))
#Negative Predictive Value (NPV) - the probability that following a negative test result, that individual will truly not have that specific disease (TN/(TN+FN))
#For a clinical setting, some research has suggested a good classification model 
#should have at least 90% PPV and 90% NPV. So our model is not quite there, 
#but its not surprising when we look at the biological question 
#at hand - blood-based classification model to distinguish AD from healthy control 
#(something that does not exist yet!). 
#Furthermore, the model should really be evaluated in an independent dataset.



# IDENTIFY PREDICTIVE FEATURES ####
#Since our model can relatively predict some of the samples in the testing set well 
#(better than random), the predictive features of the model may be of some 
#biological value. We can extract the list of these genes and do some further 
#investigation. We use the xgb.importance() function to extract the predictive 
#features of the model.

# extract features
final_model_importance <- xgb.importance(feature_names = names(RFE_training_data[2:ncol(RFE_training_data)]), model = xgb_model_final)

# check
head(final_model_importance)
dim(final_model_importance)

# plot
ggplot(final_model_importance, aes(x=reorder(Feature,Gain), y=Gain, fill=Gain)) + 
  geom_bar(stat="identity", position="dodge") + 
  coord_flip()+
  ylab("Relative importance") +
  xlab("Gene")+
  ggtitle("Final Model Variable Importance") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_gradient(low="red", high="blue") +
  theme(axis.text.y=element_blank())

# extract 68 genes
predictive_genes<-final_model_importance$Feature

# save as list
write(predictive_genes, sep="\n", file="predictive_genes.txt")


# AN EXTRA: BIOLOGICAL INTERPRETATION USING PATHWAY ANALYSIS ####

#We can perform a pathway analysis using gene set enrichment on our 68 predictive 
# genes. Pathway analysis will basically tell us which biological pathway is 
# enriched/depleted by our predictive genes. This can be easily done using 
# the web-based ConsensusPathDB.

# Navigate to this web-site (http://cpdb.molgen.mpg.de/) and then:
  
#Click on “gene set analysis”
#Click on “over-representation analysis”
#Under the “or upload a file containing gene / protein identifier”, click on “Choose File” and select your “predictive_genes.txt” file
#Click the drop down menu under “gene/protein identifier” and select “Entrez gene”
#Click proceed
#Select the “pathways as defined by pathway databases” checkbox
#Click “Find enrichment sets”
#The top most enriched pathways are all related to the immune system. 
#The top hit is the Immune system, with a p-value of 4.59e-59, and all 68 
#predictive genes are involved in this pathway. Therefore, our classification 
#model uses 68 predictive genes that are enriched in the immune system to 
#determine if a sample is a case (AD) or control (healthy). 
#The immune system has been implicated in many research papers relating to 
#Alzheimer’s disease, but it has also been implicated in many other diseases.