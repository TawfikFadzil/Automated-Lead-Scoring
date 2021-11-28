### Predictive Lead Scoring using Logistic Regression with Stepwise Selections

library(ggplot2)
library(glmnet)

# Importing data
lead <- read.csv("LeadsXEducation.csv")
summary(lead)

#Convert Response variables (Integer to Factor)
lead[lead$Converted == 0,]$Converted <- "Lost"
lead[lead$Converted == 1,]$Converted <- "Won"
lead$Converted <- as.factor(lead$Converted)

#Converting Categorical Variables to Factor
lead$Lead_Origin <- as.factor(lead$Lead_Origin)
lead$Lead_Source <- as.factor(lead$Lead_Source)
lead$Last_Activity <- as.factor(lead$Last_Activity)
lead$Current_Occupation <- as.factor(lead$Current_Occupation)
lead$Free_Copy <- as.factor(lead$Free_Copy)

#Converting Ordinal to Factor
lead[lead$Lead_Quality == 1,]$Lead_Quality <- "Worst"
lead[lead$Lead_Quality == 2,]$Lead_Quality <- "Low"
lead[lead$Lead_Quality == 3,]$Lead_Quality <- "NotSure"
lead[lead$Lead_Quality == 4,]$Lead_Quality <- "Mightbe"
lead[lead$Lead_Quality == 5,]$Lead_Quality <- "High"
lead$Lead_Quality <- as.factor(lead$Lead_Quality)

lead[lead$Activity_Index == 1,]$Activity_Index <- "Low"
lead[lead$Activity_Index == 2,]$Activity_Index <- "Medium"
lead[lead$Activity_Index == 3,]$Activity_Index <- "High"
lead$Activity_Index <- as.factor(lead$Activity_Index)

lead[lead$Profile_Index == 1,]$Profile_Index <- "Low"
lead[lead$Profile_Index == 2,]$Profile_Index <- "Medium"
lead[lead$Profile_Index == 3,]$Profile_Index <- "High"
lead$Profile_Index <- as.factor(lead$Profile_Index)

#Plotting boxplot of continuous variable before winsorizing
boxplot(lead$Total_Visits)
boxplot(lead$Total_Time_Website)
boxplot(lead$Page_Views_Per_Visit)

# Create a Function to Winsorize Data
winsor <- function(x, multiplier) {
  if(length(multiplier) != 1 || multiplier <= 0) {
    stop("bad value for 'multiplier'")}
  
  quartile1 = summary(x)[2] # Calculate lower quartile
  quartile3 = summary(x)[5] # Calculate upper quartile
  iqrange = IQR(x) # Calculate interquartile range
  
  y <- x
  boundary1 = quartile1 - (iqrange * multiplier)
  boundary2 = quartile3 + (iqrange * multiplier)
  
  y[ y < boundary1 ] <- boundary1
  y[ y > boundary2 ] <- boundary2
  y
}

#Winsorizing data for total visits
lead$Total_Visits <- winsor(lead$Total_Visits, 1.5)
lead$Page_Views_Per_Visit <- winsor(lead$Page_Views_Per_Visit, 1.5)

#Boxplot after winsorizing
with(lead, boxplot(Total_Visits))
with(lead,boxplot(Page_Views_Per_Visit))
##No more outliers. Heavily right skewed


#Split Training and Test Data
set.seed(1711)
subset <- sample(nrow(lead), nrow(lead) * 0.8)
train_lead <- lead[subset,]
test_lead <- lead[-subset,]

#Define intercept-only model
lead.null <- glm(Converted~1, data=train_lead, family=binomial)
summary(lead.null)

#Define model with all predictors
lead.full <- glm(Converted~., data=train_lead, family=binomial)
summary(lead.full)

#Perform forward stepwise regression
library(MASS)
stepwise.lead <- step(lead.null, direction='forward', scope=formula(lead.full), trace=0)
stepwise.lead$anova

#Perform backward stepwise regression
backward.lead <- step(lead.full, direction='backward', scope=formula(lead.full), trace=0)
backward.lead$anova

# Predict on test dataset for lead full
library(caret)
prob.full <- predict(lead.full, newdata=test_lead, type="response")
pred.full <- ifelse(prob.full>0.5, "Won","Lost")
converted.test <- test_lead$Converted
pred.full <- as.factor(pred.full)
confusionMatrix(pred.full, test_lead$Converted)

# Predict for stepwise
prob.step <- predict(stepwise.lead, newdata=test_lead, type="response")
pred.step <- ifelse(prob.step>0.5, "Won","Lost")
pred.step <- as.factor(pred.step)
confusionMatrix(pred.step, test_lead$Converted)

#Finding AUC and Best Cutoff Value for full model
library(pROC)
roc.full = roc(test_lead$Converted ~ prob.full)
as.numeric(roc.full$auc)
coords(roc.full, x="best", input="threshold", best.method="closest.topleft")

#Finding AUC and Best Cutoff Value for stepwise model
roc.step = roc(test_lead$Converted ~ prob.step)
as.numeric(roc.step$auc)
coords(roc.step, x="best", input="threshold", best.method="youden")

#Plot ROC Curves for full model and stepwise model
plot(roc.full, asp=NA, legacy.axes=TRUE, col="red", lty=2)
plot(roc.step, add=TRUE, col="blue", lty=3)
## Add Legend
legend("bottomright", c("Full Model", "Stepwise Model"), lty=2:3, 
       col = c("red", "blue"), bty="n", cex=0.8)
