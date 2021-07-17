#设置路径&导入包
setwd('D:/project/Clinical')
library(caret)
library(do)
library(e1071)
library(survival)
library(survminer)
library(pROC)
#读取和数据预处理
luad=read.table('nationwidechildrens.org_clinical_patient_luad.txt',sep='\t',header = T)
luad=luad[-c(1,2),]
luad=as.matrix(luad)
luad[which(luad=='[Not Available]')]=NA
luad=as.data.frame(luad)
write.csv(luad,file = 'luad.csv')
#用python对数据进行分类
#读取测试数据和训练数据
train=read.csv('train_file.csv',stringsAsFactors=FALSE)
test=read.csv('test_file.csv',stringsAsFactors=FALSE)
train=as.data.frame(train)
test=as.data.frame(test)
#挑选变量
train=train[,c(3,6,13,27,29,32,31,34,47,93,91,106)]
test=test[,c(3,6,13,27,29,32,31,34,47,93,91,106)]
write.csv(train,file = 'train_vital.csv')
write.csv(test,file = 'test_vital.csv')
#转换数据类型
train2=read.csv('train_vital.csv',stringsAsFactors=FALSE)
test2=read.csv('test_vital.csv',stringsAsFactors=FALSE)
train2[2:6]<-lapply(train2[2:6],factor)
train2[12:13]<-lapply(train2[12:13],factor)
test2[2:6]<-lapply(test2[2:6],factor)
test2[12:13]<-lapply(test2[12:13],factor)
#准备模型
model_form<-vital_status~gender+race+tobacco_smoking_history_indicator+
  tobacco_smoking_pack_years_smoked+carbon_monoxide_diffusion_dlco+
  age_at_initial_pathologic_diagnosis+ajcc_pathologic_tumor_stage+icd_10
modelControl <- trainControl(method="repeatedcv",number=5,
                             repeats=5,allowParallel=TRUE)
#逻辑回归模型
logic=train(model_form,
            data=train2,
            method="glm",
            trControl=modelControl)
test2$predict=predict(logic,test2)
table(test2$vital_status,test2$predict)
#ROC曲线
pre=predict(logic,test2)
roc=matrix(0,nrow=103,ncol=2)
roc[,1]=pre
roc[,2]=t(test2$vital_status)
names(roc)=c("pre","obs")
roc1=roc[order(roc[,1]),]
n=nrow(roc1)
tpr=fpr=rep(0,n)
modelroc=roc(roc1[,2],roc1[,1])
plot(modelroc,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="#2E9FDF",print.thres=TRUE)

#逻辑回归模型2
fit<-glm(model_form,data = train2,binomial(link='logit'),control=list(maxit=100))
test2$predict2=predict(fit,test2)
table(test2$vital_status,test2$predict2)

#随机森林模型
rf_Model <- train(model_form,
                  data=train2,
                  method="rf",
                  ntrees=500,
                  trControl=modelControl)
test2$rfPrediction <- predict(rf_Model,test2)
table(test2$vital_status,test2$rfPrediction)


model_Comparison <- 
  resamples(list(
    LogisticRegression=logic,
    RandomForest=rf_Model
  ))
summary(model_Comparison)
bwplot(model_Comparison,layout=c(2,1))

#cox回归
cox <- coxph(Surv(time, status) ~gender+tobacco_smoking_history_indicator+
               tobacco_smoking_pack_years_smoked+
               age_at_initial_pathologic_diagnosis+ajcc_pathologic_tumor_stage, data = train2)
summary(cox)
ggsurvplot(survfit(cox), data = train2, palette = "#2E9FDF", 
           ggtheme = theme_minimal(), legend = "none")
ggforest(cox,main="hazard ratio",data=train2)
test2$precox=predict(cox,test2)
