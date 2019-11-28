
# Import the required library --------------------

library(e1071)         #For Naive Bayes
library(tm)            #For text mining 
library(SnowballC)     #For stemming
library(caret)         #For the Confusion Matrix
library(wordcloud)     #For constructing wordcloud
library(RColorBrewer)  #For colors in word cloud
library(textmineR)     #For text mining
library(textstem)      #For lemmatization
library(stopwords)

# Read the file

setwd("C:/Users/User/Desktop/R Codes/")
getwd()
text <- read.csv("spam.csv")

# Take only 1st 2 columns for analysis

str(text)
#View(text)
text<- text[,1:2]
text$v2<-as.character(text$v2)

# Dimesnions of the dataframe
colnames(text)
nrow(text)

# Proportions of ham and spam messages
prop.table(table(text$v1))




# Creating 3 indicator variables -----------------------------------

phonenums<-c()
weblink<-c()
dollar<-c()
sms<-c()

for (i in c(1:nrow(text)))
{
  phonenums<-append(phonenums,(grepl("call",text[i,"v2"],ignore.case = TRUE)|grepl("care",text[i,"v2"],ignore.case = TRUE)))
  weblink<-append(weblink,(grepl("http",text[i,"v2"],ignore.case = TRUE)|grepl("wwww",text[i,"v2"],ignore.case = TRUE)))
  dollar<-append(dollar,(grepl("€",text[i,"v2"],ignore.case = TRUE)|grepl("£",text[i,"v2"],ignore.case = TRUE)))
  sms<-append(sms,(grepl("sms",text[i,"v2"],ignore.case = TRUE)|grepl("",text[i,"v2"],ignore.case = TRUE)))
}

indicator <- data.frame("phonenums" = phonenums, "weblink" = weblink, "moneysymbol" = dollar, "sms"=sms )
class(indicator)

change<-function(x)
{
  return(as.integer(as.logical(x)))
  
}

indicator<-apply(indicator, MARGIN = 2,change)
indicator<-as.data.frame(indicator)
str(indicator)



#Creating a Volatile corpus and inspecting it ---------------------
sms_corpus <- VCorpus(VectorSource(text$v2))
sms_corpus1<- VectorSource(text$v2)




# creating a DocumentTermMatrix--------------------------------
sms_dtm_main <- DocumentTermMatrix(sms_corpus, control = list(tolower = TRUE,removeNumbers = TRUE,stopwords = TRUE,removePunctuation = TRUE))

sms_dtm<- DocumentTermMatrix(sms_corpus, control = list(tolower = TRUE,removeNumbers = TRUE,stopwords = TRUE,removePunctuation = TRUE,stemming = TRUE))


# Inspecting the term document matrix  
inspect(sms_dtm[1:10,100:130])
dim(sms_dtm)


# Create vector of most frequent words of ham ( frequency > 0.1 percent)-------------------------
threshold <- 0.5
threshold1 <- 0.5

min_freq1 <- round(nrow(sms_dtm[text$v1=="ham",])*(threshold/100),0)
min_freq2 <- round(nrow(sms_dtm[text$v1=="spam",])*(threshold1/100),0)

hams<-sms_dtm[text$v1=="ham",]
spams<-sms_dtm[text$v1=="spam",]

freq_words1 <- findFreqTerms(x =hams ,lowfreq = min_freq1)
freq_words2 <- findFreqTerms(x = spams, lowfreq = min_freq2)

length(freq_words1)
length(freq_words2)

freq_words<-unique(append(freq_words1,freq_words2))


#remove columns starting with www or sms-------------------------

unique_words<-c()

for (i in freq_words)
{
  if(!(grepl("www",i)|grepl("sms",i)))
  {
    unique_words<-append(unique_words,i)
  }}




# Including terms with frequency higher than the threshold  ------------------------
sms_dtm <- sms_dtm[ , unique_words]
dim(sms_dtm)

# Binding the termdocument matrix with indicator variables------------------
sms_df<-as.data.frame(as.matrix(sms_dtm))
sms<-cbind(sms_df,indicator)


#Since Naive Bayes trains on categorical data, the numerical data is converted to categorical data ------------------

convert_values <- function(x) 
{
  x <- ifelse(x > 0, "Yes", "No")
  return(x)
}


sms <- apply(sms, MARGIN = 2,convert_values)
sms<-as.data.frame(sms)
str(sms)


#View(sms)

ncol(sms)
nrow(sms)

# Training the classifer and predicting ---------------------

#Training & Test set
sms_dtm_train <- sms[1:4457, ]
sms_dtm_test <- sms[4458:5572, ]

#Training & Test Label
sms_train_labels <- text[1:4457,"v1"]
sms_test_labels <- text[4458:5572,"v1"]

#Proportion for training & test labels
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

sms_classifier <- naiveBayes(sms_dtm_train, sms_train_labels,laplace = 1)

#Make predictions on test set
sms_test_pred <- predict(sms_classifier, sms_dtm_test)


#_-----------------------------------------------------------------

colnames(sms_dtm_train)[grepl("weblink",colnames(sms_dtm_train))]

#-------------------------------------------------------------


# Viewing the misclassifications -------------------

compare<-cbind(text[4458:5572,],sms_test_pred)
check<-data.frame()

for ( i in (1:nrow(compare)))
  
{
  if(compare[i,1]!=compare[i,3])
  {
    check<-rbind(check,compare[i,])
  }}

nrow(check)
View(check)



#Create confusion matrix---------------------
confusionMatrix(data = sms_test_pred, reference = sms_test_labels,positive = "spam", dnn = c("Prediction", "Actual"))


#-------------------------------------------------------------------------------------------------------------------

# WORD CLOUD VIZUVALIZATION

## Convert tdm to a list of text
dtm2list <- apply(sms_dtm_main, 1, function(x) {paste(rep(names(x), x), collapse=" ")})
dtm2list<-as.matrix(dtm2list)

ham <- dtm2list[text$v1=="ham",]
wordcloud(Corpus(VectorSource(ham)), max.words = 100, colors = brewer.pal(7, "Paired"), random.order = FALSE)

spam <- dtm2list[text$v1=="spam",]
wordcloud(Corpus(VectorSource(spam)), max.words = 200, colors = brewer.pal(7, "Paired"), random.order = FALSE)

### Inferences about the word cloud

# 1. In spam messages, the words used in business and promotions {call,free,reply,now,win,prize,guarantee,txt,cash,contact} 
# appears very frequently

# 2. In Ham messages, the word used in normal chats aappear frequenly





