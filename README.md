# Mental-Health-Prediction-in-IT-Company

## INTRODUCTION
Mental health includes our emotional, psychological, and social well-being. It can affect our interactions with the world, work performance and our physical health. Nowadays, mental health topic attracts more and more attentions. A positive attitude towards seeking for treatment is important for people with mental health conditions. There are many factors that may affect this attitude. This dataset includes information about attitude about mental health in the tech workplace, individual's geographic and demographic information, and supports from workplace. We can get insights about which factors would affect the attitude and how we can do to improve the situation. In the United States, approximately 70% of adults with depression are in the workforce. Employees with depression will miss an estimated 35 million workdays a year due mental illness. Those workers experiencing unresolved depression are estimated to encounter a 35% drop in their productivity, costing employers $105 billion dollars each year. So, we can predict the health treatment at their early stage by applying machine learning algorithms on this massive amount of data to extract features that we will extract from datasets. Various machine learning techniques like logistic regression, naïve bayes, support vector machine, k nearest neighbor etc.


## Problem Statement 
Mental health affects your emotional, psychological and social well-being. It affects how we think, feel, and act.  The impact of mental health to an organization can mean an increase of absent days from work and a decrease in productivity and engagement. So, Develop a model that can predict whether a employee seek treatment or not. Identify the key features that lead to mental health problems in tech space.


## Content
This dataset contains the following data:

- Timestamp
- Age
- Gender
- Country
- state: If you live in the United States, which state or territory do you live in?
- self_employed: Are you self-employed?
- family_history: Do you have a family history of mental illness?
- treatment: Have you sought treatment for a mental health condition?
- work_interfere: If you have a mental health condition, do you feel that it interferes with your work?
- no_employees: How many employees does your company or organization have?
- remote_work: Do you work remotely (outside of an office) at least 50% of the time?
- tech_company: Is your employer primarily a tech company/organization?
- benefits: Does your employer provide mental health benefits?
- care_options: Do you know the options for mental health care your employer provides?
- wellness_program: Has your employer ever discussed mental health as part of an employee wellness program?
- seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help?
- anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
- leave: How easy is it for you to take medical leave for a mental health condition?
- mentalhealthconsequence: Do you think that discussing a mental health issue with your employer would have negative consequences?
- physhealthconsequence: Do you think that discussing a physical health issue with your employer would have negative consequences?
- coworkers: Would you be willing to discuss a mental health issue with your coworkers?
- supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)?
- mentalhealthinterview: Would you bring up a mental health issue with a potential employer in an interview?
- physhealthinterview: Would you bring up a physical health issue with a potential employer in an interview?
- mentalvsphysical: Do you feel that your employer takes mental health as seriously as physical health?
- obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
- comments: Any additional notes or comments

## Data Cleaning and Preprocessing
The datasets which were collected from UCI machine learning repository and Kaggle website contain unfiltered data which must be filtered before the final data set can be used to train the model. Also, data has some categorical variables which must be modified into numerical values for which we used Pandas library of Python. In data cleaning step, first we checked whether there are any missing or junk values in the dataset for which we used the isnull() function. Then for handling categorical variables we converted them into numerical variables.

## Machine Learning Algorithms:

### a) Logistic Regression:
Logistic regression is often used a lot of times in machine learning for predicting the likelihood of response attributes when a set of explanatory independent attributes are given. It is used when the target attribute is also known as a dependent variable having categorical values like yes/no or true/false, etc. It’s widely used for solving classification problems. It falls under the category of supervised machine learning. It efficiently solves linear and binary classification problems. It is one of the most commonly used and easy to implement algorithms. It’s a statistical technique to predict classes which are binary. When the target variable has two possible classes in that case it predicts the likelihood of occurrence of the event. In our dataset the target variable is categorical as it has only two classes-yes/no.
                       
### b) Random Forest:
Random Forest is the most famous and it is considered as the best algorithm for machine learning. It is a supervised learning algorithm. To achieve more accurate and consistent prediction, random forest creates several decision trees and combines them together. The major benefit of using it is its ability to solve both regression and classification issues. When building each individual tree, it employs bagging and feature randomness in order to produce an uncorrelated tree forest whose collective forecast has much better accuracy than any individual tree’s prediction. Bagging enhances accuracy of machine learning methods by grouping them together. In this algorithm, during the splitting of nodes it takes only random subset of nodes into an account. When splitting a node, it looks for the best feature from a random group of features rather than the most significant feature. This results into getting better accuracy. It efficiently deals with the huge datasets. It also solves the issue of overfitting in datasets. It works as follows: First, it’ll select random samples from the provided dataset. Next, for every selected sample it’ll create a decision tree and it’ll receive a forecasted result from every created decision tree. Then for each result which was predicted, it’ll perform voting and through voting it will select the best predicted result.

### c) Naive Bayes:  
It is a probabilistic machine learning algorithm which is mainly used in classification problems. It is simple and easy to build. It deals with huge datasets efficiently. It can solve complicated classification problems. The existence of a specific feature in a class is assumed to be independent of the presence of any other feature according to naïve bayes theorem. It’s formula is as follows : P(S|T) = P(T|S) * P(S) / P(T) Here, T is the event to be predicted, S is the class value for an event. This equation. will find out the class in which the expected feature for classification.

### d) Support Vector Machine (SVM):
It is a powerful machine learning algorithm that falls under the category of supervised learning. Many people use SVM to solve both regression and classification problems. The primary role of SVM algorithm is that it separates two classes by creating a line of hyperplanes. Data points which are closest to the hyperplane or points of the data set that, if deleted, would change the position of dividing the hyperplane are known as support vectors. As a result, they might be regarded as essential components of the data set. The margin is the distance between hyperplane and nearest data point from either collection. The goal is to select the hyperplane with the maximum possible margin between it and any point in the training set increasing the likelihood of a new data being properly classified. SVM’s main objective is to find a hyperplane in N-dimensional space which will classify all the data points. The dimension of a hyperplane is actually dependent on the quantity of input features. If input has two features in that case the hyperplane will be a line and two dimensional plane.

### e) K Nearest Neighbor (KNN):
KNN is a supervised machine learning algorithm. It assumes similar objects are nearer to one another. When the parameters are continuous in that case knn is preferred. In this algorithm it classifies objects by predicting their nearest neighbor. It’s simple and easy to implement and also has high speed because of which it is preferred over the other algorithms when it comes to solving classification problems. The algorithm classifies whether or not the employee has health problem by taking the health treatment dataset as an input. It takes input parameters like age, Gender, etc. and classify person with health treatment. Algorithm takes following steps :- 
Step 1:  Select the value for K.
Step 2 : Find the Euclidean distance of K no. of neighbors.
Step 3 : Based on calculated distance, select the K nearest neighbors in the training data which are nearest to
              unknown data points. 
Step 4 : Calculate no. of data points in each category among these K neighbors.
 Step 5 : Assign new data points to the category which has the maximum no. of neighbors.
Step 6 : Stop.


![image](https://user-images.githubusercontent.com/108256699/208118150-bd10bef6-5b0d-47f1-b98c-580055e4df1e.png)

## Implementation Steps:
As we already discussed in the methodology section about some of the implementation details. So, the language used in this project is Python programming. We’re running python code in anaconda navigator’s Jupyter notebook. Jupyter notebook is much faster than Python IDE tools like PyCharm or Visual studio for implementing ML algorithms. The advantage of Jupyter notebook is that while writing code, it’s really helpful for Data visualization and plotting some graphs like histogram and heatmap of correlated matrices. Let’s revise implementation steps : 
a) Dataset collection.
b) Importing Libraries : Numpy, Pandas, Scikit-learn, warnings, Matplotlib and Seaborn libraries were used.
c) Exploratory data analysis : For getting more insights about data.
d) Data cleaning and preprocessing : Checked for null and junk values using isnull() and isna().sum() functions of python. In Preprocessing phase, we did feature     engineering on our dataset. As we converted categorical variables into numerical variables using function of Pandas library. Both our datasets contains some  categorical variables.
e) Model selection : We first separated X’s from y’s. X’s are features or input variables of our datasets and y’s are dependent or target variables which are crucial for predicting treatment. Then using by the importing model_selection function of the sklearn library, we splitted our X’s and y’s into train and test split using train_test_split() function of sklearn. We splitted 70% of our data for training and 30% for testing.
f) Applied ML models and created a confusion matrix of all models.
g) Deployment of the model which gave the best accuracy.
