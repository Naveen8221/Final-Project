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


