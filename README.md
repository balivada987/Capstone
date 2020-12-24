# Overview of The Project:

The primary objective is to deploy and consume ml based  webservice. I was using a dataset to predict 
DEATH_EVENT  of a person based on certain columns  such as age,anaemia,creatinine_phosphokinase,
diabetes etc. After performing some preprocessing a logistic regression model has been implemented and performed 
hyperparameter tuning of paramenters C and number of iterations . Since the model accuracy was near to 80 % an 
auto ml model has been implemented which provided an accuracy of nearly 86 % . 



# Overview of Dataset:
1. The dataset is uploaded from the system using upload command and fetched the dataset from datastore from path data/heart_failure_clinical_records_dataset.csv
2. The dependent variable in the  dataset is IS_DEATH to predict whether a person will die or not based on certain set of columns.
3. "data/heart_failure_clinical_records_dataset.csv" is data source.
![alt text](https://github.com/balivada987/Capstone/blob/main/DatasetFilePath.PNG)
![alt text](https://github.com/balivada987/Capstone/blob/main/Dataset%20screenshot.PNG)




# Implemented Logistic Regression and tuned hyper parameters such as C and numberof iterations  using HyperConfig 
Hyperparameter tuning is done on logistic regression tuning hyperparameters such as  C and n=number of iterations

![alt text](https://github.com/balivada987/Capstone/blob/main/Hyper%201.PNG)
c = regularization regulator
n = number of iterations

## Termination policy used is Bandit Policy and RandomParameter Sampling is done
![alt text](https://github.com/balivada987/Capstone/blob/main/Hyper%202.PNG)
![alt text](https://github.com/balivada987/Capstone/blob/main/Hyper%203.PNG)
 Therea are two main parameters that were tune C and number of iterations  C is tuned in the range(0.00001,0.001,0.01,0.1,1,10,100,1000,10000)
 number of iterations are tuned in the range (10, 200)
 ![alt text](https://github.com/balivada987/Capstone/blob/main/HypeParameterC_iter.PNG)
 
 ![alt text](https://github.com/balivada987/Capstone/blob/main/HyperParametersRunDetails.PNG)
 
 
 The best run has C= 0.1 and number of iterations are 38. C is the regularization parameter to get better accuracy and reduce overfit error.
![alt text](https://github.com/balivada987/Capstone/blob/main/Hyper%204.PNG)

# Best Run Run Details Widget displaying accuracy of 0.8 and regularization strength of 0.1 and Run ID

![alt text](https://github.com/balivada987/Capstone/blob/main/Hyper%205.PNG)

# Hyperparameter model Registration
![alt text](https://github.com/balivada987/Capstone/blob/main/HyperParameterRegisterModel.PNG)
# Since the accuracy of the above model is 80 % Auto ML model has been used and obtained an accuracy of 86.6 %  using Voting Ensemble model


## Auto ML
AutoML Configuration:
1. Since the dependent variable is binary  the task="classification"
2. Metric considered for model evaluation is "accuracy"
3. Dataset is passed for training
4. Building all the models in default so that we can have an overview which model is doing better
5. no. of cross validations considered are 3

In this case I am selecting default case in allowed_models  rather if we want to use some specific models we can use edit to required models
Here I considered metrics as accuracy we can also consider other metrics such as AUC, Recall, Precision etc.
![alt text](https://github.com/balivada987/Capstone/blob/main/AutoML1.png?raw=true)
![alt text](https://github.com/balivada987/Capstone/blob/main/AutoML%202.png?raw=true)
![alt text](https://github.com/balivada987/Capstone/blob/main/AutoML%203.png?raw=true)
# Register AutoML model
![alt text](https://github.com/balivada987/Capstone/blob/main/AutoMLModelRegistration.PNG)


 ![alt text](https://github.com/balivada987/Capstone/blob/main/AutoML%203.png?raw=true)

## Environment is fetched from best run environment
Following Packages were installed in the the system :
![alt text](https://github.com/balivada987/Capstone/blob/main/DependenciesScreenshot.PNG)
# Json file of Environment : https://github.com/balivada987/Capstone/blob/main/azureml_environment.json
## File containing Environment Details
# Environment Screenshot
![alt text](https://github.com/balivada987/Capstone/blob/main/AutoML%204.png?raw=true)
![alt text](https://github.com/balivada987/Capstone/blob/main/AutoML5.png?raw=true)


# Model being registered Model being deployed
A file containing the environment details

The model is deployed as a webservice and can be consumed by senting a json input.
Here we are getting [1,1] as result

# Model is deployed successfully and status of webservice is displaying as Healthy
The automl model has been registered and deployed in the same  environment as that of development environment.
The automl model has been deployed as webservice and tested it by sending a json input using post method.
![alt text](https://github.com/balivada987/Capstone/blob/main/AutoML6.png?raw=true)
#  Inference request being sent to the deployed model and service has been deleted. 
Later the webservice is deleted and the compute target is deleted
![alt text](https://github.com/balivada987/Capstone/blob/main/AutoML%207.png?raw=true)


## Screencast Link 
https://www.youtube.com/embed/1DkGPQXDv2M?feature=oembed


# Future Improvements
1. Perform oversampling or undersampling of dataset to fetch a balanced approad.
2. Perform standardization of all the variables and after that implement the model.
3. Implement Pipeline approach and parallelize pipelines to improve the model perfomance
4. Schedule the pipelines using Schedule to reduce manual execution
5. Feature Engineering canbe done on the dataset using certain techniques such as StepAIC, PCA etc
