# Implementation from kaggle kernel : https://www.kaggle.com/sazid28/home-loan-prediction/notebook
from pathlib import Path
import argparse
import os
import pandas as pd
from tensorflow import gfile
import pickle
import warnings
warnings.filterwarnings("ignore")

# Handle the input argument
parser = argparse.ArgumentParser(description='My program description')

parser.add_argument('--output-x-path', type=str, help='')
parser.add_argument('--output-x-path-file', type=str, help='')

parser.add_argument('--output-y-path', type=str, help='')
parser.add_argument('--output-y-path-file', type=str, help='')
args = parser.parse_args()


# For this example, the data is in the container so simply read it
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Dependents'].replace('3+', 3, inplace=True)
test['Dependents'].replace('3+', 3, inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)

train["Gender"].fillna(train["Gender"].mode()[0], inplace=True)
train["Married"].fillna(train["Married"].mode()[0], inplace=True)
train['Dependents'].fillna(train["Dependents"].mode()[0], inplace=True)
train["Self_Employed"].fillna(train["Self_Employed"].mode()[0], inplace=True)
train["Credit_History"].fillna(train["Credit_History"].mode()[0], inplace=True)

train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mode()[0], inplace=True)
train["LoanAmount"].fillna(train["LoanAmount"].median(), inplace=True)

test["Gender"].fillna(test["Gender"].mode()[0], inplace=True)
test['Dependents'].fillna(test["Dependents"].mode()[0], inplace=True)
test["Self_Employed"].fillna(test["Self_Employed"].mode()[0], inplace=True)
test["Loan_Amount_Term"].fillna(test["Loan_Amount_Term"].mode()[0], inplace=True)
test["Credit_History"].fillna(test["Credit_History"].mode()[0], inplace=True)
test["LoanAmount"].fillna(test["LoanAmount"].median(), inplace=True)

train["TotalIncome"] = train["ApplicantIncome"]+ train["CoapplicantIncome"]
test["TotalIncome"]=test["ApplicantIncome"]+test["CoapplicantIncome"]

train["EMI"] = train["LoanAmount"]/train["Loan_Amount_Term"]
test["EMI"] = test["LoanAmount"]/test["Loan_Amount_Term"]
train["Balance_Income"] = train["TotalIncome"] - train["EMI"]*1000  # To make the units equal we multiply with 1000
test["Balance_Income"] = test["TotalIncome"] - test["EMI"]

train = train.drop(["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"], axis=1)
test = test.drop(["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"], axis=1)

train=train.drop("Loan_ID",axis=1)
test=test.drop("Loan_ID",axis=1)

X = train.drop("Loan_Status",1)
y = train[["Loan_Status"]]

X = pd.get_dummies(X)

#gfile.MakeDirs(os.path.dirname(args.output_x_path))
#gfile.MakeDirs(os.path.dirname(args.output_y_path))

with gfile.GFile(args.output_x_path, 'w') as output_X:
  pickle.dump(X, output_X)

with gfile.GFile(args.output_y_path, 'w') as output_y:
  pickle.dump(y, output_y)


Path(args.output_x_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_x_path_file).write_text(args.output_x_path)

Path(args.output_y_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_y_path_file).write_text(args.output_y_path)