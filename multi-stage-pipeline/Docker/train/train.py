# Implementation from kaggle kernel : https://www.kaggle.com/sazid28/home-loan-prediction/notebook
import argparse
import pickle
from pathlib import Path
import boto
import boto.s3.connection
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_feature(input_x_path_file):
  print (input_x_path_file)
  filename = input_x_path_file

  print (filename)
  for key in _conn.get_bucket('swbot-prod'):
      if key.name == os.path.basename(filename):
          key.get_contents_to_filename(filename)
          print('The file {} downloaded successfully from S3 bucket {}.'.format(filename,'swbot-prod' ))
  with open(input_x_path_file, 'rb') as input_y_file:
    return pickle.loads(input_y_file.read())


def load_label(input_y_path_file):
  filename = input_y_path_file
  for key in _conn.get_bucket('swbot-prod'):
      if key.name == os.path.basename(filename):
          key.get_contents_to_filename(filename)
          print('The file {} downloaded successfully from S3 bucket {}.'.format(filename,'swbot-prod' ))
  with open(input_y_path_file, 'rb') as input_y_file:
    return pickle.loads(input_y_file.read())

parser = argparse.ArgumentParser()
parser.add_argument('--input_x_path_file', type=str, help='')
parser.add_argument('--input_y_path_file', type=str, help='')

parser.add_argument('--output_model', type=str, help='')
parser.add_argument('--output_model_path_file', type=str, help='')

args = parser.parse_args()

_conn = boto.connect_s3(
            aws_access_key_id='78R9PK1L0PMJX2Y367SD',
            aws_secret_access_key='CzTJkKlUBL7YfS6CIveQ700mvJLV1INlXOt5yqGq',
            host='ch-dc-s3-gsn-33.eecloud.nsn-net.net',
            is_secure=False,  # Using False as boto with python3 is having problem with True.
            calling_format=boto.s3.connection.OrdinaryCallingFormat()
        )

X = load_feature(args.input_x_path_file)
y = load_label(args.input_y_path_file)

x_train, x_cv, y_train, y_cv = train_test_split(X, y,
                                             test_size=0.3,
                                             random_state=1)

logistic_model = LogisticRegression(random_state=1, solver='liblinear')

logistic_model.fit(x_train,y_train.values.ravel())

pred_cv_logistic = logistic_model.predict(x_cv)
score_logistic = accuracy_score(pred_cv_logistic,y_cv)*100


with open(args.output_model, 'wb') as output_model:
  pickle.dump(logistic_model, output_model)

Path(args.output_model_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_model_path_file).write_text(args.output_model)
