# Implementation from kaggle kernel : https://www.kaggle.com/sazid28/home-loan-prediction/notebook
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

x_train, x_cv, y_train, y_cv = train_test_split(X, y,
                                             test_size=0.3,
                                             random_state=1)

logistic_model = LogisticRegression(random_state=1)

logistic_model.fit(x_train,y_train)

pred_cv_logistic = logistic_model.predict(x_cv)
score_logistic = accuracy_score(pred_cv_logistic,y_cv)*100