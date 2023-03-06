# HackerRank Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

def to_float(x):
    return [float(i) for i in x]
    
f,n = map(int, input().split())

model = Pipeline([('poly',PolynomialFeatures(degree=3)),
('linear',LinearRegression(fit_intercept=False))])

train = [to_float(input().split()) for i in range(n)]
x = [row[:-1] for row in train]
y = [row[-1:] for row in train]
model.fit(x,y)

n = int(input())

predictions = [to_float(input().split()) for i in range(n)]

for i in model.predict(predictions):
    print(i[0])
