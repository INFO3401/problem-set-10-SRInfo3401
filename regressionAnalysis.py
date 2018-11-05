#Monday:
#Submit the following functions as part of a file called regressionAnalysis.py. You will use this data to make some predictions about the nutritional aspects of various popular Halloween candies. Your code will contain three objects:
#sources: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html/https://dziganto.github.io/classes/data%20science/linear%20regression/machine%20learning/object-oriented%20programming/python/Understanding-Object-Oriented-Programming-Through-Machine-Learning/

import csv
import pandas as pd
import numpy as np
import parser
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import matplotlib
import matplotlib.pyplot as plt

#Part (a) - AnalysisData, which will have, at a minimum, attributes called dataset (which holds the parsed dataset) and variables (which will hold a list containing the indexes for all of the variables in your data). 
#AnalysisData-dataset(hold parsed data)/variable(hold list containing indexes for all variables in data)
#in self set (int val-1/0, dict={}, lists=[], string="")

class AnalysisData:
    def __init__(self):
        self.dataset = []
        self.variables = []
        
    def parserFile(self, candy_file):
        self.dataset = pd.read_csv(candy_file)
        for column in self.dataset.columns.values:
            if column != "competitorname":
                self.variables.append(column)
            
            #(class example code)
            #if (self.dataset == "csv"):
                #reader = csv.reader(open(candy_file))
                #for row in reader:
                    #self.variables.append(row)       
            #else:
                #self.data = open(candy_file).read()

#Part (b) - LinearAnalysis, which will contain your functions for doing linear regression and have at a minimum attributes called bestX (which holds the best X predictor for your data), targetY (which holds the index to the target dependent variable), and fit (which will hold how well bestX predicts your target variable).
#LinearAnalysis-bestx(holds best X predictor for data)/targetY(holds the index to the target dependent variable)(reference the variable describing whether or not a candy is chocolate)/fit (hold how well bestX predicts target variable)
#LinearAnalysis object = try to predict the amount of sugar in the candy
#Part 2(incorporation) Create a function to initialize a LinearAnalysis object that takes a targetY as its input parameter.

class LinearAnalysis:
    def __init__(self, data_targetY):
        self.bestX = ""
        self.targetY = data_targetY
        self.fit = ""
        
#Part 3(incorporation) Add a function to the LinearAnalysis object called runSimpleAnalysis. This function should take in an AnalysisData object as a parameter and should use this object to compute which variable best predicts how much sugar a candy contains using a linear regression. 
#Print the variable name and the resulting fit (use LaTeX: R^2 R 2  to report the fit). Make sure your best predictor is NOT the same as the targetY variable.
        
    def runSimpleAnalysis(self, data):
        top_sugar_variable = ""
        top_r2 = -1
        
        for column in data.variables:
            if column != self.targetY:
                data_variable = data.dataset[column].values
#ValueError: Expected 2D array, got 1D array instead:Reshape data.
                data_variable = data_variable.reshape(len(data_variable),1)
                
                regr = LinearRegression()
                regr.fit(data_variable, data.dataset[self.targetY])
                variable_prediction = regr.predict(data_variable)
                #(<dep var values<sugarcontent>, <predictedvalues>)
                r_score = r2_score(data.dataset[self.targetY],variable_prediction)
                if r_score > top_r2:
                    top_r2 = r_score
                    top_sugar_variable = column
        self.bestX = top_sugar_variable
        print(top_sugar_variable, top_r2)
        

#Part (c) - LogisticAnalysis, which will contain your functions for doing logistic regression and have at a minimum attributes called bestX (which holds the best X predictor for your data), targetY (which holds the index to the target dependent variable), and fit (which will hold how well bestX predicts your target variable).
#LogisticAnalysis-bestx(holds best X predictor for data)/targetY(hold the index to target dependent variable)/fit (hold how well bestX predicts target variable)
#LogisticAnalysis object = predict whether or not the candy is chocolate.
#Part 2(incorporation)Create the same function for LogisticAnalysis.

class LogisticAnalysis:
    def __init__(self, data_targetY):
        self.bestX = ""
        self.targetY = data_targetY
        self.fit = ""
        
        
    def runSimpleAnalysis1(self, data):
        top_sugar_variable = ""
        top_r2 = -1
        
        for column in data.variables:
            if column != self.targetY:
                data_variable = data.dataset[column].values
                data_variable = data_variable.reshape(len(data_variable),1)
                
                regr = LinearRegression()
                regr.fit(data_variable, data.dataset[self.targetY])
                variable_prediction = regr.predict(data_variable)
                r_score = r2_score(data.dataset[self.targetY],variable_prediction)
                if r_score > top_r2:
                    top_r2 = r_score
                    top_sugar_variable = column
        self.bestX = top_sugar_variable
        print(top_sugar_variable, top_r2)
        

#1. Using the candy-data.csv file in the repo, populate an AnalysisData object that will hold the data you'll use for today's problem set. You should read in the data from the CSV, store the data in the dataset variable, and initialize the xs (in your variables attribute) and targetY variables appropriately. targetY should reference the variable describing whether or not a candy is chocolate.
#code from class
#dataParser = Parser("csv")
#dataParser.parseFile("candy-data.csv")
#print(dataParser.data)

candy_analysis = AnalysisData()
candy_analysis.parserFile('candy-data.csv')

#2. Create a function to initialize a LinearAnalysis object that takes a targetY as its input parameter. Create the same function for LogisticAnalysis. Note that you will use the LinearAnalysis object to try to predict the amount of sugar in the candy and the LogisticAnalysis object to predict whether or not the candy is chocolate.
#(incorporated into problem set)


#Wednesday & Friday:

#3. Add a function to the LinearAnalysis object called runSimpleAnalysis. This function should take in an AnalysisData object as a parameter and should use this object to compute which variable best predicts how much sugar a candy contains using a linear regression. Print the variable name and the resulting fit (use LaTeX: R^2 R 2  to report the fit). Make sure your best predictor is NOT the same as the targetY variable.
#(incorporated into problem set)

candy_data_analysis = LinearAnalysis('sugarpercent')
candy_data_analysis.runSimpleAnalysis(candy_analysis)