#Monday:
#Submit the following functions as part of a file called regressionAnalysis.py. You will use this data to make some predictions about the nutritional aspects of various popular Halloween candies. Your code will contain three objects:
#sources: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html/https://dziganto.github.io/classes/data%20science/linear%20regression/machine%20learning/object-oriented%20programming/python/Understanding-Object-Oriented-Programming-Through-Machine-Learning/ https://pypi.org/project/val/

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
        top_sugar_variable = data.dataset
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
        print('Linear Regression Analysis coefficients: ', regr.coef_)
        print('Linear Regression Analysis intercept: ', regr.intercept_)
        

#Part (c) - LogisticAnalysis, which will contain your functions for doing logistic regression and have at a minimum attributes called bestX (which holds the best X predictor for your data), targetY (which holds the index to the target dependent variable), and fit (which will hold how well bestX predicts your target variable).
#LogisticAnalysis-bestx(holds best X predictor for data)/targetY(hold the index to target dependent variable)/fit (hold how well bestX predicts target variable)
#LogisticAnalysis object = predict whether or not the candy is chocolate.
#Part 2(incorporation)Create the same function for LogisticAnalysis.








#(11/7/18)
#PROBLEM SET 10
#Monday PROBLEM SET 10 (11/7/18)

class LogisticAnalysis:
    def __init__(self, data_targetY):
        self.bestX = ""
        self.targetY = data_targetY
        self.fit = -1
        
    def runSimpleAnalysis1(self, data):
        top_sugar_variable = data.dataset
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
        print('Simple Logistic Regression Analysis coefficients: ', regr.coef_)
        print('Simple Logistic Regression Analysis intercept: ', regr.intercept_)
        
#2 Problem        
    def runMultipleRegression(self, data):
        #for val in data.dataset.columns.values:
            #if val != "competitorname":
                #data_variable = data.dataset[column].values
                #data_variable = data_variable.reshape(len(data_variable),1)
                #self.variables.append(val)
            multi_regr = LogisticRegression()
            mp_r = [val for val in data.variables if val != self.targetY]
            multi_regr.fit(data.dataset[mp_r], data.dataset[self.targetY])

            variable_prediction = multi_regr.predict(data.dataset[mp_r])
            r_score = r2_score(data.dataset[self.targetY],variable_prediction)
            #if r_score > top_r2:
                #top_r2 = r_score
                #top_sugar_variable = column
        #self.bestX = top_sugar_variable
            print("fruity", r_score)
            print('Multiple Regression Analysis coefficients: ', multi_regr.coef_)
            print('Multiple Regression Analysis intercept: ', multi_regr.intercept_)
        
#Monday & Wednesday:
#1. Add a function to the LogisticAnalysis object called runSimpleAnalysis. This function should take in an AnalysisData object as a parameter and should use this object to compute which variable best predicts whether or not a candy is chocolate using logistic regression. Print the variable name and the resulting fit. Do the two functions find the same optimal variable? Which method best fits this data? Make sure your best predictor is NOT the same as the targetY variable.

#(1 TEXT ANSWERS!!) The two functioncs do not find the same optimal varaibale. When running the simple analysis on the linear data it finds the optimal variable is "Price Percent at 0.10870." While running the simple analysis on logistic data the best variable results in "Fruity at 0.55015" The logistic regression method output of .55015 confirms it is best to use when fitting this particular data.

candy_analysis = AnalysisData()
candy_analysis.parserFile("candy-data.csv")

#Linear Analysis(Problem Set 9)
candy_data_analysis = LinearAnalysis("sugarpercent")
candy_data_analysis.runSimpleAnalysis(candy_analysis)
#Logistic Analysis(Problem Set 10)
candy_data_analysis = LogisticAnalysis("chocolate")
candy_data_analysis.runSimpleAnalysis1(candy_analysis)


#2. Add a function to the LogisticAnalysis object called runMultipleRegression. This function should take in an AnalysisData object as a parameter and should use this object to compute a multiple logistic regression using all of the possible independent variables in your dataset to predict whether or not a candy is chocolate (note, you should not use your dependent variable as an independent variable). Print the variable name and resulting fit. In your testing code, create a new LogisticAnalysis object and use it to run this function on your candy data. Compare the outcomes of this and the simple logistic analysis. Which model best fits the data? Why?

candy_data_analysis = LogisticAnalysis("chocolate")
candy_data_analysis.runMultipleRegression(candy_analysis)

#(2 TEXT ANSWERS!!) When comapring the outcomes of the simple logtistic analysis and multiple regression analysis you can clearly see multiple analysis model outperforms simple analysis. You can varify this through the more percice fit by the multiple analysis output value of .76069. The multiple logistic analysis will always outperform simple because the model can now utilize all the data at once instead of running step by step in variables and columns like simple analysis. Having access to a larger porportion of data within a dataset allows multiple analysis to find a more accurate fit for the regresoin output.


#Wednesday:

#3. Write the equations for your linear, logistic, and multiple logistic regressions. Hint: Use the equations from the slides from Monday's lecture to work out what a logistic regression equation might look like. The coef_ and intercept_ attributes of your regression object will help a lot here!(ref:video 22 7:55)

#Linear Regression y = b0 + b1x
#Logistic Regression p = 1/1+e^-(b0+b1x)
#Multiple Regression p = 1/1+e^-(b0+b1x+b2x+b3x+....b11x)

#(Answer-Linear Regression Equation) y = 0.257063291665 + 0.00440378

#(Answer-Logistic Regression Equation) p = 1/1+e^-(-0.650265328323 + 0.02157451x)

#(Answer-Multiple Regression Equation) p = 1/1+e^-(-1.68260553 + -2.52858047 + -0.19697876 + 0.03940308 -0.16539952 + 0.49783674 + -0.47591613 0.81511886 + -0.59971553 + -0.2581028 + 0.3224988 + 0.05387906)


#Output Reference
#pricepercent 0.108706302017
#Linear Regression Analysis coefficients:  [0.00440378]
#Linear Regression Analysis intercept:  0.257063291665

#fruity 0.550150129132
#Simple Logistic Regression Analysis coefficients:  [ 0.02157451]
#Simple Logistic Regression Analysis intercept:  -0.650265328323

#fruity 0.760698198198
#Multiple Regression Analysis coefficients:  [[-2.52858047 -0.19697876  0.03940308 -0.16539952  0.49783674 -0.47591613 0.81511886 -0.59971553 -0.2581028   0.3224988   0.05387906]]
#Multiple Regression Analysis intercept:  [-1.68260553]



#FRIDAY:
#PROBLEM SET 10 - PART 4(ANSWERS BELOW)

#4. Identify the independent variable(s) and its type (e.g., categorical, continuous, or discrete), the dependent variable and its type, and the null hypothesis for each of the following scenarios: 

#(a) What candies contain more sugar, those with caramel or those with chocolate?

#(a-answer)The independent variable is all the different types of candies, which is a categorical type. The dependent variable would be sugar percent, which is a continious type.

#(a-answer) Null hyptohesis would tell us candies with caramel has the same amout of sugar as candies of chocolate. Meaning that there is no significance differnece between sugar percent and the type of candy being chocolate or caramel. Not good variables to use.


#(b) Are there more split ticket voters in blue states or red states? 

#(b-answer) The independent variables are either blue or red states, which is a categorical type. The dependent variable would be the split ticket voters, which is a continious type.

#(b-answer) Null hyptohesis would tell us that red states have the same amout of split ticket voters as blue states.


#(c) Do phones with longer battery life sell at a higher or lower rate than other phones?

#(c-answer) The independent variable is the duration of phone battery life, which is a continious type. The dependent variable is the different phone sales rates, which is a continious type. 

#(c-answer) Null hyptohesis would tell us that phones with short or long battery life durations will have the same phone sales rates.
 
