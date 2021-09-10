# Loan-Repayment-Prediction-with-Random-Forest-classifier
This is the code for predictng the repayment of Laon amount off of features of the borrowers (data from LendingClub).

# Overview
For this project I will be exploring publicly available data from LendingClub.com. Lending Club connects people who need money (borrowers) with people who have money (investors). I am trying to create a model that will help predict this.

Lending club had a very interesting year in 2016. On May 9, 2016, The CEO resigned from his position following an internal investigation that found a violation of the company's business practices had occurred. This data is from before they even went public.

I will use lending data from 2007-2011 and be trying to classify and predict whether or not the borrower paid back their loan in full. 
You can download the data from https://www.kaggle.com/imsparsh/lending-club-loan-dataset-2007-2011

Here are what the columns represent:

* credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
* purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
* int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
* installment: The monthly installments owed by the borrower if the loan is funded.
* log.annual.inc: The natural log of the self-reported annual income of the borrower.
* dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
* fico: The FICO credit score of the borrower.
* days.with.cr.line: The number of days the borrower has had a credit line.
* revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
* revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
* inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
* delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
* pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).
* not.fully.paid: Indicates whether loan was repayed or not.

# Usage
1. Loan Repayment prediction with Random Forest Classifier.ipynb is jupyter notebook which contains classifier model.
2. Loan_data.csv is a CSV file which contains data of 9578 customers.
3. Python code.py file contain the source code of the classifier.
