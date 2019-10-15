# -*- coding: utf-8 -*-
"""income_predictor.ipynb
Coded by Sanjay Chawla
"""

import pandas as pd
import numpy as np
import category_encoders as ce
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define paths
training_file_path = "tcd ml 2019-20 income prediction training (with labels).csv"
validation_file_path = "tcd ml 2019-20 income prediction test (without labels).csv"

# Read training data, identify missing data
missing_values_list = ["unknown", "Unknown", "UNKNOWN", "na", "n/a", "-", "--"]
training_data = pd.read_csv(training_file_path, na_values = missing_values_list)
validation_data = pd.read_csv(validation_file_path, na_values = missing_values_list)

training_data['Income in EUR'] = training_data['Income in EUR'] + 5700

# More data cleanup
def eliminate_numbers(data, column_name):
    row_count=0
    for row in data[column_name]:
        try:
            int(row)
            data.loc[row_count, column_name]=np.nan
        except ValueError:
            pass
        row_count+=1

def preprocess_data(data):
    eliminate_numbers(data, 'Gender')
    eliminate_numbers(data, 'Hair Color')
    eliminate_numbers(data, 'University Degree')
    
    # Data is Item Non-responsive we can apply either Decuctive Imputation or Proper Imputation
    # Year of Record         441 -- ignore
    # Gender               15005 -- impute -- Category
    # Age                    494 -- impute -- Continuous
    # Profession             322 -- ignore 
    # University Degree     7370 -- impute -- Category
    # Hair Color            7926 -- impute -- Category
    
    data['Age'] = data['Age'].fillna(data['Age'].median())
    # TODO: Impute individual columns instead of entire table
    data = data.ffill(axis = 0)
    return data;

ce_ordinal_encoder = ce.OrdinalEncoder(
                                    cols=['University Degree'], 
                                    mapping=[{
                                            'col':'University Degree', 
                                            'mapping':{None:0, 'No':0, 'Bachelor':1, 'Master':2, 'PhD':3}
                                        }]
                                    )
ce_one_hot_encoder_2 = ce.OneHotEncoder(
                                        cols=['Profession'], handle_unknown='ignore'
                                        )
ce_one_hot_encoder = ce.OneHotEncoder(cols=['Gender', 'Country', 'Hair Color'], handle_unknown='ignore')
training_data = preprocess_data(training_data)
training_data = ce_ordinal_encoder.fit_transform(training_data)
training_data = ce_one_hot_encoder_2.fit_transform(training_data)
training_data = ce_one_hot_encoder.fit_transform(training_data)

validation_data = preprocess_data(validation_data)
validation_data = ce_ordinal_encoder.transform(validation_data)
validation_data = ce_one_hot_encoder_2.transform(validation_data)
validation_data = ce_one_hot_encoder.transform(validation_data)

training_target = training_data['Income in EUR']
training_features = training_data.drop(columns=['Instance','Income in EUR'])

train_X, val_X, train_y, val_y = train_test_split(training_features, training_target)

linear_model = LinearRegression()

linear_model.fit(train_X, np.log(train_y))

model_prediction = linear_model.predict(val_X)

print(sqrt(mean_squared_error(val_y, np.exp(model_prediction))))

validation_data.replace([np.inf, -np.inf], np.nan)
validation_data.isnull().sum()
validation_data = validation_data.fillna(value=1, axis=0)

validation_data['Income'] = np.exp(linear_model.predict(validation_data.drop(columns=['Instance','Income'])))

validation_data.to_csv("result_3_3.csv", index=False, columns=["Instance", "Income"])