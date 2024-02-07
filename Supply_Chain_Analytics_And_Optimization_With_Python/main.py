import numpy as np
import dtale
import pandas as pd
from spellchecker import SpellChecker
import peter_romany_module
import re
df = pd.read_csv(r"training_raw_file.csv")
pd.set_option("display.max_rows", None)
pd.set_option("expand_frame_repr", False)
# print(df.head())# understand data(cols and rows)

'''              the most important columns to do the analysis 
ID,Country,Shipment Mode,Scheduled Delivery Date,Delivered to Client Date,Unit of Measure (Per Pack),Line Item Quantity,Line Item Value,Pack Price,Unit Price
'''

'''                                                         Data wrangling
                                preparing raw data for analysis via convert raw data into analysis-ready data.
'''

# Check data types for any type mismatch
# df.info()

'''
After conducting a thorough check for type mismatch within the dataset, these columns contain type mismatch:
So ID column is numerical but does not have measurement unit and no mean to order or perform operations. So must be qualitative data(categorical).(type 1 error)
Scheduled Delivery Date, and Delivered to Client Date columns are numbers but must be datetime.(type 2 error)
Freight Cost (USD) and Weight (Kilograms) columns are numerical but there are some strings.(type 3 error)
'''

# Handle type mismatch
mismatch_features = ['ID']
df[mismatch_features] = df[mismatch_features].astype(str)
df['Scheduled Delivery Date'] = pd.to_datetime(df['Scheduled Delivery Date'])
df['Delivered to Client Date'] = pd.to_datetime(df['Delivered to Client Date'])

#Change type match columns to type mismatch columns to find various types of errors
# df[['Freight Cost (USD)']].astype(dtype={'Freight Cost (USD)':float})

# Check missing data
# col00='Line Item Insurance (USD)'
# print(df[df[col00].isnull()])#row has null
# print(df[df[col00]=="?"])#row has ?
# print(df[df[col00]==""])#row has blank
# print(df[df[col00]==0])#row has 0

'''
After conducting a thorough check for missing data within the dataset, these columns contain missing data:
Shipment Mode(rows contain null), Line Item Value and Pack Price and Unit Price(rows contain 0.0 maybe missing data or an offer)
and Line Item Insurance (USD) (rows contain null and zeros) 
'''

# Handle missing data
df.drop(df[df['Shipment Mode'].isnull()].index,inplace=True)
df.drop(df[df['Line Item Insurance (USD)'].isnull()].index,inplace=True)

# Check for all types of errors that may exist to identify and obtain columns that are error-free.
# peter_romany_module.regex_for_str(df,'Product Group')
# peter_romany_module.regex_for_float(df,'Unit Price')
# peter_romany_module.regex_for_int(df,'Unit of Measure (Per Pack)')
# peter_romany_module.regex_patterns(df,'Scheduled Delivery Date',pattern='[^0-9-]',convert_to_str=True)

'''
After conducting a thorough check for all types of errors within the dataset, there are no columns contain errors.
'''

# Check misspellings
# print(SpellChecker().unknown(df)) # Get all columns that have misspelled
# print(SpellChecker().unknown(df['Country'])) # Check specific column

'''
After conducting a thorough check for misspellings within the dataset, there are no columns contain misspellings.
'''

# Check form, schema and other inconsistent
# match_indices=peter_romany_module.regex_patterns(df,'Shipment Mode','[^"Air""Truck""Air Charter""Ocean"]',return_match_indices=True)
# match_indices=peter_romany_module.regex_patterns(df,'Country','[^"Côte d\'Ivoire"]',return_match_indices=True)

'''
After conducting a thorough check for form, schema, and other inconsistencies within the dataset, all issues have been addressed
and the data appears to be consistent and accurate.
'''

# Check duplicates
# print(df[df.duplicated()])#find duplicate rows across all columns, keep=False or “fisrt”
# print(df[df.duplicated(['ID'])]) #find duplicate rows across specific column
# print(df[df.duplicated(['Shipment Mode', 'Scheduled Delivery Date','Delivered to Client Date'])]) #find duplicate rows across specific columns

'''
After conducting a thorough check for duplicates within the dataset, all issues have been addressed
and the repetition seems meaningful and precise.
'''

'''                           Data mining and analysis   or   Exploratory Data Analysis (EDA)
                                        extracting knowledge(insights) from data begins
'''

# Handle outliers
line_item_value_outliers=peter_romany_module.dealing_with_outlier(df,'Line Item Value',show_outliers=False)
unit_price_outliers=peter_romany_module.dealing_with_outlier(df,'Unit Price',show_outliers=False)
line_item_quantity_outliers=peter_romany_module.dealing_with_outlier(df,'Line Item Quantity',show_outliers=False)
pack_price_quantity_outliers=peter_romany_module.dealing_with_outlier(df,'Pack Price',show_outliers=False)
line_item_insurance_outliers=peter_romany_module.dealing_with_outlier(df,'Line Item Insurance (USD)',show_outliers=False)
unit_of_measure_outliers=peter_romany_module.dealing_with_outlier(df,'Unit of Measure (Per Pack)',show_outliers=False)

'''The method dealing_with_outlier() alternates between activation and deactivation to ensure the acquisition of accurate and comprehensive information from actual data.
It identifies outliers, storing them for further study and insight extraction purposes.'''

peter_romany_module.insights_by_descriptive_analytics(df,'Line Item Value')
peter_romany_module.insights_by_descriptive_analytics(df,'Line Item Quantity')

return_match_indices=peter_romany_module.regex_patterns(df,'Weight (Kilograms)','[A-Z]',return_match_indices=True,printing_result=False)
return_match_indices1=peter_romany_module.regex_patterns(df,'Freight Cost (USD)','[A-Z]',return_match_indices=True,printing_result=False)
# Encoding non-numeric values as -1 for numerical consistency in order to be able to make a classification and independent_sample_ttest
df.loc[return_match_indices,'Weight (Kilograms)']=-1
df.loc[return_match_indices1,'Freight Cost (USD)']=-1
df[['Weight (Kilograms)','Freight Cost (USD)']]=df[['Weight (Kilograms)','Freight Cost (USD)']].astype(dtype={'Weight (Kilograms)':int,'Freight Cost (USD)':float})

# Handle outliers
weight_outliers=peter_romany_module.dealing_with_outlier(df,'Weight (Kilograms)',show_outliers=False)
freight_cost_outliers=peter_romany_module.dealing_with_outlier(df,'Freight Cost (USD)',show_outliers=False)

# Top countries by the number of projects
top_countries = df.groupby('Country')['Project Code'].nunique().sort_values(ascending=False).head(3)
print("\nTop countries by the number of projects:")
print(top_countries)
# Average line item quantity by product group
avg_quantity_by_product_group = df.groupby('Product Group')['Line Item Quantity'].mean()
print("\nAverage line item quantity by product group:")
print(avg_quantity_by_product_group)
# Average line item value by vendor
avg_value_by_vendor = df.groupby('Vendor')['Line Item Value'].mean().sort_values(ascending=False).head(3)
print("\nAverage line item value by vendor (Top 3):")
print(avg_value_by_vendor)
# Average line item value by country
avg_value_by_country = df.groupby('Country')['Line Item Value'].mean().sort_values(ascending=False).head(3)
print("\nAverage line item value by country (Top 3):")
print(avg_value_by_country)
# Total line item quantity and value by vendor
total_quantity_value_by_vendor = df.groupby('Vendor').agg({'Line Item Quantity': 'sum', 'Line Item Value': 'sum'}).sort_values(by='Line Item Value', ascending=False).head(3)
print("\nTotal line item quantity and value by vendor (Top 3):")
print(total_quantity_value_by_vendor)
# Average delivery delay by country
df['Delivery Delay'] = (df['Delivered to Client Date'] - df['Scheduled Delivery Date']).dt.days
avg_delay_by_country = df.groupby('Country')['Delivery Delay'].mean().sort_values(ascending=False).head(3)
print("\nAverage delivery delay by country (Top 3):")
print(avg_delay_by_country)
# Most common product group by country
common_product_group_by_country = df.groupby('Country')['Product Group'].apply(lambda x: x.mode()[0]).head(3)
print("\nMost common product group by country (Top 3):")
print(common_product_group_by_country)
# Percentage of projects fulfilled via each fulfillment method
fulfillment_percentage = df['Fulfill Via'].value_counts(normalize=True) * 100
print("\nPercentage of projects fulfilled via each fulfillment method:")
print(fulfillment_percentage)
# Number of projects per year
df['Year'] = df['Scheduled Delivery Date'].dt.year
projects_per_year = df.groupby('Year')['Project Code'].nunique().sort_index()
print("\nNumber of projects per year:")
print(projects_per_year)
# Average line item quantity and value by product group
avg_quantity_value_by_product_group = df.groupby('Product Group').agg({'Line Item Quantity': 'mean', 'Line Item Value': 'mean'})
print("\nAverage line item quantity and value by product group:")
print(avg_quantity_value_by_product_group)
# Distribution of shipment modes
shipment_mode_distribution = df['Shipment Mode'].value_counts(normalize=True) * 100
print("\nDistribution of shipment modes:")
print(shipment_mode_distribution)
# Top manufacturing sites by total line item value
top_manufacturing_sites = df.groupby('Manufacturing Site')['Line Item Value'].sum().nlargest(3)
print("\nTop manufacturing sites by total line item value (Top 3):")
print(top_manufacturing_sites)
# Average freight cost by shipment mode
avg_freight_cost_by_shipment_mode = df.groupby('Shipment Mode')['Freight Cost (USD)'].mean()
print("\nAverage freight cost by shipment mode:")
print(avg_freight_cost_by_shipment_mode)
# Most common vendors for each product group
most_common_vendors_by_product_group = df.groupby('Product Group')['Vendor'].apply(lambda x: x.mode()[0])
print("\nMost common vendors for each product group:")
print(most_common_vendors_by_product_group)
# Average line item quantity and value by shipment mode
avg_quantity_value_by_shipment_mode = df.groupby('Shipment Mode').agg({'Line Item Quantity': 'mean', 'Line Item Value': 'mean'})
print("\nAverage line item quantity and value by shipment mode:")
print(avg_quantity_value_by_shipment_mode,'\n')

freight_cost_with_fulfill_via_direct_drop=df[df['Fulfill Via']=='Direct Drop']['Freight Cost (USD)']
freight_cost_with_fulfill_via_from_rdc=df[df['Fulfill Via']=='From RDC']['Freight Cost (USD)']
freight_cost_with_fulfill_via_direct_drop.name='freight_cost_with_fulfill_via_direct_drop'
freight_cost_with_fulfill_via_from_rdc.name='freight_cost_with_fulfill_via_from_rdc'

# Check for Entropy and data diversity
# print(freight_cost_with_fulfill_via_direct_drop.count())
# print(freight_cost_with_fulfill_via_from_rdc.count())

'''
After conducting a comprehensive analysis of entropy and data diversity within the dataset, it is evident that the data demonstrates balance, ensuring more equitable predictions and promoting model fairness.
'''

var1,var2=freight_cost_with_fulfill_via_direct_drop,freight_cost_with_fulfill_via_from_rdc
peter_romany_module.check_normality(var1,var2)
peter_romany_module.check_variance_homogeneity(var1,var2)
peter_romany_module.independent_sample_ttest(var1,var2,equal_variance=True)
# peter_romany_module.paired_sample_ttest(var1,var2)
peter_romany_module.one_way_anova(df,df['Product Group'],df['Line Item Quantity'],robust_anova=True)

deep_dive = pd.DataFrame({'ARV_Product_Group': df[df['Product Group'] == 'ARV']['Line Item Quantity'].tolist()})
peter_romany_module.insights_by_descriptive_analytics(deep_dive,'ARV_Product_Group')

variable1=df['Line Item Value']
variable2=df['Line Item Quantity']
peter_romany_module.check_normality(variable1,variable2)
peter_romany_module.check_variance_homogeneity(variable1,variable2)
peter_romany_module.pearsonr(variable1,variable2)
peter_romany_module.spearmanr(variable1,variable2)
peter_romany_module.sorted_zscore(variable1,show_z_score=False)
peter_romany_module.sorted_rank(variable1,show_percent_rank=False)

# df = df.drop(columns='Dosage')
# d = dtale.show(df, host='localhost', subprocess=False)
# d.open_browser()

'''                                                       Machine Learning
                 Dive into machine learning, leveraging algorithms to extract meaningful insights and make informed predictions from data.
'''

peter_romany_module.linear_regression(df,['Line Item Value'],'Line Item Insurance (USD)',False)
peter_romany_module.ridge_regression(df,['Line Item Value'],'Line Item Insurance (USD)',1,False)

# Encoding non-numeric values for numerical consistency in order to be able to make a SVC
peter_romany_module.label_encoder(df,'First Line Designation',show_label_codes=False)
peter_romany_module.label_encoder(df,'Shipment Mode',show_label_codes=False)
peter_romany_module.label_encoder(df,'Country',show_label_codes=False)
peter_romany_module.label_encoder(df,'Manufacturing Site',show_label_codes=False)

# Check for Entropy and data diversity
# print(df['Encoded First Line Designation'].value_counts())

'''
After conducting a comprehensive analysis of entropy and data diversity within the dataset, it is evident that the data demonstrates balance, ensuring more equitable predictions and promoting model fairness.
'''

peter_romany_module.support_vector_classification(df,['Weight (Kilograms)','Freight Cost (USD)'],'Encoded First Line Designation',Kernel="linear")
peter_romany_module.logistic_regression(df,['Weight (Kilograms)','Freight Cost (USD)'],'Encoded First Line Designation')
peter_romany_module.knn_classifier(df,['Weight (Kilograms)','Freight Cost (USD)'],'Encoded First Line Designation',K=5,majority_or_weighted_vote='uniform',euclidean_distance_or_else=2)
# perform clustering based on Freight Cost (USD)
optimal_k_means_model, optimal_cluster_labels = peter_romany_module.k_means_clustering(df, ['Freight Cost (USD)'])

peter_romany_module.standardization(df,'Weight (Kilograms)')
peter_romany_module.standardization(df,'Freight Cost (USD)')
svc_model=peter_romany_module.support_vector_classification(df,['Encoded Country', 'Encoded Manufacturing Site', 'Weight (Kilograms)', 'Freight Cost (USD)'],'Encoded Shipment Mode',Regularization=600,Kernel="rbf")

# Making a Predictive System
input_data = pd.DataFrame([[8,4,-0.446430,-0.137689]], columns=['Encoded Country', 'Encoded Manufacturing Site', 'Weight (Kilograms)', 'Freight Cost (USD)'])
prediction = svc_model.predict(input_data)
print("The prediction is: ",prediction)
if (prediction[0] == 0):
  print('The Shipment Mode is Air')
elif (prediction[0] == 1):
    print('The Shipment Mode is Truck')
elif (prediction[0] == 2):
    print('The Shipment Mode is Air Charter')
else:
  print('The Shipment Mode is Ocean')
