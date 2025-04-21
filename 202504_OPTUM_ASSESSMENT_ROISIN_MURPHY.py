# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
################## STEP 1 - IMPORT DATA #######################
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Set file path
file_path = 'C:/Users/bubbl/Documents/Job Search 2025/Optum Take Home Assessment/Data'

# Create outpatient claims dataframe, selecting only relevant columns
OP_Claims_Col_Names = ['DESYNPUF_ID', 'CLM_ID', 'SEGMENT', 'CLM_FROM_DT', 'CLM_THRU_DT', 'AT_PHYSN_NPI', 'PRVDR_NUM', 'CLM_PMT_AMT', 'ADMTNG_ICD9_DGNS_CD']
OP_Claims_Col_Dtypes = {'DESYNPUF_ID': str, 'CLM_ID': 'Int64', 'SEGMENT': 'Int64', 'AT_PHYSN_NPI': 'Int64', 'PRVDR_NUM': str, 'CLM_PMT_AMT': float, 'ADMTNG_ICD9_DGNS_CD': str}
OP_Claims_Parse_Dates = ['CLM_FROM_DT', 'CLM_THRU_DT']  # date fields
OP_Claims = pd.read_csv(f'{file_path}/DE1_0_2008_to_2010_Outpatient_Claims_Sample_20.csv', usecols = OP_Claims_Col_Names, dtype= OP_Claims_Col_Dtypes, parse_dates = OP_Claims_Parse_Dates, keep_default_na=True)

# Create member beneficiaries dataframe, selecting only relevant columns
Ben_Col_Names = ['DESYNPUF_ID', 'BENE_BIRTH_DT', 'BENE_DEATH_DT', 'BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'BENE_ESRD_IND', 'SP_STATE_CODE', 'BENE_COUNTY_CD', 'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA']
Ben_Col_Dtypes = {'DESYNPUF_ID': str, 'BENE_ESRD_IND': str} # remaining fields all reading in correctly as int (excl date fields below)
Ben_Parse_dates = ['BENE_BIRTH_DT', 'BENE_DEATH_DT' ] # date fields
Beneficiaries = pd.read_csv(f'{file_path}/DE1_0_2009_Beneficiary_Summary_File_Sample_20.csv', usecols= Ben_Col_Names, dtype=Ben_Col_Dtypes, parse_dates=Ben_Parse_dates)


# Check data for Nulls and correct datatypes

OP_Claims.info()
# OP_Claims has 790044 rows as expected pg 5 User Documentation, null values present in CLM_FROM and CLM_THRU dates (11028 rows), AT_PHYSN_NPI (17,440 rows), ADMTNG_ICD9_DGNS_CD (Don't think this field will be relevant, may remove later, ~600k null rows)

Beneficiaries.info()
# Beneficiaries has 114641 rows as expected pg 5 User Documentation, only column with null values is BENE_DEATH_DT (112,811 nulls) as would be expected

# Remove duplicate rows
OP_Claims.drop_duplicates(inplace = True)
Beneficiaries.drop_duplicates(inplace = True)

# check row count again to see if duplicates were present in the data
OP_Claims.info()
Beneficiaries.info()
# No duplicates were found


# Join the beneficiary summary file and the outpatient claim data based on Unique Identifier DESYNPUF_ID as per User Documentation pg 12

Merged_DF = pd.merge(OP_Claims, Beneficiaries, how="left", on="DESYNPUF_ID")


################## STEP 2 - INVESTIGATE THE DATA #######################
# Interrogate the dataset and check fields for unexpected values

print(Merged_DF['PRVDR_NUM'].value_counts())
# There are providers with up to 9895 claim records and some with only 1. 6307 distinct values

print(Merged_DF['CLM_PMT_AMT'].value_counts(bins = 20))
# negative claim payment amounts are possible

print(Merged_DF['ADMTNG_ICD9_DGNS_CD'].value_counts(dropna = True))
# 4400 distinct values excluding NAs

print(Merged_DF['BENE_BIRTH_DT'].value_counts(ascending = True, bins = 10))
# Oldest patients born 1931, youngest born 1983. As expected # patients per age bracket increases with age. 
# Medicare is the government healthcare program for US citizens aged 65+ so DOB 1983 should not be possible? 


print(Merged_DF['BENE_SEX_IDENT_CD'].value_counts(normalize=True))
# Higher proportion of patients have sex=2 (58%) - no M/F classification in the user manual. Possible values 1,2.

print(Merged_DF['BENE_RACE_CD'].value_counts(normalize=True))
# 84% patients have race = 1, possible values that exist within the dataset are 1,2,3,5.

print(Merged_DF['BENE_ESRD_IND'].value_counts())
# Possible values are 0,Y - End stage renal disease Indicator. Could change to Y/N but not necessary in my opinion

print(Merged_DF['SP_STATE_CODE'].value_counts(normalize=True))
print(len(Merged_DF['SP_STATE_CODE'].value_counts()))
# 52 distinct values

print(Merged_DF['BENE_COUNTY_CD'].value_counts())
# 309 distinct counties

# Now check specific chronic disease fields
print(Merged_DF['SP_ALZHDMTA'].value_counts(normalize=True))

print(Merged_DF['SP_CHF'].value_counts(normalize=True))

print(Merged_DF['SP_CHRNKIDN'].value_counts(normalize=True))

print(Merged_DF['SP_CNCR'].value_counts(normalize=True))

print(Merged_DF['SP_COPD'].value_counts(normalize=True))

print(Merged_DF['SP_DEPRESSN'].value_counts(normalize=True))

print(Merged_DF['SP_DIABETES'].value_counts(normalize=True))

print(Merged_DF['SP_ISCHMCHT'].value_counts(normalize=True))

print(Merged_DF['SP_OSTEOPRS'].value_counts(normalize=True))

print(Merged_DF['SP_RA_OA'].value_counts(normalize=True))

print(Merged_DF['SP_STRKETIA'].value_counts(normalize=True))
# Unclear on whether 1 or 2 represents 'Yes' - Assuming 1  = 'Yes', inferring from the data 
# and percentages in the user manual matching up with this data set, and generally speaking 1 = True in datasets such as these

# Data appears consistent from this brief inspection

# Correlation matrix

corr_df = Merged_DF.corr(numeric_only = True)
# from a brief scan the most correlated illnesses are chonic kidney disease and diabetes - not strongly correlated


################## STEP 3 - CONCATENATE CHRONIC ILLNESS COLUMNS #######################

# get the list of column names you want to concatenate - very much specific to the excel not changing column order
chr_illness_column_names = list(Merged_DF.columns.values)[16:27] 
# print(chr_illness_column_names)
Merged_DF['Critical_Illnesses_Concat'] = ''

# This loop is inefficient but does what I want it to - if I have more time I will attempt this another way to improve runtime
# Runtime ~15 mins
for i in chr_illness_column_names:
    # Create a variable to hold the string referencing the current chronic illness
    add_term = ''
    if i == 'SP_ALZHDMTA':
        add_term = 'Alzheimer; '
    elif i == 'SP_CHF':
        add_term = 'Heart Failure; '
    elif i == 'SP_CHRNKIDN':
        add_term = 'Kidney Disease; '
    elif i == 'SP_CNCR':
        add_term = 'Cancer; '
    elif i == 'SP_COPD':
        add_term = 'COPD; '
    elif i == 'SP_DEPRESSN':
        add_term = 'Depression; '
    elif i == 'SP_DIABETES':
        add_term = 'Diabetes; '
    elif i == 'SP_ISCHMCHT':
        add_term = 'Ischemic Heart Disease; '
    elif i == 'SP_OSTEOPRS':
        add_term = 'Osteoporosis; '
    elif i == 'SP_RA_OA':
        add_term = 'Rheumatoid Osteo/Arthritis; '
    elif i == 'SP_STRKETIA':
        add_term = 'Stroke/TIA; '
       
 # Loop through the rows of the dataframe       
    for row in Merged_DF.index:
# Check if the current illness column is a yes (1) or no (2)
        if Merged_DF.loc[row, i] == [1]:
# Check if the new column for concatenating illness has less than 2 illnesses listed - if yes add current illness
            Merged_DF.loc[row, 'Critical_Illnesses_Concat'] += add_term


# Create a variable where patients with 3 or more illnesses are marked as having multiple illnesses
Merged_DF['Multiple_Illnesses'] = ''
Merged_DF['Multiple_Illnesses'] = np.where(Merged_DF['Critical_Illnesses_Concat'].str.count(';').between(0,2, inclusive='both'), 'N', 'Y')


# Here we have a variable that shows up to 2 illness, or 'multiple' for 3 or more, or 'None where no illnesses are present
Merged_DF['Critical_Illnesses_Concat2'] = ''
Merged_DF['Critical_Illnesses_Concat2'] = np.where(Merged_DF['Multiple_Illnesses'] == 'Y', 'Multiple', np.where(Merged_DF['Critical_Illnesses_Concat'] == '', 'None', Merged_DF['Critical_Illnesses_Concat']))



################## STEP 4 - AGGREGATE AT PROVIDER AND CHRONIC ILLNESS LEVELS #######################

# sum of claims and count of members
Provider_Aggregate = Merged_DF.groupby(['AT_PHYSN_NPI'], as_index=False).agg({'DESYNPUF_ID':['nunique'], 'CLM_PMT_AMT':['sum']})
Provider_Aggregate.sort_values(by=('DESYNPUF_ID', 'nunique'), ascending = False)

CHR_Illness_Group_Aggregate = Merged_DF.groupby(['Critical_Illnesses_Concat2'], as_index=False).agg({'DESYNPUF_ID':['nunique'], 'CLM_PMT_AMT':['sum']})
CHR_Illness_Group_Aggregate.sort_values(by=('DESYNPUF_ID', 'nunique'), ascending = False)

Illness_Grp_By_Provider_Cost = Merged_DF.groupby(['AT_PHYSN_NPI', 'Critical_Illnesses_Concat2']).agg({'DESYNPUF_ID':['nunique'], 'CLM_PMT_AMT':['sum']})



################## STEP 5 - BASIC SUMMARIES #######################

# What is the distribution of races?
print(Beneficiaries['BENE_RACE_CD'].value_counts(normalize=True))
# 82.8% patients have race = 1, possible values that exist within the dataset are 1,2,3,5.


# What is the most common chronic illness combination?
print(Merged_DF.groupby(['Critical_Illnesses_Concat']).agg({'DESYNPUF_ID':['nunique']}).sort_values(by=('DESYNPUF_ID', 'nunique'), ascending = False))
# Diabetes; Ischemic Heart Disease; is the most common COMBINATION


# Which chronic illness combination has the total highest cost? (looking at all cominations, not using the field where > 2 = 'Multiple')
CHR_Illness_Highest_Cost = Merged_DF.groupby(['Critical_Illnesses_Concat']).agg({'CLM_PMT_AMT':['sum']})
Highest_Cost_Combination = CHR_Illness_Highest_Cost.sort_values(by=('CLM_PMT_AMT', 'sum'), ascending = False).index[1]
print(Highest_Cost_Combination)
# Heart Failure; Kidney Disease; Diabetes; Ischemic Heart Disease; 
# Claims against beneficiaries with no chronic illness tags have the overall highest cost (index 0)
# Using the concatenated illnesses field:
CHR_Illness_Group_Aggregate.sort_values(by=('CLM_PMT_AMT', 'sum'), ascending = False)
   

# Which chronic illness combination has the highest cost per member?
Illness_Group_Cost_Per_Member = Merged_DF.groupby(['Critical_Illnesses_Concat']).agg({'DESYNPUF_ID':['nunique'], 'CLM_PMT_AMT':['sum']})
Illness_Group_Cost_Per_Member['Cost_Per_Member'] = Illness_Group_Cost_Per_Member['CLM_PMT_AMT', 'sum']/ Illness_Group_Cost_Per_Member['DESYNPUF_ID', 'nunique'] # Could divide by Num_Members (variable initiated below)
Illness_Group_Cost_Per_Member.sort_values(by=('Cost_Per_Member'), ascending = False, inplace=True)
Max_Cost_Per_Member_Combination = Illness_Group_Cost_Per_Member.index[0]
print(Max_Cost_Per_Member_Combination)


# If we were to do it using the aggregated field then the groupings with maximum cost per member would be 'Multiple' followed by "Kidney Disease; Diabetes;"
CHR_Illness_Group_Aggregate.columns = CHR_Illness_Group_Aggregate.columns.get_level_values(0)
CHR_Illness_Group_Aggregate['Cost_Per_Member'] = CHR_Illness_Group_Aggregate['CLM_PMT_AMT'] / CHR_Illness_Group_Aggregate['DESYNPUF_ID']
Max_Cost_Per_Member_Combo = CHR_Illness_Group_Aggregate.sort_values(by=('Cost_Per_Member'), ascending = False)
print(Max_Cost_Per_Member_Combo[0:2])


################## STEP 6 - BENCHMARKING #######################

# For each provider (use AT_PHYSN_NPI) & chronic illness group, calculate the cost per member.
Cost_Per_Member_DF = Merged_DF[['AT_PHYSN_NPI', 'DESYNPUF_ID', 'Critical_Illnesses_Concat2', 'CLM_PMT_AMT']]
Cost_Per_Member_DF = Cost_Per_Member_DF.groupby(['AT_PHYSN_NPI', 'Critical_Illnesses_Concat2'], as_index=False).agg({'DESYNPUF_ID':['nunique'], 'CLM_PMT_AMT':['sum']}, inplace=True)
Cost_Per_Member_DF['Cost_Per_Member'] = Cost_Per_Member_DF['CLM_PMT_AMT', 'sum'] / Cost_Per_Member_DF['DESYNPUF_ID', 'nunique'] 
Cost_Per_Member_DF.sort_values(by=('Cost_Per_Member'), inplace=True)


Unique_Crit_illness_Groups = Cost_Per_Member_DF['Critical_Illnesses_Concat2'].unique()
len(Unique_Crit_illness_Groups)
# 68 unique groupings of critical illnesses


# Show histogrms with distribution of cost per member based on chronic illness group
for i in Unique_Crit_illness_Groups:
    temp_df = Cost_Per_Member_DF[Cost_Per_Member_DF['Critical_Illnesses_Concat2']==i]
    plt.hist(temp_df['Cost_Per_Member'], bins=100)
    plt.title(f'{i}')
    plt.suptitle('Distribution of cost per member by provider for illness combination:')
    plt.xlabel('Cost per Member')
    plt.ylabel('Count of Providers')
    plt.show()
    
# Histograms are generally all extremely right skewed

# Now repeat these histograms after removing cases where there is only one member per combination

Cost_Remove_Single_Members = Cost_Per_Member_DF[Cost_Per_Member_DF['DESYNPUF_ID', 'nunique'] > 1]
Unique_Crit_illness_Groups2 = Cost_Remove_Single_Members['Critical_Illnesses_Concat2'].unique()

for i in Unique_Crit_illness_Groups2:
    temp_df = Cost_Remove_Single_Members[Cost_Remove_Single_Members['Critical_Illnesses_Concat2']==i]
    plt.hist(temp_df['Cost_Per_Member'], bins=100)
    plt.title(f'{i}')
    plt.suptitle('Distribution of cost per member by provider where # members > 1 for illness combination:')
    plt.xlabel('Cost per Member')
    plt.ylabel('Count of Providers')
    
    plt.show()

# Still generally right skewed but less so

# Which providers are consistently expensive across chronic illnesses they treat?

# Per illness combination, get the 80th percentile for cost per member as a marker for 'expensive'
Eightieth_Percentiles = Cost_Per_Member_DF.groupby('Critical_Illnesses_Concat2')['Cost_Per_Member'].quantile(0.8).reset_index()
Eightieth_Percentiles.rename(columns = {'Cost_Per_Member':'Eightieth_Percentile'}, inplace=True)

Cost_Per_Member_DF.columns = Cost_Per_Member_DF.columns.get_level_values(0)
Expensive_Provider_DF = Cost_Per_Member_DF.merge(Eightieth_Percentiles, how = 'left', on = 'Critical_Illnesses_Concat2')

# Create a flag to show where a provider is considered expensive or not
Expensive_Provider_DF['Expensive_Flag'] = np.where(Expensive_Provider_DF['Cost_Per_Member']>=Expensive_Provider_DF['Eightieth_Percentile'], 'Y', 'N')

# Now we want to calculate a % of expensive flags per provider, those with >=50% are consistently expensive
Expensive_Provider_DF2 = Expensive_Provider_DF[['AT_PHYSN_NPI', 'Expensive_Flag']].groupby('AT_PHYSN_NPI', as_index=False).value_counts()
Expensive_Provider_DF2.columns = ['AT_PHYSN_NPI', 'Expensive_Flag', 'Count_Of_Flags']
Temp_Exp_Provider = Expensive_Provider_DF2.groupby(['AT_PHYSN_NPI'], as_index=False).agg({'Count_Of_Flags':['sum']})
Temp_Exp_Provider.columns = ['AT_PHYSN_NPI', 'Sum_Flags_Per_Provider']
Expensive_Provider_DF2 = Expensive_Provider_DF2.merge(Temp_Exp_Provider, on='AT_PHYSN_NPI')

Expensive_Provider_DF2['Expensive_Provider'] = np.where(Expensive_Provider_DF2['Expensive_Flag'] == 'Y', np.where(Expensive_Provider_DF2['Count_Of_Flags']/Expensive_Provider_DF2['Sum_Flags_Per_Provider'] >= 0.5, 'Y', 'N'), 'N')

List_Of_Expensive_Providers = Expensive_Provider_DF2['AT_PHYSN_NPI'].where(Expensive_Provider_DF2['Expensive_Provider']=='Y').dropna()
