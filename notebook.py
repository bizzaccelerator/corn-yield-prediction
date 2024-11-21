#!/usr/bin/env python
# coding: utf-8

# # Data Exploratory Analysis
# The following outlines the process I used to understand and analyze the dataset.

# In[1]:


# The first step involves importing the libraries required for the process:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# The graphics style selected is:
plt.style.use('ggplot')

# Statistical packages used
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, f_oneway, kruskal, pearsonr

# Model packages used
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# In[2]:


# The following allows us to view all the columns of the dataset, regardless of its size:
pd.set_option('display.max_columns', None)


# In[3]:


# Then the dataset is loaded as:
corn = pd.read_csv("C://Users/jober/Data_Projects/corn-yield-prediction/Dataset/corn_data.csv", sep=";", )


# ## Step 1: Understanding the data
# This step give us a general sense of the dataset: 

# In[4]:


corn.shape


# In[5]:


corn.head()


# In[6]:


corn.columns


# In[7]:


# Using the info() method, we can quickly identify the data type of each column and detect null values:"
corn.info()


# In[8]:


# The number of null values in the dataset is confirmed as:
corn.isna().sum()


# In[9]:


# The describe() function provides basic statistics for the numerical variables in the dataset:
corn.describe()


# ## Step 2: Data preparation
# Now that I have a general understanding of the data, some cleaning is needed before proceeding with further analysis.

# In[10]:


# Checking for duplicates:
corn.duplicated().sum()


# In[11]:


corn.loc[corn.duplicated(subset=['Farmer'])].shape


# In[12]:


# The column 'Farmer' indicates a unique record for each of the 422 platantion leader's.
corn['Farmer'].value_counts()


# Then there are no registries duplicated.

# In[13]:


# In addition, the following columns are not useful for creating a predictive model because they return the same value for all rows, as shown below:
cols = ['County', 'Crop', 'Power source', 'Water source','Crop insurance']
for c in cols:
    print(corn[c].value_counts())


# In[14]:


# Additionally, the columns 'Latitude' and 'Longitude' do not provide value due to their low variance within the analyzed county.


# In[15]:


# Then, our subset selected for analysis is:
corn_subset = corn[['Education', 'Gender', 'Age bracket',
                    'Household size', 'Acreage', 'Fertilizer amount', 'Laborers',
                    'Yield', 'Main credit source', 'Farm records', 
                    'Main advisory source', 'Extension provider', 'Advisory format', 
                    'Advisory language']]
corn_subset.head()


# In[16]:


# Column names in our refined dataframe are converted to lowercase, and spaces are removed for consistency and usability:
corn_subset.columns = [name.lower() for name in corn_subset.columns]
corn_subset.columns = [name.replace(" ","_") for name in corn_subset.columns]


# In[17]:


corn_subset.head()


# In[18]:


# Then, let's see the null values
corn_subset.isna().sum()


# In[19]:


# The null values in the 'acreage' column are:
corn_subset[corn_subset['acreage'].isna()]


# In[20]:


# The 71 entries lacking records of the amount of cultivated land are not useful for our objective. 
# Those registries represent:
missing_land = corn_subset['acreage'].isna().sum()
amount_ml = (missing_land / corn.shape[0])*100
print(f'The percentage of registries with missing values of cultivated land represent {amount_ml}')


# While removing a large number of missing values is generally not advisable, the lack of access to the research team for clarification and the limited usefulness of this data for our model, these rows will be removed from the dataframe.

# In[21]:


# The resulting dataframe is:
filter = corn_subset['acreage'].isna()
corn_subset = corn_subset[~filter]


# In[22]:


# The null values in the 'education' columns are:
corn_subset[corn_subset['education'].isna()].shape


# It makes sense that farmers in a developing country might have little to no formal education. Therefore, we can reasonably infer that many of them have not achieved any formal academic qualifications.

# In[23]:


# We populate the missing values in the education column with "No educated":
corn_subset.loc[corn_subset['education'].isna()] = corn_subset.loc[corn_subset['education'].isna()].fillna('No educated')


# In[24]:


corn_subset['education'].value_counts()


# In[25]:


# Finally, our cleaned dataset does not contains missing values:
corn_subset.isna().sum()


# In[26]:


# The main statistics for out clean dataset are:
corn_subset.describe(include='all')


# ## Step 3: Feature understanding
# 
# Now, it is important to understand how the selected variables behave:
# 
# ### Target variable (Yield):

# In[27]:


sns.histplot(
    corn_subset,
    x="yield", hue="gender",
    multiple="stack",
    kde=True,
    palette="light:m_r",
    edgecolor=".3",
    linewidth=.5,
    log_scale=False,
)


# In[28]:


# The Central Tendency measures are
mean = corn_subset['yield'].mean()
median = corn_subset['yield'].median()
print(f"Mean: {mean}, Median: {median}")

# The Dispersion measures are
std_dev = corn_subset['yield'].std()
iqr = corn_subset['yield'].quantile(0.75) - corn_subset['yield'].quantile(0.25)
print(f"Standard Deviation: {std_dev}, IQR: {iqr}")

# The Shape measures are
skewness = corn_subset['yield'].skew()
kurtosis = corn_subset['yield'].kurt()
print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")


# In[29]:


# The Outliers can be identifyed from a Boxplot
sns.boxplot(x=corn_subset['yield'])
plt.title('Box Plot')
plt.show()


# There are no outliers visible at first glance.

# - The variable 'education':

# In[30]:


# The education variable behave as:
sns.boxplot(x=corn_subset['education'], y=corn_subset['yield'])


# In[31]:


# To validate the variability in yield explained by the farmer's 'education', I'll execute a one-way anova 

# The firts step is spliting the categories in the column as follows:
groups = [group["yield"].values for name, group in corn_subset.groupby("education")]

# Then the actual ANOVA is performed using 
f_stat, p_value = f_oneway(*groups)
print(f"F-statistic: {f_stat}, P-value: {p_value}")

print(f"The p-value {p_value} is smaller than 0.05, then there is variability explained by the education of farmers.")
print("Thats why It's important to include  this variable in the model")


# - The variable 'gender':

# In[32]:


# The 'gender' variable behave as:
sns.boxplot(x="gender", y="yield", data=corn_subset)
plt.title("Yield of corn distribution by Gender")
plt.show()


# In[33]:


# Gruoping the dataframe using gender 
male_yield = corn_subset[corn_subset["gender"] == "Male"]["yield"]
female_yield = corn_subset[corn_subset["gender"] == "Female"]["yield"]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
print("Yield of corn for Male:", shapiro(male_yield))
print("Yield of corn for Female:", shapiro(female_yield))

# Test for Equal Variances (Levene’s Test)
stat, p = levene(male_yield, female_yield)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f'There is homogeinity of variance between genders [{p} is smaller than 0.05]')


# In[34]:


# Use T-test
t_stat, t_p = ttest_ind(male_yield, female_yield, equal_var=True)
print("\nTwo-sample T-test:")
print(f"T-statistic: {t_stat}, P-value: {t_p}")


# As identified, the calculated p-value of 0.0934717 is higher than 0.05. Therefore, there is no statistical difference between the genders of the farmers, and this variable will not be included in our first iteration of the models.
# 
# - Variable 'age_braket':

# In[35]:


# The 'age_bracket' variable behave as:
sns.boxplot(x="age_bracket", y="yield", data=corn_subset)
plt.title("Yield of corn distribution by Age braket")
plt.show()


# In[36]:


# From the illustration above, I identifyed atypical data in the '46-55' age_bracket category. Then,
# I'll use the function described as:

def atypical_data (df, target, variable, label):

    segment_data = df[df[variable] == label][target]

    Q1 = segment_data.quantile(0.25)
    Q3 = segment_data.quantile(0.75)
    IQR = Q3 - Q1
    Inf_Limit = Q1 - (1.5*IQR)
    Sup_Limit = Q3 + (1.5*IQR)

    outliers = df[(df[variable] == label) & ((df[target] < Inf_Limit) | (df[target] > Sup_Limit))]
    
    return outliers


# In[37]:


# The farmers between 46 and 55 years old with atypical production of corn are: 
aty_age = atypical_data(corn_subset, 'yield', 'age_bracket', '46-55')
aty_age.head(15)


# While there is one row in the yield column with 600 units, this individual appears to have greater access to resources. He cultivate four times more land than others and use fertilizer more intensively. He also uses his own financial resources for farming. It is reasonable to infer that this farmer is wealthier than their peers. I concluded that there are no illogical entries in the dataset for this variable.

# In[38]:


# The firts step is spliting the categories in the column as follows:
groups_age = [group["yield"].values for name, group in corn_subset.groupby("age_bracket")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
print("yield of 18-35:", shapiro(groups_age[0]))
print("yield of 36-45:", shapiro(groups_age[1]))
print("yield of 46-55:", shapiro(groups_age[2]))
print("yield of 56-65:", shapiro(groups_age[3]))
print("yield of above 65:", shapiro(groups_age[4]))

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_age)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f'There is homogeinity of variance between age brackets [{p} is smaller than 0.05]')


# In[39]:


# Then a one-way ANOVA is performed using 
f_stat, p_value = f_oneway(*groups_age)
print(f"F-statistic: {f_stat}, P-value: {p_value}")

print(f"The p-value {p_value} is smaller than 0.05, then there is variability explained by the age of farmers.")
print("Thats why It's important to include  this variable in the model")


# - Variable 'household_size': <br>
# <br>
# The values taken by the variable are:

# In[40]:


print(corn_subset['household_size'].unique())
print('\n While those are numeric values, they are best treated as categories.')


# In[41]:


#Transformation of the data type in the variable 'household_size':
corn_subset['household_size'] = corn_subset['household_size'].apply(str)


# In[42]:


# Now, the 'household_size' variable behave as:
sns.boxplot(x="household_size", y="yield", data=corn_subset)
plt.title("Yield of corn distribution by household size")
plt.show()


# In[43]:


# As evaluated in the age_bracket variable, the are some possible outliers at second and nineth categories. Then,
aty_hs_2 = atypical_data(corn_subset, 'yield', 'household_size', '2')
aty_hs_2


# In[44]:


# And also:
aty_hs_9 = atypical_data(corn_subset, 'yield', 'household_size', '9')
aty_hs_9


# These are records of farmers that break the consistency of the dataframe, particularly due to the small number of laborers they use. Therefore, it is reasonable to classify them as outliers that should be removed.

# In[45]:


# Merging the 2 subsets of outliers:
aty_hs_merged = pd.concat([aty_hs_2, aty_hs_9], axis=0, ignore_index=False)
aty_hs_merged
# Dropping the outliers
corn_subset = corn_subset.drop(aty_hs_merged.index)


# In[46]:


# Let's continue the same procedure for categorical variables as before:
groups_hsize = [group["yield"].values for name, group in corn_subset.groupby("household_size")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
for position, value in enumerate(groups_hsize):
    try:
        print(f"yield of {position}:", shapiro(value))
    except:
        print(f"error at {position}:")

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_hsize)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f'There is homogeinity of variance between household size [{p} is smaller than 0.05]')


# In[47]:


# Some of the household sizes does not follow a normal distribution. 
# Then, I'll apply the non-parametric alternative test for comparaison (Kruskal-Wallis Test):

stat, p = kruskal(*groups_hsize)
print("\nKruskal-Wallis H Test:")
print(f"U-statistic: {stat}, P-value: {p}")


# The P-value calculated 0.044118057103630565 is smaller than 0.05, then tehre is variability explained by the number of households in the dataset. Therefore, This variable should be included in the model.
# 
# - Variable 'laborers':
# 

# In[48]:


# Now, the 'laborers' variable behave as:
sns.boxplot(x="laborers", y="yield", data=corn_subset)
plt.title("Yield of corn distribution by number of laborers")
plt.show()


# In[49]:


#Transformation of the data type in the variable 'laborers':
corn_subset['laborers'] = corn_subset['laborers'].apply(str)

# The are some possible outliers in the category of '3' outliers. Then,
aty_lb_3 = atypical_data(corn_subset, 'yield', 'laborers', '3')
aty_lb_3


# However, I have no arguments to eliminate those registries from the dataset. Therefore, I will keep it for now.

# In[50]:


# Let's continue the same procedure for categorical variables as before:
groups_laborers = [group["yield"].values for name, group in corn_subset.groupby("laborers")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
for position, value in enumerate(groups_laborers):
    try:
        print(f"yield of {position}:", shapiro(value))
    except:
        print(f"error at {position}:")

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_laborers)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f"There isn't homogeinity of variance between number of laborers [{p} is greater than 0.05]")


# In[51]:


# The assumption of normally distributed data could not be proven, and there is no evidence of heteroscedasticity. 
# Then, I'll apply the non-parametric alternative test for comparaison (Kruskal-Wallis Test):

stat, p = kruskal(*groups_laborers)
print("\nKruskal-Wallis H Test:")
print(f"U-statistic: {stat}, P-value: {p}")


# The P-value calculated 0.03737355155094598 is smaller than 0.05, then there is variability explained by the number of laborers used for corn production. Therefore, This variable should be included in the model.
# 
# - Variable 'main_credit_source': 

# In[52]:


# Now, the 'main_credit_source' variable behave as:
sns.boxplot(x="main_credit_source", y="yield", data=corn_subset)
plt.title("Yield of corn distribution by the credit source used")
plt.show()


# In[53]:


# Let's continue the same procedure for categorical variables as before:
groups_credit = [group["yield"].values for name, group in corn_subset.groupby("main_credit_source")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
for position, value in enumerate(groups_credit):
    try:
        print(f"yield of {position}:", shapiro(value))
    except:
        print(f"error at {position}:")

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_credit)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f"There are unequal variances between credit source types because [{p} is greater than 0.05]")


# In[54]:


# Then, I'll apply the non-parametric alternative test for comparaison (Kruskal-Wallis Test):
stat, p = kruskal(*groups_credit)
print("\nKruskal-Wallis H Test:")
print(f"U-statistic: {stat}, P-value: {p}")


# The p-value of 0.257627 calculated above is greater than 0.05; therefore, there is no significant impact on corn production due to the source of credit used by the farmers.
# 
# - Variable 'farm_records':

# In[55]:


# Now, the 'farm_records' variable behave as:
sns.boxplot(x="farm_records", y="yield", data=corn_subset)
plt.title("Yield of corn distribution by record")
plt.show()


# In[56]:


# Let's continue the same procedure for categorical variables as before:
farm_wr = corn_subset[corn_subset['farm_records'] == 'Yes']['yield']
farm_wnr = corn_subset[corn_subset['farm_records'] == 'No']['yield']

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
print("Yield of corn for farms with records:", shapiro(farm_wr))
print("Yield of corn for farms without records:", shapiro(farm_wnr))

# Test for Equal Variances (Levene’s Test)
stat, p = levene(farm_wr, farm_wnr)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f"There are unequal variances between both groups because [{p} is greater than 0.05]")


# In[57]:


# Then, I'll apply the non-parametric alternative test for comparaison (U Mann-Withney Test):
stat, p = mannwhitneyu(farm_wr, farm_wnr, alternative='two-sided')
print(f"U-statistic: {stat}, P-value: {p}")


# The p-value of 0.9900469 calculated above is greater than 0.05; therefore, there is no significant impact on corn production due to the practice of keeping records in the plantations.
# 
# - Variable 'main_advisory_source':

# In[58]:


# Now, the 'main_advisory_source' variable behave as:
sns.boxplot(x="main_advisory_source", y="yield", data=corn_subset)
plt.title("Yield of corn distribution by advisory source")
plt.show()


# In[59]:


# Let's continue the same procedure for categorical variables as before:
groups_adv = [group["yield"].values for name, group in corn_subset.groupby("main_advisory_source")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
for position, value in enumerate(groups_adv):
    try:
        print(f"yield of {position}:", shapiro(value))
    except:
        print(f"error at {position}:")

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_adv)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f"There are unequal variances between different advisory groups because {p} is greater than 0.05")


# In[60]:


# Then, I'll apply the non-parametric alternative test for comparaison (Kruskal-Wallis Test):
stat, p = kruskal(*groups_adv)
print("\nKruskal-Wallis H Test:")
print(f"U-statistic: {stat}, P-value: {p}")


# The calculated p-value 0.01383421 is smaller than 0.05, then the variable of advisory group will be included in the model.
# 
# - Variable 'extension_provider':

# In[61]:


# Now, the 'extension_provider' variable behave as:
sns.boxplot(x="extension_provider", y="yield", data=corn_subset)
plt.title("Yield of corn distribution by extension provider")
plt.show()


# In[62]:


aty_gv = atypical_data(corn_subset, 'yield', 'extension_provider', 'National Government' )
aty_gv


# In[63]:


# Dropping the outliers for extension provider
corn_subset = corn_subset.drop(aty_gv.index)
corn_subset.shape


# In[64]:


# Let's continue the same procedure for categorical variables as before:
groups_ext = [group["yield"].values for name, group in corn_subset.groupby("extension_provider")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
for position, value in enumerate(groups_ext):
    try:
        print(f"yield of {position}:", shapiro(value))
    except:
        print(f"error at {position}:")

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_ext)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f"There are unequal variances between different extension provider because {p} is greater than 0.05")


# In[65]:


# Then, I'll apply the non-parametric alternative test for comparaison (Kruskal-Wallis Test):
stat, p = kruskal(*groups_ext)
print("\nKruskal-Wallis H Test:")
print(f"U-statistic: {stat}, P-value: {p}")


# The p-value of 0.67826571 calculated above is greater than 0.05; therefore, there is no significant impact on corn production based on the group extension.
# 
# - Variable 'advisory_format':

# In[66]:


# Now, the 'advisory_format' variable behave as:
sns.boxplot(x="advisory_format", y="yield", data=corn_subset)
plt.title("Yield of corn distribution by advisory format")
plt.show()


# In[67]:


# Let's continue the same procedure for categorical variables as before:
groups_fmt = [group["yield"].values for name, group in corn_subset.groupby("advisory_format")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
for position, value in enumerate(groups_fmt):
    try:
        print(f"yield of {position}:", shapiro(value))
    except:
        print(f"error at {position}:")

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_fmt)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f"There are unequal variances between different advisory format because {p} is greater than 0.05")


# In[68]:


# Then, I'll apply the non-parametric alternative test for comparaison (U Mann-Withney Test):
stat, p = mannwhitneyu(groups_fmt[0], groups_fmt[1], alternative='two-sided')
print(f"U-statistic: {stat}, P-value: {p}")


# The p-value of 0.360541 calculated above is greater than 0.05; therefore, there is no significant impact on corn production based on the advisory format.
# 
# - Variable 'advisory_language':

# In[69]:


# Now, the 'advisory_language' variable behave as:
sns.boxplot(x="advisory_language", y="yield", data=corn_subset)
plt.title("Yield of corn distribution by advisory language")
plt.show()


# In[70]:


# Let's continue the same procedure for categorical variables as before:
groups_lang = [group["yield"].values for name, group in corn_subset.groupby("advisory_language")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
for position, value in enumerate(groups_lang):
    try:
        print(f"yield of {position}:", shapiro(value))
    except:
        print(f"error at {position}:")

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_lang)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f"There are unequal variances between different advisory format because {p} is greater than 0.05")


# In[71]:


# Then, I'll apply the non-parametric alternative test for comparaison (Kruskal-Wallis Test):
stat, p = kruskal(*groups_lang)
print("\nKruskal-Wallis H Test:")
print(f"U-statistic: {stat}, P-value: {p}")


# The p-value of 0.1486936 calculated above is greater than 0.05; therefore, there is no significant impact on corn production based on the advisory language.

# In[72]:


# Now, the 'acreage' variable behave as:
sns.boxplot(x="acreage", y="yield", data=corn_subset)
plt.title("Yield of corn distribution by acres of land")
plt.show()


# In[73]:


#Transformation of the data type in the variable 'acreage':
corn_subset['acreage'] = corn_subset['acreage'].apply(str)

# The are some possible outliers in the category of '3' outliers. Then,
aty_ac_15 = atypical_data(corn_subset, 'yield', 'acreage', '1.5')
aty_ac_15 


# The farmers identifyed previously produce less yield than its peers in the category; however, the amount of laborers used is typical and the fertilizer amount used represent the median value of the category. Then, those registries are considered as insiders and will not be removed.

# In[74]:


# Let's continue the same procedure for categorical variables as before:
groups_acr = [group["yield"].values for name, group in corn_subset.groupby("acreage")]

# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
for position, value in enumerate(groups_acr):
    try:
        print(f"yield of {position}:", shapiro(value))
    except:
        print(f"error at {position}:")

# Test for Equal Variances (Levene’s Test)
stat, p = levene(*groups_acr)
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f"There are unequal variances depending on the amount of land cultivated because {p} is greater than 0.05")


# In[75]:


# Then, I'll apply the non-parametric alternative test for comparaison (Kruskal-Wallis Test):
stat, p = kruskal(*groups_acr)
print("\nKruskal-Wallis H Test:")
print(f"U-statistic: {stat}, P-value: {p}")


# The amount of cultivable land used in production was proven to be a variable that explain a lot of corn yield because the p-value calculated was almost zero.

# In[76]:


# The groups can be checked as:
for name, group in corn_subset.groupby("advisory_language"):
    print(f"advisory_language {name}: {group['yield'].values}")


# ### Additionally, the continuous variables are visualized as follows:

# In[77]:


sns.pairplot(corn_subset)


# - Variable 'fertilizer_amount':

# In[78]:


# The variable 'acreage' can be visualized as:
corn_subset.plot(kind='scatter', x = 'fertilizer_amount', y = 'yield',
                 title='Yield of corn by land')


# In[79]:


# Test for Normality (Shapiro-Wilk)
print("Normality Test (Shapiro-Wilk):")
print(f"fertilizer:", shapiro(corn_subset['fertilizer_amount']))

# Test for Equal Variances (Levene’s Test)
stat, p = levene(corn_subset['fertilizer_amount'], corn_subset['yield'])
print("\nLevene’s Test for Equal Variance:")
print(f"Statistic: {stat}, P-value: {p}")
print(f"There is heterocedasticity depending on the amount of fertilizer used because {p} is smaller than 0.05")


# In[80]:


# The correlation of the numerical variables is stimated by the Test of Pearson:
pearsonr(x=corn_subset['yield'], y=corn_subset['fertilizer_amount'], 
         alternative='two-sided', method=None)


# The variable is moderately and positively correlated with our target variable (yield). Therefore, it will be included in our model.

# The final cleaned data is the following:

# In[81]:


corn_subset.info()


# In[82]:


corn_subset.head()


# In[83]:


categorical_columns = ['education', 'age_bracket', 'household_size', 'laborers', 
                       'main_advisory_source', 'acreage']

# The significant variables identifyed were:
significant_var = ['education', 'age_bracket', 'household_size', 'laborers', 'main_advisory_source', 
               'acreage', 'fertilizer_amount', 'yield']

# The not significant variables identifyed were:
not_significant_var = ['gender', 'main_credit_source', 'farm_records', 'extension_provider', 
                       'advisory_format', 'advisory_language']


# ## Step 4: Model identification
# The cleaned dataset is filtered to include the significant variables identified above.

# In[84]:


corn_cleaned = corn_subset[significant_var]
corn.reset_index
corn_cleaned.head()


# Working dataset is prepared and splitted as follows:

# In[85]:


# Preparation dataset
# data_encoded = pd.get_dummies(corn_cleaned, columns=categorical_columns, drop_first=True)

X = corn_cleaned.drop('yield', axis=1)
y = corn_cleaned['yield']

# Turning the dataframes into dictionaries:
X_dic = X.to_dict(orient='records')


# In[86]:


# The data is transformed to dictionaries as:
X_dic[0]


# In[87]:


# Instanciating the vectorizer for Hot Encoding:
dv = DictVectorizer(sparse=False)

# Applying the vectorizer:
X_encoded = dv.fit_transform(X_dic)


# In[88]:


# The vectorized rows are transformed to the form of:
print(f'The column names are: {dv.get_feature_names_out()}')
print('\n The first element of the transformed dataset is: ')
X_encoded[0]


# Dataset splitted as follows: 60% for training, 20% for validation, and 20% for testing.

# In[89]:


# We first split for testing
X_full_train, X_test, y_full_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
# Then we split again for validation
X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.25, random_state=42)

# The lenght of the datasets can be validated as:
print(f'The number of registries in the train dataset is {len(X_train)}, in the validation dataset is {len(X_val)}, and in the test dataset is {len(X_test)}.')


# ### - Let's try some models:
# 
# __1. Linear Regression Model:__

# In[90]:


# The model is trained as follows:
linear = LinearRegression()
linear.fit(X_train, y_train)

# The trained model is used to predict the values in the test dataset:
y_pred_val = linear.predict(X_val)

# The main indicator for assessing the validity of the model is the Root Mean Squared Error (RMSE).
print("Linear Regression Metrics:")
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_val)))
print("R² Score:", r2_score(y_val, y_pred_val))


# Refining the parameters of a linear regression model typically involves introducing regularization techniques like Ridge Regression (L2 regularization) or Lasso Regression (L1 regularization). The hiperparameters can be selected as:

# In[91]:


# The first regularized model [Ridge] is 
ridge = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_ridge = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_ridge.fit(X_train, y_train)

print("Best Ridge Alpha:", grid_ridge.best_params_)
ridge_best = grid_ridge.best_estimator_


# In[92]:


# The second regularized model [Lasso] is 
lasso = Lasso()
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_lasso = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_lasso.fit(X_train, y_train)

print("Best Lasso Alpha:", grid_lasso.best_params_)
lasso_best = grid_lasso.best_estimator_


# In[93]:


# The evaluation of metrics for the model will be done using this formula:
def evaluate_model(y_test, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Performance:")
    print(f"  RMSE: {rmse}")
    print(f"  R² Score: {r2}")


# In[94]:


# Predictions for regularized models are:
y_pred_ridge = ridge_best.predict(X_val)
y_pred_lasso = lasso_best.predict(X_val)

# The evaluation for the linear models are:
evaluate_model(y_val, y_pred_val, "Linear Regression")
evaluate_model(y_val, y_pred_ridge, "Ridge Regression")
evaluate_model(y_val, y_pred_lasso, "Lasso Regression")


# The best model is the Lasso Regression model, as it has the lowest RMSE and the highest R² score. This model explains 81.83% of the variability in corn yield and has an average deviation of 51.041 units in corn production between the actual values in the test dataset and the model's predictions.
# 
# __2. Random Forest Model:__

# In[95]:


# The model is trained as follows:
random_forest = RandomForestRegressor(random_state=42)

# The trained model is used to predict the values in the test dataset:
random_forest.fit(X_train, y_train)
y_pred_val = random_forest.predict(X_val)

# The main indicator for assessing the validity of the model is the Root Mean Squared Error (RMSE).
print("Random Forest Metrics:")
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_val)))
print("R² Score:", r2_score(y_val, y_pred_val))


# In[96]:


# The parameters of the trained model are:
random_forest.get_params()


# In[97]:


# Tunning the hyperparameters is crucial. In this case, I'll define the followings:
param_grid = {
    'n_estimators': [20, 50, 100, 200, 300],    # Number of trees
    'max_depth': [None, 10, 20, 30],            # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],            # Minimum samples required to split
    'min_samples_leaf': [1, 2, 4],              # Minimum samples in a leaf
    'max_features': [1.0, 'sqrt', 'log2'],      # Features to consider at each split
}


# In[98]:


# I'll use GridSearchCV for exhaustive tuning
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=2, n_jobs=-1, 
                           error_score='raise')
grid_search.fit(X_train, y_train)

print("Best parameters for our Random forest model are:", grid_search.best_params_)
best_model_grid = grid_search.best_estimator_


# In[99]:


# The evaluation for both random forest models on the test set is:
for name, model in [("Raw Random forest", random_forest), ("Grid Search", best_model_grid)]:
    y_pred_val = model.predict(X_val)
    mse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"{name} - Test MSE: {mse}")
    print("R² Score:", r2_score(y_val, y_pred_val))


# The optimized random forest model (Grid search) has proven to explain almost 88.2% of the variability in yield of corn and lower average deviation from the actual value in the test set (43.7286 units) .
# 
# __3. Gradient Boosted Trees (GBT) Model:__

# In[100]:


# The model is trained as follows:
gbt = GradientBoostingRegressor(random_state=42)

# The trained model is used to predict the values in the test dataset:
gbt.fit(X_train, y_train)
y_pred_val = gbt.predict(X_val)

# The main indicator for assessing the validity of the model is the Root Mean Squared Error (RMSE).
print("GBT Metrics:")
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_val)))
print("R² Score:", r2_score(y_val, y_pred_val))


# In[101]:


gbt.get_params()


# In[102]:


# Tunning the hyperparameters is crucial. In this case, I'll define the followings:
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5],
    'n_estimators': [20, 50, 100, 200, 300],
    'max_depth': [3, 5, 7, 10, 20],
}


# In[103]:


# I'll use GridSearchCV for exhaustive tuning
grid_gbt = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                        param_grid=param_grid,
                        cv=3, scoring='neg_mean_squared_error')
grid_gbt.fit(X_train, y_train)

print("Best Parameters:", grid_gbt.best_params_)


# In[104]:


# The evaluation for both GBT models on the validation set is:
for name, model in [("Raw Gradient Boosted trees", gbt), ("Grid GBT", grid_gbt)]:
    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"{name} - Test MSE: {rmse}")
    print("R² Score:", r2_score(y_val, y_pred_val))


# Of the two Gradient Boosted Trees models, the optimized version is the most suitable. This model explains 87.375% of the variability in corn production and produces an average deviation of 45.105 units between the test values and the predicted values.
# 
# __To summarize, the chosen models produce the following RMSE and R² scores when applied to the test dataset:__

# In[105]:


# The list of models evaluated are:
listed_models = [("Linear Regression", linear), ("Rigde Regression", ridge_best), ("Lasso Regression", lasso_best), 
                 ("Raw Random forest", random_forest), ("Grid Search", best_model_grid), 
                 ("Raw Gradient Boosted trees", gbt), ("Grid GBT", grid_gbt)]

# The evaluation is performed by: 
result_scores = [] 
for name, model in listed_models:
    y_pred = model.predict(X_test)
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_value = r2_score(y_test, y_pred)
    result_scores.append([name, mse, r2_value])

scores_summary = pd.DataFrame(result_scores, columns=['Model', 'RSME', 'R-squared'])

# The summary of evaluation metrics is:
scores_summary


# The best model is the Optimized Gradient Boosted Trees (Grid GBT) because it produces the lowest average deviation from the test values (41.775) and provides the highest explanation of variability in yield production (90.1378%).
