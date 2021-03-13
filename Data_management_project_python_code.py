#!/usr/bin/env python
# coding: utf-8

# # I. Data Loading & Representation

# ## Necessary Librairies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import missingno as msno # not pre installed in anaconda
import folium # not pre installed in anaconda 
import math

from pandas.api.types import is_numeric_dtype
from scipy.stats import ttest_ind
from matplotlib import pyplot

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# to have nice graphs
sns.set_style(style="darkgrid")
sns.set_context("poster")
plt.style.use('ggplot')


# In[3]:


# to visualize whole dataframes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.options.display.width = 0


# # Ressources Used:
# - map: <br>
# https://python-visualization.github.io/folium/quickstart.html <br>
# https://france-geojson.gregoiredavid.fr/ -> to get geojson data of french departements
# - predictiv algorithme: <br>
# https://realpython.com/logistic-regression-python/  <br>
# https://machinelearningmastery.com/calculate-feature-importance-with-python/

# Warning :
# This project uses librairies/packages wich are not pre-installed in Anaconda. Two solutions are provided, so that you can make work the code in the different notebooks:
# 
# 1. You can simply install manually the two only packages we used which are not in the pre-installed Anadonda package, in your base environment or any virtual anaconda environment you want: <br>
# a. Folium -
# pip install folium <br>
# b. Missingno -
# pip install missingno
# 2. You can run in Anaconda prompt the following command: conda env create --name your_env_name --file=environment.yml

# ## Load Data

# In[4]:


# Main dataset
data = pd.read_csv('project-24-files/dataset.csv')
# Additional information
df_contract = pd.read_csv('project-24-files/dataset_contract.csv')
df_club = pd.read_csv('project-24-files/dataset_CLUB.csv')
# Geographical information
df_city_adm = pd.read_csv('project-24-files/city_adm.csv')
df_city_loc = pd.read_csv('project-24-files/city_loc.csv')
df_city_pop = pd.read_csv('project-24-files/city_pop.csv')
df_dep = pd.read_csv('project-24-files/departments.csv')
df_reg = pd.read_csv('project-24-files/regions.csv')

# dataset for modelling 
df = pd.read_csv('project-24-files/model_data.csv', index_col='Key')


# In[5]:


data.head()


# ## 1) Merge the main dataset with the Additional information variables (contract, club) and the geographical data

# In[6]:


# 1st we are going to set the key as the index in all three dataframes (data, df_contract, df_club)
data.set_index('key', inplace=True)
df_contract.set_index('key', inplace=True)
df_club.set_index('key', inplace=True)


# In[7]:


# join the three df based on key index
df_add_info = data.join(df_contract)
df_add_info = df_add_info.join(df_club)


# In[8]:


# we add a column Outcome_stat where success is encoding by 1 and failure by 0,
# this will help us in performing some statistical analysis
df_add_info['Outcome_stat'] = np.where(df_add_info['Outcome'] == 'success', 1,
                                        0)


# In[9]:


df_add_info.tail()


# Now we have a new dataframe containing all the basic information plus the additional information (contract, club)

# We now want to add the geographical data to our main dataset. In order to do so we will use the "insee_code" to merge df_city_adm, df_city_loc, df_city_pop to the df_add_info dataframe. Then we will use the "dep" and "reg" variables to add the departements and region data to the complete dataset.

# In[10]:


df_temp = pd.merge(df_city_adm, df_city_loc, on='Insee_code', how='outer')
df_city = pd.merge(df_temp, df_city_pop, on='Insee_code', how='outer') 


# In[11]:


# add departements and region data to city data (to get data about the full administrative and geographical structure of France)
df_france = pd.merge(df_city, df_dep, on='dep', how='outer')
df_france = pd.merge(df_france, df_reg, on='reg', how='outer')


# In[12]:


# Quick check of missing values
df_france.isna().sum()


# In[13]:


# total dataset concatenating the dataframe with additional informations and the dataframe with geographical information
df_total = pd.merge(df_add_info, df_france, on='Insee_code', how='inner')


# In[14]:


df_total.columns


# In[15]:


df_total.isna().sum()


# Now that we have built a dataset containing all the variables at hand, we are concerned about dealing with the missing values.

# ## 2) Dealing with Missing values

# ### a) Contract variable

# In[16]:


print('The total number of missing values in the "contract" variable is:',df_total['contract'].isna().sum())
print('Proportion of person with a missing contract in our dataset',
      (df_total['contract'].isna().sum() / df_total.shape[0]))


# We have a lot of missing values for this variable. The way we deal with those missing values may have a big impact on the results of our analysis, so this process must be taken with care. 
# 
# We can first try to understand what could have caused those indidividuals to have a missing contract in the dataset:
# * The individual is active and has a contract but his contract has been missed in the collection stage
# * The individual is inactive and hence he has no working contract (retiree, housewife, student etc.)
# 
# A first step could be to **try to identify inactive individuals using the "act" variable (which describes the nature of the activity of the individual) and assign them a new category of the "contract" variable: inactive.**
# We can then assume that the individuals left with a missing value in "contract" may be people for which the contract has not been properly registered in the collection process stage. Based on the number of observations in that case we can simply remove them of the dataset (if not too many) or try to identify their true contract using other variables (if they are too many).
# 
# If none of those techniques enables us to identify clearly the underlying aspect of the missing values, **we can also take those missing values as a category and rename if for instance "inactives or missing contracts or other"**. We could also remove all the observations with a missing value (but it might a bad idea since it would exclude inactives and other category of individuals from our sample).

# #### i) Try to identify inactives among the observations with a missing "contract"

# In[17]:


df_total.columns


# useful variables in our investigation:
# * IS_STUDENT
# * act (très utile) inactifs:
#     * (AT12","Chômeurs")
#     * "AT21","Retraités ou préretraités"
#     * AT22","Elèves, étudiants, stagiaires non rémunéré de 14 ans ou plus"
#     * "AT23","Moins de 14 ans"
#     * "AT24","Femmes ou hommes au foyer"
#     * "AT25","Autres inactifs"
# * Occupation_8 (retraités, Autres personnes sans activité professionelle)
# * Household_type ("Famille principale composée d'un couple d’aucun 'actif ayant un emploi' -> chomeur ont il une catégorie dans 'contract' déja?)

# <u>**Students:**</u>

# In[18]:


print('proportion of individuals which are students and have a missing value in "contract":', 
      df_total.loc[df_total['IS_STUDENT'],'contract'].isna().sum() / len(df_total[df_total['IS_STUDENT']]))
print('Total amount of missing values of "contract" which are explained by the individual being a student so inacitve:', 
      df_total.loc[df_total['IS_STUDENT'],'contract'].isna().sum())


# The first result is logic since most students are inactive. The second shows that already 1/4 of our missing values are easily explained using the **'IS_STUDENT' variable**. To precie our analysis and seperate inactive students from active students (with a contract) with can use the **"act" variable**.

# In[19]:


nb_inact_stu_nan_contr = df_total.loc[(df_total['IS_STUDENT']) & (df_total['act'] == 'AT22'),'contract'].isna().sum()
print('number of inactive students with a missing "contract" value:', nb_inact_stu_nan_contr) 
       
print('proportion of inactive student with a nan value in "contract":',
      nb_inact_stu_nan_contr / len(df_total[(df_total['IS_STUDENT']) & (df_total['act'] == 'AT22')]))


# We have found the number of students which are for sure inactive & which have a missing contract. Hence we can assign them the "inactive" category for the "contract" variable.

# In[20]:


msno.matrix(df_total.loc[df_total['IS_STUDENT'], ['Outcome','contract', 'act']]) 
plt.title('Missing values of "contract" for Students in our dataset')
plt.show()


# <u>**Housemen or Housewifes**</u>:

# In[21]:


# use 'act' AT24 to spot Housewifes


# In[22]:


nb_hwf_nan_contr = df_total.loc[df_total['act'] == 'AT24', 'contract'].isna().sum()
print('nber of housemen or housewifes with a missing value in "contract":', nb_hwf_nan_contr)
print('proportion of housemen or housewifes with a missing value in "contract":', 
      nb_hwf_nan_contr / len(df_total[df_total['act'] == 'AT24']))


# <u>**Retirees**</u>:

# In[23]:


# let's use "Occupation_8" variable to spot them.


# In[24]:


nb_ret_nan_contr = df_total[df_total['Occupation_8'] == 'CSP7']['contract'].isna().sum()
print('number of retirees with a missing value in "contract":', nb_ret_nan_contr)
print('proportion of retirees with a missing value in contract:', 
      nb_ret_nan_contr / len(df_total[df_total['Occupation_8'] == 'CSP7']))


# Again we have identified inactive peoples in our sample dataset for which the nan value in "contract" must simply be replaced by a new category "inactive".

# Hence using the information gathered through our triple analysis we can safely replace a bunch of missing values in the "contract" variable by creating a new category inactives. Once we have replaced those missing values, we will able to see how much nan's are left and try to fill the holes left.

# In[25]:


# assigning the "inactive" category to inactive students in our dataset:
df_total.loc[(df_total['IS_STUDENT']) & (df_total['act'] == 'AT22'), 'contract'] = 'inactive'
# to housewifes:
df_total.loc[df_total['act'] == 'AT24', 'contract'] = 'inactive'
# to retirees
df_total.loc[df_total['Occupation_8'] == 'CSP7', 'contract'] = 'inactive'


# <u>identify missing values left</u>

# In[26]:


print('missing values left in "contract":', df_total['contract'].isna().sum())


# Those missing values are possibly:
# * inactives that we have not spot yet (ex: Occupation_8 CSP8 "Autres personnes sans activité professionelle" ! chomeurs ou inactifs ? , act -> AT25 => "Autres inactifs", AT23 -> 'moins de 14 ans')
# * unemployed (ex: occupation_8 CSP8 "Autres personnes sans activité professionelle" , act -> AT12 "chomeurs")
# * active people for which the true contract has not been approprietly collected in the collection phase.
#     * drop those observations (Solution adopted - limits the risk of miss-assignments)
#     * try to identify their true contracts based on other variables and similitarities with other individuals in the sample with a defined contract.

# In[27]:


# When looking at CSP8 "Autres personnes sans activité professionellle we need to seperate inactives from unemployed people 
# using act"
print(df_total[(df_total['Occupation_8'] == 'CSP8') & df_total['contract'].isna()]['act'].value_counts())
print('\namong CSP8 personnes with a nan in "contract":\n 444 are other inactives  \n 102 are unemployed')


# In[28]:


df_total[df_total['act'] == 'AT25']['contract'].isna().sum()


# In[29]:


# spot childrens in dataset
df_total[df_total['act'] == 'AT23']['contract'].isna().sum()


# In[30]:


# unemployed 
df_total[df_total['act'] == 'AT12']['contract'].isna().sum()


# Using CSP8 of 'Occupation_8' and 'AT12' of the variable act we can spot the status of the last individuals with a missing value in "contract" which are other inactive people or unemployed people. Let's assign them to a category for the "contract variable".

# In[31]:


# assign "unemployed" category to the "contract" variable for people wich are unemployed
df_total.loc[df_total['act'] == 'AT12', 'contract'] = 'unemployed'
# assign "inactive" category to the "contract" variable for the last inactive people of the sample
df_total.loc[df_total['act'] == 'AT25', 'contract'] = 'inactive'


# <u>Look at the last missing values</u>

# In[32]:


df_total['contract'].isna().sum()


# In[33]:


df_total[df_total['contract'].isna()]


# Having a quick look to those observations we notice that those people our either retirees or young adults which are doing doing a non paying internship but are not students, so we can simply assign them to the "inactive" category.

# In[34]:


df_total.loc[df_total['contract'].isna(), 'contract'] = 'inactive'


# In[35]:


df_total['contract'].isna().sum()


# ### b) Club variable

# NB: These are only sport clubs

# In[36]:


df_total['CLUB'].isna().sum()


# The most plausible explanation for missing values in CLUB is that the people don't belong to any club.

# In[37]:


print('average age of people belonging to a sport club:', df_total[~df_total['CLUB'].isna()]['age_2020'].mean())


# In[38]:


print('average age of people not belonging to a sport club:',df_total[df_total['CLUB'].isna()]['age_2020'].mean())


# In[39]:


print('number of different sport clubs:',len(df_total['CLUB'].unique()))


# In[40]:


df_total['CLUB'].value_counts()[:10]


# Hence a good way to deal with missing values of 'CLUB' as well as to reencode the variable to make it clearer could be to transform the variable 'CLUB' into a boolean variable 'IS_CLUB' 1: belongs to a club, 0: don't belong to a club.

# In[41]:


df_total['IS_CLUB'] = np.where(df_total['CLUB'].isna(), False, True)
del df_total['CLUB']


# In[42]:


df_total.columns


# ## 3) Variable Encoding

# In[43]:


# create a new dataframes where variables have been reencoded
df_encode = df_total.copy()


# In[44]:


df_encode.head()


# In[45]:


# degree 
# act
# Household_type
# Occupation_8
# contract -> reencode for clarity + recategorize


# ### i) Rename categories to have more explicit nominations

# In[46]:


# Occupation_8
l_original_occ = list(df_total['Occupation_8'].unique())
l_new_occ = ['Agriculteurs exploitants', "Professions Intermédiaires", "Artisans commercants et chefs d'entreprise",
             "Ouvriers", "Retraités", "Autres personnes sans activité professionnelle", "Cadres et professions intellectuelles supérieures", "Employés"]
df_encode['Occupation_8'].replace(to_replace=l_original_occ, value=l_new_occ, inplace=True)


# In[47]:


# act
l_origin_act = list(df_total['act'].unique())
l_origin_act = [x for x in sorted(l_origin_act)]
l_new_occ = ["Actifs ayant un emploi, y compris sous apprentissage ou en stage rémunéré.", 
             "Chômeurs", "Retratés ou préretraités", "Elèves, étudiants, stagiaires non rémunéré de 14 ans ou plus",
             "Femmes ou hommes au foyer", "Autres inactifs"]
df_encode['act'].replace(to_replace=l_origin_act, value=l_new_occ, inplace=True)


# In[48]:


# Household_type
l_origin_hous = list(df_total['Household_type'].unique())
l_origin_hous = [x for x in sorted(l_origin_hous)]
l_new_hous = ['Homme vivant seul', 'Femme vivant seule', 'Plusieurs personnes sans famille', 'Famille monoparentale homme',
              'Famille monoparentale femme', 'Famille avec couple de deux actifs en emploi', 
              "Famille ou seul homme à le statut d'actif en emploi", "Famille ou seule femme à le statut d'actif en emploi",
              "Famille ou aucun à le statut d'actif en emploi"]
df_encode['Household_type'].replace(to_replace=l_origin_hous, value=l_new_hous, inplace=True)


# In[49]:


# contract
old_contract_names = ["contrat1|1", "contrat1|2", "contrat1|3", "contrat1|4",
                      "contrat1|5", "contrat1|6", "contrat2|1", "contrat2|2",
                      "contrat2|3"]
new_contract_names = ['apprentissage ou professionalisation', 'intérimaire',
                      'emplois aidés', 'stagiaire', 'autre à durée limitée',
                      'CDI ou fonctionnaire', 'Indépendants', 'Employeurs',
                      'Aides familiaux']
df_encode['contract'].replace(to_replace=old_contract_names, 
                                value=new_contract_names,
                                inplace=True)


# In[50]:


# degree
l_origin_degree = list(df_total['degree'].unique())
l_origin_degree = [x for x in sorted(l_origin_degree)]
l_new_degree = ["Pas de scolarité", "aucun diplôme primaire", "aucun diplôme collège", "CEP", "Brevet des collèges",
                "CAP, BEP ou diplôme de niveau équivalent", 
                "Baccalauréat général ou technologique, brevet supérieur, capacité en droit, DAEU, ESEU",
                "Baccalauréat ou brevet professionnel, diplôme équivalent", "BAC+2", "BAC+3 ou BAC+4", "BAC+5",
                "Doctorat de recherche"]
df_encode['degree'].replace(to_replace=l_origin_degree, value=l_new_degree, inplace=True)


# In[51]:


df_encode['degree'].value_counts()


# ### ii) Regroup categories to get more insighful variables

# <u>**contract**</u>

# In[52]:


df_encode['contract'].value_counts()


# In[53]:


l_outsiders = ['stagiaire', 'intérimaire', 'emplois aidés', 
               'autre à durée limitée', 'apprentissage ou professionalisation', 'unemployed']
l_insiders = ['CDI ou fonctionnaire']
l_independants = ['Indépendants', 'Employeurs', 'Aides familiaux']


# In[54]:


# reencode contract variable by replacing the old sub-categories by the 
# new categories

df_encode['contract'].replace(to_replace=l_outsiders, value='outsiders',
                               inplace=True)
df_encode['contract'].replace(to_replace=l_insiders, value='insiders',
                               inplace=True)
df_encode['contract'].replace(to_replace=l_independants, value='independants',
                               inplace=True)
df_encode['contract'].replace(to_replace='inactive', value='no contrat', inplace=True)


# In[55]:


df_encode['contract'].value_counts()


# <u>**household type**</u>

# In[56]:


df_encode['Household_type'].value_counts()


# In[57]:


l_celib = ['Homme vivant seul', 'Femme vivant seule']
l_coloc = ['Plusieurs personnes sans famille']
l_mono_fam = ['Famille monoparentale femme', 'Famille monoparentale homme']
l_famille = ['Famille avec couple de deux actifs en emploi', "Famille ou aucun à le statut d'actif en emploi",
             "Famille ou seule femme à le statut d'actif en emploi", "Famille ou seul homme à le statut d'actif en emploi"]

df_encode['Household_type'].replace(to_replace=l_celib, value='célibataire', inplace=True)
df_encode['Household_type'].replace(l_coloc, 'colocataire', inplace=True)
df_encode['Household_type'].replace(l_mono_fam, 'famille monoparentale', inplace=True)
df_encode['Household_type'].replace(l_famille, 'famille', inplace=True)


# In[58]:


df_encode['Household_type'].value_counts()


# <u>**degree**</u>

# In[59]:


df_encode['degree'].value_counts()


# We create new categories:
# * sans diplome: "Pas de scolarité", "aucun diplôme primaire", "aucun diplôme collège"
# * diplôme inférieur au baccalauréat: "CEP", "Brevet des collèges", "CAP, BEP ou diplôme de niveau équivalent"
# * Baccalauréat général ou technologique, diplôme équivalent : "Baccalauréat général ou technologique, brevet supérieur, capacité en droit, DAEU, ESEU"
# * Baccalauréat professionnel, diplôme équivalent: "Baccalauréat ou brevet professionel, diplôme équivalent"
# * BAC +2
# * BAC +3 ou BAC+4
# * BAC+5
# * Doctorat de recherche
# 

# In[60]:


# categories we need to rename or regroup
l_sans_diplome = ['Pas de scolarité', 'aucun diplôme primaire', 'aucun diplôme collège']
l_dipl_inf_bac = ['CEP', 'Brevet des collèges', "CAP, BEP ou diplôme de niveau équivalent"]
l_bac_gen_tech = ['Baccalauréat général ou technologique, brevet supérieur, capacité en droit, DAEU, ESEU']

df_encode['degree'].replace(l_sans_diplome, "sans diplôme", inplace=True)
df_encode['degree'].replace(l_dipl_inf_bac, 'diplôme inférieur au baccalauréat', inplace=True)
df_encode['degree'].replace(l_bac_gen_tech, 'Baccalauréat général ou technologique, diplôme équivalent', inplace=True)

df_encode['degree'].value_counts()


# regroupe HD0_1, HD0_2, HD0_3; -> sans diplome
# HD1_1, HD1_2, HD1_3 -> diplome inférieur au bac
# le reste on regroupe pas 

# In[61]:


df_encode.head(3)


# In[ ]:





# #### Reencoding of the arrondissements of the bigest cities
# we can see in the dataset that for the 3rd bigest cities of France, there is not only one Insee code, But many insee code corresponding to the several arrondissements of the city.
# In[62]:


Paris= ['Paris 1er arrondissement','Paris 15e arrondissement','Paris 18e arrondissement','Paris 19e arrondissement','Paris 20e arrondissement','Paris 16e arrondissement','Paris 17e arrondissement','Paris 18e arrondissement', 'Paris 2e arrondissement', 'Paris 3e arrondissement', 'Paris 4e arrondissement','Paris 5e arrondissement','Paris 6e arrondissement','Paris 7e arrondissement','Paris 8e arrondissement','Paris 9e arrondissement','Paris 10e arrondissement','Paris 11e arrondissement','Paris 12e arrondissement','Paris 13e arrondissement','Paris 14e arrondissement']
Lyon=['Lyon 1er arrondissement','Lyon 2e arrondissement','Lyon 3e arrondissement','Lyon 4e arrondissement','Lyon 5e arrondissement','Lyon 6e arrondissement','Lyon 7e arrondissement','Lyon 8e arrondissement','Lyon 9e arrondissement']
Marseille=['Marseille 1er arrondissement','Marseille 2e arrondissement','Marseille 3e arrondissement','Marseille 4e arrondissement','Marseille 5e arrondissement','Marseille 5e arrondissement','Marseille 6e arrondissement', 'Marseille 7e arrondissment','Marseille 8e arrondissment','Marseille 9e arrondissement','Marseille 10e arrondissement','Marseille 11e arrondissement', 'Marseille 12e arrondissement','Marseille 13e arrondissement','Marseille 14e arrondissement','Marseille 15e arrondissement']


df_encode['Nom de la commune'].replace(to_replace=Paris, value='Paris',
                               inplace=True)
df_encode['Nom de la commune'].replace(to_replace=Lyon, value='Lyon',
                               inplace=True)
df_encode['Nom de la commune'].replace(to_replace=Marseille, value='Marseille',
                               inplace=True)


# In[63]:


# We replace inhabitants value of arrondissement of paris by the total population of paris
df_encode.loc[df_encode['Nom de la commune'].str.startswith('Paris'), 'INHABITANTS'] = 1925548


# In[64]:


#we do the same for marseille
df_encode.loc[df_encode['Nom de la commune'].str.startswith('Lyon'), 'INHABITANTS'] = 514707


# In[65]:


#we do the same for lyon
df_encode.loc[df_encode['Nom de la commune'].str.startswith('Marseille'), 'INHABITANTS'] = 686370


# ### iii) Modify type of variables 

# In[66]:


# we change the type of categorical data to the Pandas categories type
# we identify categorical data for which we have less than 100 unique different values (to capture departements as a categorical 
# data)


# In[67]:


for col_name in df_encode.columns:
    # spot categorical variables but careful we don't want to change the category of boolean variables
    if (len(df_encode[col_name].unique()) <= 100) and (len(df_encode[col_name].unique()) > 2):
        df_encode[col_name] = df_encode[col_name].astype('category')

# manual rectifications
df_encode['Outcome_stat'] = df_encode['Outcome_stat'].astype('bool')
df_encode['sex'] = df_encode['sex'].astype('category')
df_encode['Outcome'] = df_encode['Outcome'].astype('category')


# In[68]:


# we do the same fo the dataframe containing administrative and geographical data of France
for col_name in df_france.columns:
    if (len(df_encode[col_name].unique()) <= 100) and (len(df_encode[col_name].unique()) > 2):
        df_france[col_name] = df_france[col_name].astype('category')


# In[69]:


df_encode.info()


# ## 3) bis - Drop useless variables

# In[70]:


pivot_csp = df_encode.pivot_table(index=['OCCUPATION_24'], columns=['Outcome'], 
                                values='f_name', aggfunc='count')
denominateur_csp = df_encode.pivot_table(index=['OCCUPATION_24'], 
                                values='f_name', aggfunc='count')
deno_array_csp = np.array(denominateur_csp)
pivot_freq_csp = round(pivot_csp / deno_array_csp * 100, 2)


# In[71]:


l_csp_cat = list(df_encode['OCCUPATION_24'].unique())

for cat in l_csp_cat:
    if type(cat) == str:
        cat_data = df_encode[df_encode['OCCUPATION_24'] == cat]['Outcome_stat']
    else:
        cat_data = df_encode[df_encode['OCCUPATION_24'].isna()]['Outcome_stat']
    if len(cat_data) < 30:
        conf_int = st.t.interval(alpha=0.95, df=len(cat_data)-1, loc=np.mean(cat_data), 
                      scale=st.sem(cat_data))
    else:
        conf_int = st.norm.interval(alpha=0.95, loc=np.mean(cat_data), 
                         scale=st.sem(cat_data))
    print('The 95% CI of ' + str(cat) + ' is:' + str((conf_int)))


# In[72]:


del df_encode['OCCUPATION_24']


# we droped this variable, due to the following fact:
#     frequences of OCCUPATION_8 where precise enough :we observed an homogeneity between the "sous catégorie des CSP" in OCCUPATION_24.
#     confidence intervals of some OCCUPATION_24 categories where not good enough

# ## 4) Analyze administrative and geographical structure of France

# > Careful the dep is a string type not a integer we might want to change that !

# In[73]:


df_france.head()


# In[74]:


# A revoir pour l'instant ...


# In[75]:


# Insee code -> relates to a particular city
# Outer index -> reg -> then departement -> city name -> inhabitants + X + Y etc.
df_test = df_france.set_index(['reg', 'dep', 'Insee_code'])


# In[76]:


df_test.head()


# ## 5. Save data merged and encoded

# In[77]:


df_encode.index.name = 'Key'


# In[78]:


df_encode.to_csv('project-24-files/total_data_encoded.csv', encoding='utf-8')
df_france.to_csv('project-24-files/geographical_data_encoded.csv', encoding='utf-8')


# # 2.2 Descriptive and predictive  part

# # Part 1: Descriptive part

# ## A. Study of the global success rate

# In[79]:


outcome_vc = df_add_info['Outcome'].value_counts()
outcome_vc


# In[80]:


sns.barplot(x=outcome_vc.index, y=outcome_vc.values)
plt.title('Outcome MC')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Outcome')
plt.ylabel('Number')
plt.show()


# ## B. Reencoding of the number variable, in order to make them more readible

# In[81]:


df_encode['agecat'] = pd.cut(df_encode['age_2020'], range(15, 100, 10))
#we are reencoding the age_2020 variable


# In[82]:


# CODE TO REENCODE "INHABITANTS" VERSION "AIRE URBAINE"
def reencode_aire_urbaine(nb_inhab):
    if 0 < nb_inhab <= 2000:
        aire_urbaine = 'aire rurale'
    elif 2000 < nb_inhab <= 20000:
        aire_urbaine = 'petite ville'
    elif 20000 < nb_inhab <= 100000:
        aire_urbaine = 'ville_moyenne'
    elif 100000 <  nb_inhab <= 1500000:
        aire_urbaine = 'grande ville'
    elif nb_inhab>1500000:
        aire_urbaine='Paris'
    return aire_urbaine

# j'applique la fonction créee pour le reencoding
df_aire_urbaine = df_encode.copy()
# reencoding
df_aire_urbaine['aire_urbaine'] = df_aire_urbaine['INHABITANTS'].apply(lambda x: reencode_aire_urbaine(x))


# ## C. Identification of charateristic of people in the success category and in the faillure category

# #### 1. Gender

# In[83]:


pivot_sex=df_add_info.pivot_table(index='sex', columns=['Outcome'], values='f_name', aggfunc='count')

pivot_proportion_success_sex=pivot_sex[['success']].div(57.20, axis=0)
pivot_prop_sex_f=pivot_sex[['failure']].div(42.80, axis=0)

df_pivot_prop_sex_s = pd.DataFrame(pivot_proportion_success_sex)
df_pivot_prop_sex_f=pd.DataFrame(pivot_prop_sex_f)

#graphical part of the success sample:
df_pivot_prop_sex_s.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of men and women in the success sample")
plt.show()
#graphical part of the faillure sample:
df_pivot_prop_sex_f.plot(kind='bar')
plt.ylabel('faillure (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of women and men in the faillure sample")
plt.show()

# we can see here that: the success sample contains more women and the faillure sample more men.
# #### 2. IS_STUDENT

# In[84]:


pivot_student=df_add_info.pivot_table(index='IS_STUDENT', columns=['Outcome'], values='f_name', aggfunc='count')

pivot_proportion_success_student=pivot_student[['success']].div(57.20, axis=0)
pivot_prop_student_f=pivot_student[['failure']].div(42.80, axis=0)

df_pivot_prop_student_s = pd.DataFrame(pivot_proportion_success_student)
df_pivot_prop_student_f=pd.DataFrame(pivot_prop_student_f)

#graphical part of the success sample:
df_pivot_prop_student_s.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of student and non_student in the success sample")
plt.show()
#graphical part of the faillure sample:
df_pivot_prop_student_f.plot(kind='bar')
plt.ylabel('faillure (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of student and non_student in the faillure sample")
plt.show()

# we can see in both case that the both sample contain more people which are not student that students. We can't really conclude anything with this result
# In[85]:


#### 3. CSP


# In[86]:


pivot_csp=df_add_info.pivot_table(index='Occupation_8', columns=['Outcome'], values='f_name', aggfunc='count')

pivot_proportion_success_csp=pivot_csp[['success']].div(57.20, axis=0)
pivot_prop_csp_f=pivot_csp[['failure']].div(42.80, axis=0)

df_pivot_prop_csp_s = pd.DataFrame(pivot_proportion_success_csp)
df_pivot_prop_csp_f=pd.DataFrame(pivot_prop_csp_f)

#graphical part of the success sample:
df_pivot_prop_csp_s.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of  CSP categories in the success sample")
plt.show()
#graphical part of the faillure sample:
df_pivot_prop_csp_f.plot(kind='bar')
plt.ylabel('faillure (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of CSP categories in the faillure sample")
plt.show()


# In[87]:


#### 4. Activites


# In[88]:


pivot_act=df_encode.pivot_table(index='act', columns=['Outcome'], values='f_name', aggfunc='count')

pivot_proportion_success_act=pivot_act[['success']].div(57.20, axis=0)
pivot_prop_act_f=pivot_act[['failure']].div(42.80, axis=0)

df_pivot_prop_act_s = pd.DataFrame(pivot_proportion_success_act)
df_pivot_prop_act_f=pd.DataFrame(pivot_prop_act_f)

#graphical part of the success sample:
df_pivot_prop_act_s.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of activities in the success sample")
plt.show()
#graphical part of the faillure sample:
df_pivot_prop_act_f.plot(kind='bar')
plt.ylabel('faillure (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of activities in the faillure sample")
plt.show()


# #### 5.Contract

# In[89]:


pivot_contract=df_encode.pivot_table(index='contract', columns=['Outcome'], values='f_name', aggfunc='count')

pivot_proportion_success_contract=pivot_contract[['success']].div(57.20, axis=0)
pivot_prop_contract_f=pivot_contract[['failure']].div(42.80, axis=0)

df_pivot_prop_contract_s = pd.DataFrame(pivot_proportion_success_contract)
df_pivot_prop_contract_f=pd.DataFrame(pivot_prop_contract_f)

#graphical part of the success sample:
df_pivot_prop_contract_s.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of contracts in the success sample")
plt.show()
#graphical part of the faillure sample:
df_pivot_prop_contract_f.plot(kind='bar')
plt.ylabel('faillure (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of contracts in the faillure sample")
plt.show()


# #### 6.CLUB

# In[90]:


pivot_club=df_encode.pivot_table(index='IS_CLUB', columns=['Outcome'], values='f_name', aggfunc='count')

pivot_proportion_success_club=pivot_club[['success']].div(57.20, axis=0)
pivot_prop_club_f=pivot_club[['failure']].div(42.80, axis=0)

df_pivot_prop_club_s = pd.DataFrame(pivot_proportion_success_club)
df_pivot_prop_club_f=pd.DataFrame(pivot_prop_club_f)

#graphical part of the success sample:
df_pivot_prop_club_s.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of people with a club or not in the success sample")
plt.show()
#graphical part of the faillure sample:
df_pivot_prop_club_f.plot(kind='bar')
plt.ylabel('faillure (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of people with a club or not in the faillure sample")
plt.show()


# #### 7. Degree

# In[91]:


pivot_degree=df_encode.pivot_table(index='degree', columns=['Outcome'], values='f_name', aggfunc='count')

pivot_proportion_success_degree=pivot_degree[['success']].div(57.20, axis=0)
pivot_prop_degree_f=pivot_degree[['failure']].div(42.80, axis=0)

df_pivot_prop_degree_s = pd.DataFrame(pivot_proportion_success_degree)
df_pivot_prop_degree_f=pd.DataFrame(pivot_prop_degree_f)

#graphical part of the success sample:
df_pivot_prop_degree_s.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of degree in the success sample")
plt.show()
#graphical part of the faillure sample:
df_pivot_prop_degree_f.plot(kind='bar')
plt.ylabel('faillure (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of degree in the faillure sample")
plt.show()


# #### 8. Household type

# In[92]:


pivot_Household_type=df_encode.pivot_table(index='Household_type', columns=['Outcome'], values='f_name', aggfunc='count')

pivot_proportion_success_Household_type=pivot_Household_type[['success']].div(57.20, axis=0)
pivot_prop_Household_type_f=pivot_Household_type[['failure']].div(42.80, axis=0)

df_pivot_prop_Household_type_s = pd.DataFrame(pivot_proportion_success_Household_type)
df_pivot_prop_Household_type_f=pd.DataFrame(pivot_prop_Household_type_f)

#graphical part of the success sample:
df_pivot_prop_Household_type_s.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of household types in the success sample")
plt.show()
#graphical part of the faillure sample:
df_pivot_prop_Household_type_f.plot(kind='bar')
plt.ylabel('faillure (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of Household types in the faillure sample")
plt.show()


# #### 9. Age

# In[93]:


pivot_age=df_encode.pivot_table(index='agecat', columns=['Outcome'], values='f_name', aggfunc='count')

pivot_proportion_success_age=pivot_age[['success']].div(57.20, axis=0)
pivot_prop_age_f=pivot_age[['failure']].div(42.80, axis=0)

df_pivot_prop_age_s = pd.DataFrame(pivot_proportion_success_age)
df_pivot_prop_age_f=pd.DataFrame(pivot_prop_age_f)

#graphical part of the success sample:
df_pivot_prop_age_s.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of age categories in the success sample")
plt.show()
#graphical part of the faillure sample:
df_pivot_prop_age_f.plot(kind='bar')
plt.ylabel('faillure (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of age categories in the faillure sample")
plt.show()


# #### 10. City type

# In[94]:


pivot_City_type=df_encode.pivot_table(index='City_type', columns=['Outcome'], values='f_name', aggfunc='count')

pivot_proportion_success_City_type=pivot_City_type[['success']].div(57.20, axis=0)
pivot_prop_City_type_f=pivot_City_type[['failure']].div(42.80, axis=0)

df_pivot_prop_City_type_s = pd.DataFrame(pivot_proportion_success_City_type)
df_pivot_prop_City_type_f=pd.DataFrame(pivot_prop_City_type_f)

#graphical part of the success sample:
df_pivot_prop_City_type_s.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of adminitrative city type  in the success sample")
plt.show()
#graphical part of the faillure sample:
df_pivot_prop_City_type_f.plot(kind='bar')
plt.ylabel('faillure (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of administrative City types in the faillure sample")
plt.show()


# #### 11. Population size of the City

# In[95]:


pivot_aire_urbaine=df_aire_urbaine.pivot_table(index='aire_urbaine', columns=['Outcome'], values='f_name', aggfunc='count')

pivot_proportion_success_aire_urbaine=pivot_aire_urbaine[['success']].div(57.20, axis=0)
pivot_prop_aire_urbaine_f=pivot_aire_urbaine[['failure']].div(42.80, axis=0)

df_pivot_prop_aire_urbaine_s = pd.DataFrame(pivot_proportion_success_aire_urbaine)
df_pivot_prop_aire_urbaine_f=pd.DataFrame(pivot_prop_aire_urbaine_f)

#graphical part of the success sample:
df_pivot_prop_aire_urbaine_s.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of city's sizes  in the success sample")
plt.show()
#graphical part of the faillure sample:
df_pivot_prop_aire_urbaine_f.plot(kind='bar')
plt.ylabel('faillure (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.title("proportion of city's sizes in the faillure sample")
plt.show()


# ## 4. Robustness of the results

# In[ ]:





# # Part 2: Predictive part

# ## A. Computering of the frequences of the several variable, and comparison to the global success rate

# #### 1. Gender

# In[96]:


pivot_gender = df_add_info.pivot_table(index=['sex'], columns=['Outcome'], 
                                values='f_name', aggfunc='count')
denominateur_gender = df_add_info.pivot_table(index=['sex'], 
                                values='f_name', aggfunc='count')
deno_array_gender = np.array(denominateur_gender)
pivot_freq_gender = round(pivot_gender / deno_array_gender * 100, 2)
print(pivot_freq_gender)

# here, we can see that the gender seems to have an influence: males have more faillure that the average of the sample. Being a female doesn't seem to determine.
# We can suppose that those results are relevant, due to the confidence intervals (see algorithme in the next part)
# #### 2. IS_STUDENT

# In[97]:


#we have to compute the frequences of success and failure for both categories
pivot_student = df_add_info.pivot_table(index=['IS_STUDENT'], columns=['Outcome'], 
                                values='f_name', aggfunc='count')
denominateur_student = df_add_info.pivot_table(index=['IS_STUDENT'], 
                                values='f_name', aggfunc='count')
deno_array_student = np.array(denominateur_student)
pivot_freq_student = round(pivot_student / deno_array_student * 100, 2)
print(pivot_freq_student)

# here, we can see that the students categorie are perfectly on the average. 
# being not a student is causes a little bit more faillure, but that's not so visible. We will consider that even if the confidences intervals are good, these categorie doesn't seem so relevant
# #### 3. CSP

# In[98]:


pivot_csp = df_encode.pivot_table(index=['Occupation_8'], columns=['Outcome'], 
                                values='f_name', aggfunc='count')
denominateur_csp = df_encode.pivot_table(index=['Occupation_8'], 
                                values='f_name', aggfunc='count')
deno_array_csp = np.array(denominateur_csp)
pivot_freq_csp = round(pivot_csp / deno_array_csp * 100, 2)

#graphical part:
df_pivot_freq_csp = pd.DataFrame(pivot_freq_csp)
df_pivot_freq_csp.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.show()

# we can see that all the categories present really manifest results
# we can see that being in the category "Agriculteurs exploitants" (particularly) or "Professions Intermédiaires"  or "Artisans commercants et chefs d'entreprise", "cadre et profession intellectuelle supérieur" or "autres personnes sans activité professionnelle", or in "employés" give more chance to have success result, that the average of the sample, while being in the categories "ouvriers, and "retraités" give strong chances to have faillure
# We can then say that this variable is really usefull and readible to predict the results of the marketing campaign.
# #### 4. Activity

# In[99]:


pivot_act = df_encode.pivot_table(index=['act'], columns=['Outcome'], 
                                values='f_name', aggfunc='count')
denominateur_act = df_encode.pivot_table(index=['act'], 
                                values='f_name', aggfunc='count')
deno_array_act = np.array(denominateur_act)
pivot_freq_act = round(pivot_act / deno_array_act * 100, 2)

#graphical part:
df_pivot_freq_act = pd.DataFrame(pivot_freq_act)
df_pivot_freq_act.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.show()

# we can see intersting results (different for the average result of the sample) for:
# Retraités and pré-retraités, but wich is probably linked to the result in the variable CSP: they are less convinced by the campaigne
# eleves étudiants et stagiaires: a little bit better results than the sample
# femmes ou hommes au foyer: the marketing campaigne has here a really good success rate
# Autres inactifs: much better result than the average success rate of the sample too.

# For the "actifs ayant un emploi" and "chômeur" categorie, we can see that the result is almost near from the global sample (just a little bit higher for the "actifs ayant un emploi"). Hence, we won't take these categories in account, and modelize it through the CSP (of course, chômeurs and actif ayant un emploi won't be distinguished though the catégorie CSP: they are considered as actifs. But that's not a problem, if being unemployed has not a real impact).
# we have almost good confidence interval (except maybe for "autre inactif", wich is a little bit less good, maybe due to the size of the sample): we can them take in account this variable.
# #### 5. Contract

# In[100]:


pivot_cont = df_encode.pivot_table(index=['contract'], columns=['Outcome'], 
                                values='f_name', aggfunc='count')
denominateur_cont = df_encode.pivot_table(index=['contract'], 
                                values='f_name', aggfunc='count')
deno_array_cont = np.array(denominateur_act)
pivot_freq_cont = round(pivot_act / deno_array_act * 100, 2)

#graphical part:
df_pivot_freq_cont = pd.DataFrame(pivot_freq_cont)
df_pivot_freq_cont.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.show()

# here, we have good confidence interval. The intersting results are:
#    "independants": they have a particularly strong success rate
#    "no contract": they have a lower success rate than the average of the sample. But this category is probably better analysed through the variable "act"
#    "insiders" have a little bit better results
   
# outsiders don't have a relevant rate (really near from the average of the sample)
# #### 6. CLUBS

# In[101]:


pivot_CLUB = df_encode.pivot_table(index=['IS_CLUB'], columns=['Outcome'], 
                                values='f_name', aggfunc='count')
denominateur_CLUB = df_encode.pivot_table(index=['IS_CLUB'], 
                                values='f_name', aggfunc='count')
deno_array_CLUB = np.array(denominateur_CLUB)
pivot_freq_CLUB = round(pivot_CLUB / deno_array_CLUB * 100, 2)
print(pivot_freq_CLUB)

#graphical part:
df_pivot_freq_CLUB = pd.DataFrame(pivot_freq_CLUB)
df_pivot_freq_CLUB.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.show()

# this variable doesn't seem to indicate an impact.  Indeed, results of both categories are particularly close.
# #### 7. Degree

# In[102]:


pivot_degree = df_encode.pivot_table(index=['degree'], columns=['Outcome'], 
                                values='f_name', aggfunc='count')
denominateur_degree = df_encode.pivot_table(index=['degree'], 
                                values='f_name', aggfunc='count')
deno_array_degree = np.array(denominateur_degree)
pivot_freq_degree = round(pivot_degree / deno_array_degree * 100, 2)
print(pivot_freq_degree)

#graphical part:
df_pivot_freq_degree = pd.DataFrame(pivot_freq_degree)
df_pivot_freq_degree.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.show()

# we can say that:
#    the confidence interval are pretty good,exept for doctorat de recherche. We won't study this category.
#    categories seem to have manifest results, and different from the sample. But we don't really see a correlation between the number of year of education, and the success rate.
    
#    "diplome inferieur au baccalauréat", "baccalaureat general ou technologique", and to a lesser extend the category "bac+3 ou BAC+ 4" have a low success rate.
    
#    "Bac+5", a "bac+2", a"Baccalaureat ou brevet professionnel" and to a lesser extend "sans diplôme" get a higher success rate than the average.
# #### 8. Household_type

# In[103]:


pivot_household = df_encode.pivot_table(index=['Household_type'], columns=['Outcome'], 
                                values='f_name', aggfunc='count')
denominateur_household = df_encode.pivot_table(index=['Household_type'], 
                                values='f_name', aggfunc='count')
deno_array_household = np.array(denominateur_household)
pivot_freq_household = round(pivot_household / deno_array_household * 100, 2)
print(pivot_freq_household)

#graphical part:
df_pivot_freq_household = pd.DataFrame(pivot_freq_household)
df_pivot_freq_household.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.show()

# Results seems robust (see next part with the confidence intervals), exept for the "colocataire" (but there result are pretty manifest, and the global confidence interval is far higher than the success rate of the global sample. Hence, we consider than we can keep this category)
# we can see that having a "famille" doesn't give a particurlar relevant result
# people of the category "celibataire" have few chance to get a "success"
# people of the category "famille monoparentale", and "colocataire" have a lot of chance to get a success
# #### 9. Age

# In[104]:


pivot_agecat = df_encode.pivot_table(index=['agecat'], columns=['Outcome'], 
                                values='f_name', aggfunc='count')
denominateur_agecat = df_encode.pivot_table(index=['agecat'], 
                                values='f_name', aggfunc='count')
deno_array_agecat = np.array(denominateur_agecat)
pivot_freq_agecat = round(pivot_agecat / deno_array_agecat * 100, 2)
print(pivot_freq_agecat)

#graphical part:
df_pivot_freq_agecat = pd.DataFrame(pivot_freq_agecat)
df_pivot_freq_agecat.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.show()


# In[105]:


#study of the correlation of the variable "age_cat"
df_corr = df_total[['Outcome_stat', 'age_2020']]
sns.heatmap(df_corr.corr(), cmap="BuPu")
plt.title('Correlation graph')
plt.show()

# we can see that we can't speak exactly of correlation betweean age and the Outcome
# However, some catégories of age have clearly manifest results, wich are robusts due to the confidence intervals:
#    we can see that after 55 years old, the chance of success begins law, and decreases with the age. It is consistent with the results we got with "retraités" (in the Occupation_8 and in the "act" variable).
#    From 25 to 45 years old, there is a higher chance to get a success.
# #### 10. City_type

# In[106]:


pivot_commune = df_encode.pivot_table(index=['City_type'], columns=['Outcome'], 
                                values='f_name', aggfunc='count')
denominateur_commune = df_encode.pivot_table(index=['City_type'], 
                                values='f_name', aggfunc='count')
deno_array_commune = np.array(denominateur_commune)
pivot_freq_commune = round(pivot_commune / deno_array_commune * 100, 2)
print(pivot_freq_commune)
df_pivot_freq_commune = pd.DataFrame(pivot_freq_commune)
df_pivot_freq_commune.plot(kind='bar')
plt.ylabel('Success (%)')
# design tick labels
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(rotation=55, ha="right")
plt.show()

# We can see here, that there is no correlation with the importance of the administrative fonction of the commune, and the success rate. For example, people in the sample from Paris have more chance to have a success, as in the chef-lieu canton, while people from prefecture de région have less chance than the sample average to have a success.
# Despite of that, their confidence interval are robust, and we can observe some manifest results depending on this administrative fonction:
#    capital d'etat, chef lieu canton, and sous-prefecture have a high success rate.
#    commune simple have a success rate a little bit lawer than the average
#    prefecture and prefecture de region have a strong faillure rate
# #### 11.The population size, thought the variable Aire_urbaine

# In[107]:


#we first study the correlation between the size of the population, and the success rate
df_corr = df_total[['Outcome_stat','INHABITANTS']]
sns.heatmap(df_corr.corr(), cmap="BuPu")
plt.title('Correlation graph')
plt.show()

#if we study the correlation of the "inhabitants" variable, we can see a deccorelation with the variable Outcome. It doesn't means we can't observe interesting results.
# In[108]:


pivot_aire_urb = df_aire_urbaine.pivot_table(index=['aire_urbaine'], columns=['Outcome'], 
                                values='INHABITANTS', aggfunc='count')

denominateur_aire_urb = df_aire_urbaine.pivot_table(index=['aire_urbaine'], 
                                values='INHABITANTS', aggfunc='count')
deno_array_aire_urb = np.array(denominateur_aire_urb)
pivot_freq_aire_urb = round(pivot_aire_urb / deno_array_aire_urb * 100, 2)
print(pivot_freq_aire_urb)

# first, we can see manifest and robust results (to observe that, going in the part "confidence interval"):
#    -the robustness of our results can justify in a certain way the categories we chose. 
#    -as we already saw with the correlation between "INHABITANTS" and "Outcome_stat", we don't see linear correlation between the size of the city and the success rate.
#    -we get particurlar law rate of success in the "grande ville".
#    -we get a higher rate of success than the average of the sample in the "villes moyennes" and in "Paris".
#    -we get a little bit higher faillure rate in "zone rurale" and "petite ville": it could be more interesting to link them.

# This result doesn't seem to point a link between potential "economical, demographical" dynamics (often link to the size, as we said to justify our reencoding), as we though.
# However, the disparities between Paris and the "grandes villes", question us on the impact or North and South differences.
# ### nota bene: 
# we didn't study the variable "Region" and "département" in this part. Indeed, we studeed them in the part 2.3 .

# ## 2. Robustness of the results: the confidence intervals

# #### Gender

# In[109]:


l_gender_cat = list(df_add_info['sex'].unique()) # list containing all the
# category names

# loop over each category
for cat in l_gender_cat:
    # get the data of outcome of MC for each category
    if type(cat) == str:
        cat_data = df_add_info[df_add_info['sex'] == cat]['Outcome_stat']
    # for missing values
    else:
        cat_data = df_add_info[df_add_info['sex'].isna()]['Outcome_stat']
    # check if the len of cat_data is > 30 ou < 30 to decide if we use 
    # Normal or t distribution for 95% CI
    if len(cat_data) < 30:
        conf_int = st.t.interval(alpha=0.95, df=len(cat_data)-1, loc=np.mean(cat_data), 
                      scale=st.sem(cat_data))
    else:
        conf_int = st.norm.interval(alpha=0.95, loc=np.mean(cat_data), 
                         scale=st.sem(cat_data))
        # st.sem(cat_data) = sigma / np.sqrt(N)
    print('The 95% CI of ' + str(cat) + ' is:' + str((conf_int)))


# #### student

# In[110]:


l_cat_student = list(df_encode['IS_STUDENT'].unique()) # list containing all the
# category names

# loop over each category
for cat in l_cat_student:
    # get the data of outcome of MC for each category
    if type(cat) == np.bool_:
        cat_data = df_encode[df_encode['IS_STUDENT'] == cat]['Outcome_stat']
    # for missing values
    else:
        cat_data = df_encode[df_encode['IS_STUDENT'].isna()]['Outcome_stat']
    # check if the len of cat_data is > 30 ou < 30 to decide if we use 
    # Normal or t distribution for 95% CI
    if len(cat_data) < 30:
        conf_int = st.t.interval(alpha=0.95, df=len(cat_data)-1, loc=np.mean(cat_data), 
                      scale=st.sem(cat_data))
    else:
        conf_int = st.norm.interval(alpha=0.95, loc=np.mean(cat_data), 
                         scale=st.sem(cat_data))
        # st.sem(ind_data) = sigma / np.sqrt(N)
    print('The 95% CI of ' + str(cat) + ' is:' + str((conf_int)))


# #### CSP

# In[111]:


l_csp_cat = list(df_encode['Occupation_8'].unique())

for cat in l_csp_cat:
    if type(cat) == str:
        cat_data = df_encode[df_encode['Occupation_8'] == cat]['Outcome_stat']
    else:
        cat_data = df_encode[df_encode['Occupation_8'].isna()]['Outcome_stat']
    if len(cat_data) < 30:
        conf_int = st.t.interval(alpha=0.95, df=len(cat_data)-1, loc=np.mean(cat_data), 
                      scale=st.sem(cat_data))
    else:
        conf_int = st.norm.interval(alpha=0.95, loc=np.mean(cat_data), 
                         scale=st.sem(cat_data))
    print('The 95% CI of ' + str(cat) + ' is:' + str((conf_int)))


# #### activity

# In[112]:


l_act_cat = list(df_encode['act'].unique())

for cat in l_act_cat:
    if type(cat) == str:
        cat_data = df_encode[df_encode['act'] == cat]['Outcome_stat']
    else:
        cat_data = df_encode[df_encode['act'].isna()]['Outcome_stat']
    if len(cat_data) < 30:
        conf_int = st.t.interval(alpha=0.95, df=len(cat_data)-1, loc=np.mean(cat_data), 
                      scale=st.sem(cat_data))
    else:
        conf_int = st.norm.interval(alpha=0.95, loc=np.mean(cat_data), 
                         scale=st.sem(cat_data))
    print('The 95% CI of ' + str(cat) + ' is:' + str((conf_int)))


# #### contract

# In[113]:


l_contr_cat = list(df_encode['contract'].unique()) # list containing all the
# category names

# loop over each category
for cat in l_contr_cat:
    # get the data of outcome of MC for each category
    if type(cat) == str:
        cat_data = df_encode[df_encode['contract'] == cat]['Outcome_stat']
    # for missing values
    else:
        cat_data = df_encode[df_encode['contract'].isna()]['Outcome_stat']
    # check if the len of cat_data is > 30 ou < 30 to decide if we use 
    # Normal or t distribution for 95% CI
    if len(cat_data) < 30:
        conf_int = st.t.interval(alpha=0.95, df=len(cat_data)-1, loc=np.mean(cat_data), 
                      scale=st.sem(cat_data))
    else:
        conf_int = st.norm.interval(alpha=0.95, loc=np.mean(cat_data), 
                         scale=st.sem(cat_data))
        # st.sem(cat_data) = sigma / np.sqrt(N)
    print('The 95% CI of ' + str(cat) + ' is:' + str((conf_int)))


# #### degree

# In[114]:


l_contr_degree = list(df_encode['degree'].unique()) # list containing all the
# category names

# loop over each category
for cat in l_contr_degree:
    # get the data of outcome of MC for each category
    if type(cat) == str:
        cat_data = df_encode[df_encode['degree'] == cat]['Outcome_stat']
    # for missing values
    else:
        cat_data = df_encode[df_encode['degree'].isna()]['Outcome_stat']
    # check if the len of cat_data is > 30 ou < 30 to decide if we use 
    # Normal or t distribution for 95% CI
    if len(cat_data) < 30:
        conf_int = st.t.interval(alpha=0.95, df=len(cat_data)-1, loc=np.mean(cat_data), 
                      scale=st.sem(cat_data))
    else:
        conf_int = st.norm.interval(alpha=0.95, loc=np.mean(cat_data), 
                         scale=st.sem(cat_data))
        # st.sem(cat_data) = sigma / np.sqrt(N)
    print('The 95% CI of ' + str(cat) + ' is:' + str((conf_int)))


# #### Household_type

# In[115]:


l_contr_h = list(df_encode['Household_type'].unique()) # list containing all the
# category names

# loop over each category
for cat in l_contr_h:
    # get the data of outcome of MC for each category
    if type(cat) == str:
        cat_data = df_encode[df_encode['Household_type'] == cat]['Outcome_stat']
    # for missing values
    else:
        cat_data = df_encode[df_encode['Household_type'].isna()]['Outcome_stat']
    # check if the len of cat_data is > 30 ou < 30 to decide if we use 
    # Normal or t distribution for 95% CI
    if len(cat_data) < 30:
        conf_int = st.t.interval(alpha=0.95, df=len(cat_data)-1, loc=np.mean(cat_data), 
                      scale=st.sem(cat_data))
    else:
        conf_int = st.norm.interval(alpha=0.95, loc=np.mean(cat_data), 
                         scale=st.sem(cat_data))
        # st.sem(cat_data) = sigma / np.sqrt(N)
    print('The 95% CI of ' + str(cat) + ' is:' + str((conf_int)))


# #### age

# In[116]:


l_contr_agecat = list(df_encode['agecat'].unique()) # list containing all the
# category names

# loop over each category
for cat in l_contr_agecat:
    # get the data of outcome of MC for each category
    try:
        if math.isnan(cat):
            cat_data = df_encode[df_encode['agecat'].isna()]['Outcome_stat']
    except TypeError:
        cat_data = df_encode[df_encode['agecat'] == cat]['Outcome_stat']

    # check if the len of cat_data is > 30 ou < 30 to decide if we use 
    # Normal or t distribution for 95% CI
    if len(cat_data) < 30:
        conf_int = st.t.interval(alpha=0.95, df=len(cat_data)-1, loc=np.mean(cat_data), 
                      scale=st.sem(cat_data))
    else:
        conf_int = st.norm.interval(alpha=0.95, loc=np.mean(cat_data), 
                         scale=st.sem(cat_data))
        # st.sem(cat_data) = sigma / np.sqrt(N)
    print('The 95% CI of ' + str(cat) + ' is:' + str((conf_int)))


# #### City_type

# In[117]:


l_contr_commune = list(df_encode['City_type'].unique()) # list containing all the
# category names

# loop over each category
for cat in l_contr_commune:
    # get the data of outcome of MC for each category
    if type(cat) == str:
        cat_data = df_encode[df_encode['City_type'] == cat]['Outcome_stat']
    # for missing values
    else:
        cat_data = df_encode[df_encode['City_type'].isna()]['Outcome_stat']
    # check if the len of cat_data is > 30 ou < 30 to decide if we use 
    # Normal or t distribution for 95% CI
    if len(cat_data) < 30:
        conf_int = st.t.interval(alpha=0.95, df=len(cat_data)-1, loc=np.mean(cat_data), 
                      scale=st.sem(cat_data))
    else:
        conf_int = st.norm.interval(alpha=0.95, loc=np.mean(cat_data), 
                         scale=st.sem(cat_data))
        # st.sem(cat_data) = sigma / np.sqrt(N)
    print('The 95% CI of ' + str(cat) + ' is:' + str((conf_int)))


# In[118]:


#### Size of the population of the city


# In[119]:


l_contr_taille_commune = list(df_aire_urbaine['aire_urbaine'].unique()) # list containing all the
# category names

# loop over each category
for cat in l_contr_taille_commune:
    # get the data of outcome of MC for each category
    if type(cat) == str:
        cat_data = df_aire_urbaine[df_aire_urbaine['aire_urbaine'] == cat]['Outcome_stat']
    # for missing values
    else:
        cat_data = df_aire_urbaine[df_aire_urbaine['aire_urbaine'].isna()]['Outcome_stat']
    # check if the len of cat_data is > 30 ou < 30 to decide if we use 
    # Normal or t distribution for 95% CI
    if len(cat_data) < 30:
        conf_int = st.t.interval(alpha=0.95, df=len(cat_data)-1, loc=np.mean(cat_data), 
                      scale=st.sem(cat_data))
    else:
        conf_int = st.norm.interval(alpha=0.95, loc=np.mean(cat_data), 
                         scale=st.sem(cat_data))
        # st.sem(cat_data) = sigma / np.sqrt(N)
    print('The 95% CI of ' + str(cat) + ' is:' + str((conf_int)))


# In[ ]:





# ## Data Prepartion

# In[120]:


# variable selected for prediction of MC outcome
# - sex
# - age_2020
# - degree
# - act
# - Occupation_8
# - contract
# - City_type
# - INHABITANTS
# - aire_urbaine
# is_south


# In[121]:


# adding is_south variable
l_south_reg = ['Nouvelle-Aquitaine', 'Occitanie',"Provence-Alpes-Côte d'Azur", "Auvergne-Rhône-Alpes", "Corse"]
df['is_south'] = np.where(df['Nom de la région'].isin(l_south_reg), 1, 0)


# In[122]:


df_model = df[['Outcome_stat', 'sex', 'age_2020', 'degree', 'act', 'Occupation_8', 'contract', 'City_type', 'INHABITANTS',
               'aire_urbaine', 'is_south']].copy()


# In[123]:


df_model.head()


# In[124]:


df_clf = pd.get_dummies(df_model)


# In[125]:


df_clf.columns


# In[126]:


# drop useless dummies
df_clf.drop('sex_Male', inplace=True, axis=1)


# In[127]:


len(df_clf.columns) # 42 columns


# ## Model Building

# ### <u>Benchmark Model 1 (using a minimal set of variables)</u>

# In[128]:


# Results of the algorithm are very poor let's try with minimal variables


# In[129]:


y = df_clf['Outcome_stat']
X = df_clf[['age_2020', 'sex_Female', 'is_south']]
X['contract'] = df['contract'].copy()
X = pd.get_dummies(X, columns=['contract'])
X.head()


# In[130]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[131]:


model = LogisticRegression(solver='liblinear', random_state=42)


# In[132]:


model.fit(X_train, y_train)
ytrain_proba = model.predict_proba(X_train)
ytrain_pred = model.predict(X_train)
accuracy_score(ytrain_pred, y_train)


# In[133]:


# evaluate accuracy using test set
yhat = model.predict(X_test)
print(accuracy_score(y_test, yhat))


# In[134]:


# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# In[135]:


### <u>Benchmark Model 2 (using all the variables identified as useful in the descriptive analysis)</u>


# In[136]:


y = df_clf['Outcome_stat']
X = df_clf.drop('Outcome_stat', axis=1)


# In[137]:


X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True, 
                                                    test_size=0.30, random_state=42)


# In[138]:


model.fit(X_train, y_train)


# In[139]:


ytrain_proba = model.predict_proba(X_train)
ytrain_pred = model.predict(X_train)


# In[140]:


accuracy_score(ytrain_pred, y_train)


# In[141]:


# evaluate accuracy using test set
yhat = model.predict(X_test)
print(accuracy_score(y_test, yhat))


# In[142]:


# Results are really bad using all the variables, let's proceed do some feature selection


# ## Model - Using Decision Tree Classifier

# In[143]:


y = df_clf['Outcome_stat']
X = df_clf.drop('Outcome_stat', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True, 
                                                    test_size=0.30, random_state=42)


# In[144]:


DecisionTree_Class_Model = DecisionTreeClassifier(random_state=42)
DecisionTree_Class_Model.fit(X_train, y_train)


# In[145]:


ytrain_pred = DecisionTree_Class_Model.predict(X_train)


# In[146]:


ytrain_pred


# In[147]:


accuracy_score(y_train, ytrain_pred) # there's clearly overfitting


# In[148]:


# using test set:
y_pred = DecisionTree_Class_Model.predict(X_test)
accuracy_score(y_test, y_pred)
# a bit of overfitting, let's then look at feature importance.


# In[149]:


l_var_dt = list(X.columns)
l_var_keep = []


# In[150]:


# using CART (classficiation and regression tree) algortihm to assess Feature Importance 
# ressource used: https://machinelearningmastery.com/calculate-feature-importance-with-python/
importance = DecisionTree_Class_Model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, {}, Score: %.5f'.format(l_var_dt[i]) % (i,v))
    if v > 0.01: # threshold to select variables
        l_var_keep.append(l_var_dt[i])
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[151]:


l_var_keep


# ## Last Model improved with Feature selection

# In[152]:


###### from this variable selection it seems that the most relevant variables in our prediction are:
# - age_2020
# - INHABITANTS
# - sex 
# - is_south
# - Occupation_8
# - degree 
# - City type
# let's build a new decision tree algortihm only using those variables


# In[153]:


y = df_clf['Outcome_stat']
X = df[['age_2020', 'INHABITANTS', 'is_south', 'Occupation_8', 'degree', 'City_type']]
X['sex'] = df_clf.loc[:, 'sex_Female'].copy()
X = pd.get_dummies(X, columns=['Occupation_8', 'degree', 'City_type'])
X


# In[154]:


X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True, 
                                                    test_size=0.30, random_state=42)


# In[155]:


DecisionTree_Class_Model = DecisionTreeClassifier(random_state=42)
DecisionTree_Class_Model.fit(X_train, y_train)


# In[156]:


ytrain_pred = DecisionTree_Class_Model.predict(X_train)


# In[157]:


accuracy_score(y_train, ytrain_pred) # still signs of overfitting 


# In[158]:


# using test set:
y_pred = DecisionTree_Class_Model.predict(X_test)
accuracy_score(y_test, y_pred)
# overfitting remains fairly strong.However results on the the test set are fairly good, despite reducing the use of many variables.

# We need to correct for overfitting. The Decision Tree built tends to overfit the data, by creating level of depths and leafs
# until the training data is perfectly fitted. This leads to poor generalization of our algorithm to new observations and hence
# poor predictability power. We hence neeed to restrain the depth and the number of leafs to the level at which the algorithm
# can infer useful information from the dataset while avoiding noise. 
# We therefore are going to apply post pruning to our a Decision Tree Classifier using a cost complexity parameter (alpha) that 
# enables us to spot the right level of deepness of our decision tree.
# the code used to perform this post pruning is directly borrowed from the following github notebook:
# https://github.com/krishnaik06/Post_Pruning_DecisionTre/blob/master/plot_cost_complexity_pruning.ipynb

# In[159]:


path = DecisionTree_Class_Model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


# In[160]:


ccp_alphas


# In[161]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))


# In[162]:


train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()


# In[163]:


# to avoid overfitting we want to choose alpha (a level of complexity) for which the accuracy on train set and test set is
# the best, while limiting the performance gap between those sets (we want to generalize well).
# Here it seems that a very small alpha of 0.003 could do the job (having an alpha between 0 and 0.01)


# In[164]:


clf = DecisionTreeClassifier(random_state=42, ccp_alpha=0.003)
clf.fit(X_train,y_train)


# In[165]:


ytrain_pred = clf.predict(X_train)
accuracy_score(y_train, ytrain_pred) # the overfitting is well reduced


# In[166]:


pred=clf.predict(X_test)
accuracy_score(y_test, pred) # the algorithm seems to generalize well and the overall performance on the test set is improved!




# # Part 3: Grouped analysis

# ## Import Data
# > NEED TO ASSIGN CATEGORIES TO VARIABLES AGAIN
# In[167]:


# modify categories of variables
for col_name in df_total.columns:
    # spot categorical variables but careful we don't want to change the category of boolean variables
    if (len(df_total[col_name].unique()) <= 100) and (len(df_total[col_name].unique()) > 2):
        df_total[col_name] = df_total[col_name].astype('category')

# manual rectifications
df_total['Outcome_stat'] = df_total['Outcome_stat'].astype('bool')
df_total['sex'] = df_total['sex'].astype('category')
df_total['Outcome'] = df_total['Outcome'].astype('category')
df_total['age_2020'] = df_total['age_2020'].astype('int')


# In[168]:


df_total.head(3)


# In[169]:


df_total.info()


# ## Departemental Analysis

# In[170]:


# Find the departement where the success rate was the highest
df_total.groupby('Nom du département')['Outcome_stat'].mean().sort_values(ascending=False)

# The results of the marketing campaign seem to be very disparate accross departements. We need to check how the sample population is distributed accross departements. Maybe we can use confidence intervals to assess more trutefully the marketing campain results in each departements
# In[171]:


# count how many sample individuals each departements have
repr_dep = df_total.groupby('Nom du département')['f_name'].count().sort_values(ascending=False) 


# In[172]:


# Top 10 departements which are the most represented in the sample
sns_plot = sns.barplot(repr_dep.values[:10], list(repr_dep.index[:10]), palette='BuPu_r')
fig = sns_plot.get_figure()
plt.xticks(rotation=90)
plt.title('Top 10 most represented departements in the sample')
plt.show()
fig.savefig("project-24-files/fig_bar_most_dep.png")


# The sample isn't evenly distributed accross departements. We have departements which are highly represented in the sample while others are way less present. Few possibilites:

# we could concentrate our departemental analysis only on departements where the number of individuals from the sample are higher than 100 for instance<u>**Look in more details to the distribution of the variables accross departements**:</u>
# In[173]:


# boxplot analysis for numerical variables
for col_name in df_total.columns:
    if is_numeric_dtype(df_total[col_name]):
        plt.boxplot(df_total.groupby('Nom du département')[col_name].mean())
        plt.title(col_name)
        plt.show()
# Only care about INHABITANTS, Outcome_stat, IS_STUDENT, age_2020, IS_club

# IS_STUDENT pretty evenly distributed with a few outliers maybe explained by the fact low representation of certain departements
# age_2020 pretty evenly distributed with again a few outliers
# outcome_stat not very evenly distributed (box is pretty long)
# INHABITANTS -> not very informative we know that some departements are way more populated than others.
# IS_CLUB pretty evenly distributed with a few outliers (again maybe due to the unequal representation of some departements)
# **distribution of categorical variables accross dep:**

# In[174]:


# visualize the distribution of categorical variables
# create a dataframe containing as col each category with the count for each dep
df_temp = df_total.groupby(['dep', 'sex'])['f_name'].count().unstack()
# obtain the proportions of men and women for each dep
df_temp2 = df_temp.div(df_temp.sum(axis=1), axis= 0)
# plot the distributions of women accross departements to view if sexes
# are evenly distributed accross departements
plt.boxplot(df_temp2.iloc[:, 0])
plt.title(df_temp.columns[0])
plt.show()


# In[175]:


# generalize this technique to all categorical variables with less than 10
# different categories


# In[176]:


# simple function to check if variable's type is categorical

def is_categorical(df_column):
    return df_column.dtype.name == 'category'


# In[177]:


for col_name in df_total.columns:
    if (len(df_total[col_name].unique()) < 10) & (is_categorical(df_total[col_name])):
        df_temp = df_total.groupby(['dep', col_name])['f_name'].count().unstack()
        df_temp2 = df_temp.div(df_temp.sum(axis=1), axis= 0)
        # We want to find the most 
        # frequent category for each variable so that we boxplot it
        # we sum over all the proportions of each categories accross 
        # dep to find the most frequent category.
        # PRETTY ROUGH DECOMPOSE THE CODE TO UNDERSTAND BETTER THE PROCESS
        prop_most_freq_cat = df_temp2[df_temp2.sum().sort_values(ascending=False)[:1].index]
        plt.boxplot(prop_most_freq_cat.values)
        plt.title(prop_most_freq_cat.columns[0])
        plt.show()
 

# We notice that:
# * The sex proportion is pretty evenly distributed accross departements in our sample data
# * The proportion of less educated people is fairly evenly distributed accross departeThe proportion of less educated people fairly evenly distributed accross departements in our sample.ments in our sample.
# * The proportion of "actifs" people is fairly evenly distributed accross departements in our sample.
# * The proportion of retiree people is fairly evenly distributed accross departements in our sample.
# * The proportions of families is also pretty evenly dsitributed accross departements in our sample data
# * As we have seen the campaign success isn't highly evenly distributed accross departments.-> look deeper into this with confidence intervals
# * The proportion of inactive people (people without a working contract) is more or less evenly distributed accross departements in our sample (but a bit less evenly distributed than sex and families so maybe look deeper into it)
# ## Regional Analysis

# ## Overview

# In[178]:


# Find the region where the success rate was the highest
df_total.groupby('Nom de la région')['Outcome_stat'].mean().sort_values(ascending=False)


# In[179]:


# count how many sample individuals each region have
repr_reg = df_total.groupby('Nom de la région')['f_name'].count().sort_values(ascending=False) 
repr_reg


# In[180]:


# Top 5 regions which are the most represented in the sample
sns.barplot(repr_reg.values[:5], list(repr_reg.index[:5]), palette='GnBu_r')
plt.xticks(rotation=90)
plt.title('Top 5 most represented regions in the sample')
plt.show()


# In[181]:


# Top 5 regions which are the less represented in the sample
sns.barplot(repr_reg.values[-5:], list(repr_reg.index[-5:]), palette='GnBu')
plt.xticks(rotation=90)
plt.title('Top 5 less represented regions in the sample')
plt.show()


# ### <u>Identification of the typical profile of an individual who responded postively to the marketing campaign in each region</u>:

# In[182]:


# median age of individuals who responsed postively to the MC per reg
df_total[df_total['Outcome_stat']].groupby('Nom de la région')['age_2020'].median()


# In[183]:


# typical activity of an individual for which the MC succeed per reg
df_total[df_total['Outcome_stat']].groupby(['Nom de la région', 'act'])['f_name'].count()


# ## North - South Analysis

# In[184]:


# from the regional visualization (below) we notice that the sucess rate is unevenly distributed among the "North"
# and the "South" of France. The success rate seems significantly low in the south of France in comparisson to the success rate
# in the North of France
# let's run a statistical test to verify this visual observation.


# In[185]:


l_south_reg = ['Nouvelle-Aquitaine', 'Occitanie',"Provence-Alpes-Côte d'Azur", "Auvergne-Rhône-Alpes", "Corse"]

# let's compute the avg success rate in "southern" regions of France
avg_sr_south = df_total[df_total['Nom de la région'].isin(l_south_reg)]['Outcome_stat'].mean()

# computation of the avg success rate in "nothern" regions of France
avg_sr_north = df_total[~df_total['Nom de la région'].isin(l_south_reg)]['Outcome_stat'].mean()

print('Average success rate in "southern" regions of France:', avg_sr_south)
print('Average success rate in "northern" regions of France:', avg_sr_north)


# In[186]:


# let's run a statistical student test for confirmation:
def run_student_test(sample_1, sample_2, alpha, feature):
    stat, p = ttest_ind(sample_1, sample_2)
    print('Statistics={stat:.{digits}f}, p={p:.{digits}f}'.format(stat=stat,
                                                                  digits=3,
                                                                  p=p))
    if p > alpha:
        print("There's no significant difference " +
              "of avg " + feature + " between the southern and nothern regions in our sample")
    else:
        print("There's a significant difference " +
              "of avg " + feature +" between the southern and nothern regions in our sample")


# In[187]:


arr_otc_south = np.array(df_total[df_total['Nom de la région'].isin(l_south_reg)]['Outcome_stat'])
arr_otc_north = np.array(~df_total[df_total['Nom de la région'].isin(l_south_reg)]['Outcome_stat'])
run_student_test(arr_otc_south, arr_otc_north, 0.05, "success rate")


# In[188]:


# Our t test confirms our visual observation. We observe a strong correlation. North or South localisation seems to coincide 
# with high or low success rate of the MC. Is this link causal ? We need to perform further analysis. In fact it is possible
# that there's a hidden variable wich is the true causal of the MC campaign and creates this correlation among North/South location
# and MC outcome. 
# Using our analysis in 2.2 we can focus on the variables we arleady know to explain significantly the Marketing Campaign 
# and verify their distribution accross the north and the south. For instance if we observe a significant difference in the
# distribution of age between North and South it is then possible that age of individuals truely explains the difference in 
# success rate and not the localisation. 


# In[189]:


# run statistical test about the distribution of age between North and South of FRance in the sample.


# In[190]:


arr_age_south = np.array(df_total[df_total['Nom de la région'].isin(l_south_reg)]['age_2020'])
arr_age_north = np.array(df_total[~df_total['Nom de la région'].isin(l_south_reg)]['age_2020'])
run_student_test(arr_age_south, arr_age_north, 0.05, 'age')


# In[191]:


# We observed in 2.2 that women were more likely to respond favorably to the MC than men; let's analyse the distribution of women
# accross north & south of france.
df_copy = df_total.copy()
# let's create boolean is_women to facilitate the making of the statistical test
df_copy['is_women'] = np.where(df_total['sex'] == 'Female', 1, 0)

arr_fem_south = np.array(df_copy[df_copy['Nom de la région'].isin(l_south_reg)]['is_women'])
arr_fem_north = np.array(df_copy[~df_copy['Nom de la région'].isin(l_south_reg)]['is_women'])
run_student_test(arr_fem_south, arr_fem_north, 0.05, 'proportion of female')


# In[192]:


# After analyzing the age and sex distribution it seems that there's no hidden variables between the correlation of north/south 
# localisation and the success rate of the Marketing campaign.


# In[193]:


############################################################################################
# Finally let's look at the distribution of the CSP ('Occupation_8') accross north and south


# In[194]:


# quick look at the distribution of occupations in the South of our sample.
vc_occup_south = df_total[df_total['Nom de la région'].isin(l_south_reg)]['Occupation_8'].value_counts()
vc_occup_south


# In[195]:


# quick look at the distribution of Occupations in the North in our sample
vc_occup_north = df_total[~df_total['Nom de la région'].isin(l_south_reg)]['Occupation_8'].value_counts() 
vc_occup_north


# In[196]:


# At a first glance we see that the order of frequency is the same in the north and in the south.
# let's investigate the proportions
print('Occupation distribution in the South of France based on our sample')
print(round(vc_occup_south / len(df_total[df_total['Nom de la région'].isin(l_south_reg)]), 2))
print('Occupation distribution in the North of France based on our sample')
print(round(vc_occup_north / len(df_total[~df_total['Nom de la région'].isin(l_south_reg)]), 2))


# In[197]:


# distribution of Occupation is almost identical accross North-South in our sample.


# Hence it seems that localisation between North and South is a determinant factor influencing the success rate of the Marketing Campaign.

# ### <u>Typical profile analysis</u>:

# In[198]:


#let's finish by looking at the typical profile of an individual for which the Marketing campaign was sucessful 
# accross North/South localisation
# create a new variable bool for this purpose is_south = 1 if Individual is in the South of France, = 0 if in the North.
df_copy['is_south'] = np.where(df_copy['Nom de la région'].isin(l_south_reg), 1, 0)
print('median age of people who responded postively to the MC in South of France:', 
      df_copy['age_2020'][(df_copy['Outcome_stat'] == 1) & (df_copy['is_south']==1)].median())
print('median age of people who responded postively to the MC in North of France:', 
      df_copy['age_2020'][(df_copy['Outcome_stat'] == 1) & (df_copy['is_south']==0)].median())

print('\nTypical occupation of people responding postively to the MC, in the North:\n',
       df_copy['Occupation_8'][(df_copy['Outcome_stat']==1) & (df_copy['is_south']==0)].mode().str[:])
print('\nTypical occupation of people responding postively to the MC, in the South:\n',
       df_copy['Occupation_8'][(df_copy['Outcome_stat']==1) & (df_copy['is_south']==1)].mode().str[:])


# In[199]:


def typical_profile_success_mc(var, cat, df):
    """
    var: which variable analyzed
    cat: True, analysis with the mode for categorical variables, False: median analysis for numerical variables
    df: dataframe used 
    """
    # var: which variable analyzed
    # cat: True, analysis with the mode for categorical variables, False: median analysis for numerical variables
    # df: dataframe used 
    if cat:
        print('\nTypical {} of people responding postively to the MC, in the North:\n'.format(var), 
               df[var][(df['Outcome_stat']==1) & (df['is_south']==0)].mode().str[:])
        print('\nTypical {} of people responding postively to the MC, in the South:\n'.format(var),
               df[var][(df['Outcome_stat']==1) & (df['is_south']==1)].mode().str[:])
        print(' ')
    else:
        print('\nMedian {} of people responding postively to the MC, in the North:\n'.format(var), 
               df[var][(df['Outcome_stat']==1) & (df['is_south']==0)].median())
        print('\nMedian {} of people responding postively to the MC, in the South:\n'.format(var),
               df[var][(df['Outcome_stat']==1) & (df['is_south']==1)].median())
        print(' ')


# In[200]:


# example of usage of the new function
typical_profile_success_mc('sex', cat=True, df=df_copy)


# In[201]:


num_var_to_analyze = ['age_2020'] 
cat_var_to_analyze = ['sex', 'contract', 'Occupation_8', 'act', 'Household_type']


# In[202]:


# typical profile analysis for cat variables
for var in cat_var_to_analyze:
    typical_profile_success_mc(var, cat=True, df=df_copy)


# ## Geographical representation of the results of the Marketing campaign accross departements

# ###  Departemental visualization

# In[203]:


# file geojson with coordinates of french departements, enables to have 
# an overlay on the map representing the departements: 
# link to download the file: https://france-geojson.gregoiredavid.fr/repo/departements.geojson
dep_geo = 'project-24-files/departements.json'


# In[204]:


df_folium = df_total.groupby('dep')['Outcome_stat'].mean()
# convert the serie to a dataframe object
df_folium = pd.DataFrame(df_folium.values, df_folium.index, 
                         columns=['avg_outc_per_dep'])
# change the type of index of df_folium to avoid complications
df_folium.index = df_folium.index.astype('object')

# add columns of data we want to visualize
df_folium['num_individuals_per_dep'] = df_total['dep'].value_counts()
df_folium['dep_code'] = df_folium.index

# convert zip_codes into string object so it matches the codes in the geojson file
df_folium['dep_code'] = df_folium['dep_code'].astype(str)


# In[205]:


m = folium.Map(location=[46.7, 1.7], zoom_start=5)

folium.Choropleth(
    geo_data=dep_geo,
    data=df_folium,
    columns=['dep_code', 'num_individuals_per_dep'],
    key_on='feature.properties.code',
    fill_color='BuPu',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of individuals per departement in sample'
).add_to(m)

m



# ### Regional visualization

# In[206]:


dep_geo = 'project-24-files/regions.json'


# In[207]:


df_folium = df_total.groupby('reg')['Outcome_stat'].mean()
# convert the serie to a dataframe object
df_folium = pd.DataFrame(df_folium.values, df_folium.index, 
                         columns=['avg_outc_per_reg'])
# change the type of index of df_folium to avoid complications
df_folium.index = df_folium.index.astype('object')

# add columns of data we want to visualize
df_folium['num_individuals_per_reg'] = df_total['reg'].value_counts()
df_folium['reg_code'] = df_folium.index

# convert zip_codes into string object so it matches the codes in the geojson file
df_folium['reg_code'] = df_folium['reg_code'].astype(str)


# <u>Map regional individuals distribution</u>:

# In[208]:


m = folium.Map(location=[46.7, 1.7], zoom_start=5)

folium.Choropleth(
    geo_data=dep_geo,
    data=df_folium,
    columns=['reg_code', 'num_individuals_per_reg'],
    key_on='feature.properties.code',
    fill_color='GnBu',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of individuals per reg in sample'
).add_to(m)

m


# <u>Map suceess rate of MC per region</u>:
# 

# In[209]:


m = folium.Map(location=[46.7, 1.7], zoom_start=5)

folium.Choropleth(
    geo_data=dep_geo,
    data=df_folium,
    columns=['reg_code', 'avg_outc_per_reg'],
    key_on='feature.properties.code',
    fill_color='GnBu',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Success rate of the Marketing Campaign per region'
).add_to(m)

m

