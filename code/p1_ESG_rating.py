#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to compute the ESG index for all countries in the World

@author: geraldineconti
"""

import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import geopandas as gpd
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from iteration_utilities import flatten


mypath = '/Users/geraldineconti/Desktop/projet_Pictet/'

################################################################################

def combine_indicators(list_indicators,dict_indicators,weights):
    '''
        Combine indicators that belong to the same (sub)-group

        Reference :

        Args:
            list_indicators (list of string): indicator names to be used in the combination 
            dict_indicators (dict): dictionary containing all the indicators for all the countries
            weights (list of floats): weight values to be used in the combination

        Returns:
            df_weighted (Dataframe) : dataframe containing the combined indicator values (weighted, normalized) for all the countries 
    '''
       
    list_df = []
    list_df_nonan = []
    
    i = 0 
    
    # filter the indicators of interest and include them in the lists 
    for key, df in dict_indicators.items():
        if key not in list_indicators:
            continue
        list_df.append(df*weights[i])
        df_nonan = df.notna().astype(int)*weights[i]
        list_df_nonan.append(df_nonan)
        i = i+1
        
    # combined values of indicators 
    df_combined = reduce(lambda x, y: x.add(y, fill_value=0), list_df)
    
    # number of non-NaN entries 
    df_nentries = reduce(lambda x, y: x.add(y, fill_value=0), list_df_nonan)
        
    # normalization
    df_weighted = (df_combined/df_nentries)
    
    return df_weighted

################################################################################

def plot_world_map(df,suffix):
    
    '''
        Function to plot the World Map, superimposing the ESG index values 

        Reference :

        Args:
            df (Dataframe): Dataframe containing the ESG index values for all the countries
            suffix (string): string to define which case it is (E,S,G,ESG)

        Returns:
            -
    '''
    
    fig, ax = plt.subplots(figsize=(20, 12))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[(world.pop_est>0) & (world.name!="Antarctica")]
    
    # Bug fix 
    world.loc[world['name']=='France','iso_a3'] = 'FRA'
    world.loc[world['name']=='Norway','iso_a3'] = 'NOR'
    world.loc[world['name']=='Somalia','iso_a3'] = 'SOM'

    world_merged  = world.merge(df, on=['iso_a3'],how='left')
    world_merged = world_merged.sort_values(by=['2020'],ascending=False)
    
    world_merged.plot(column='2020',ax=ax,legend=True, cmap='hot', edgecolor='black',legend_kwds={'label': suffix+" Rating (2020)",'orientation': "horizontal"},missing_kwds={"color": "lightgrey","edgecolor": "red","hatch": "///","label": "Missing values"} )
    
    fig.savefig(mypath+'figures/ESG_rating_'+suffix+'.jpeg')
    
################################################################################

# read input file (https://databank.worldbank.org/source/environment-social-and-governance-(esg)-data#)
ESG_file = pd.read_excel(mypath+'/input/ESGEXCEL.xlsx',sheet_name=['Data','Series'])
df_topic = ESG_file['Series']
df_data = ESG_file['Data']

# to get the topic and name of indicator
df_merged = df_data.merge(df_topic, on=['Indicator Code','Indicator Name'],how='left')

# variable to tell how to read the indicator (large value : positive/negative ESG impact?)
df_merged['Trend'] = 1
list_negative_trend = []

# Environment ###################################################################

# list of all the indicators belonging to the Environment - Emissions sub-group 
list_E_emissions = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Environment: Emissions & pollution')])))
# weights associated with the indicators (default : all equal) 
weights_E_emissions = [1]*len(list_E_emissions)
# list of indicators that have a negative impact (the higher value, the worse it is for ESG) 
list_negative_trend.append(list_E_emissions)

list_E_endowment = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Environment: Natural capital endowment and management')])))
weights_E_endowment = [1]*len(list_E_endowment)
list_negative_trend.append(['EN.MAM.THRD.NO', 'ER.H2O.FWTL.ZS', 'NY.ADJ.DFOR.GN.ZS', 'NY.ADJ.DRES.GN.ZS'])

list_E_energy = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Environment: Energy use & security')])))
weights_E_energy = [1]*len(list_E_energy)
list_negative_trend.append(['EG.EGY.PRIM.PP.KD', 'EG.ELC.COAL.ZS', 'EG.IMP.CONS.ZS', 'EG.USE.COMM.FO.ZS', 'EG.USE.PCAP.KG.OE'])

list_E_risk = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Environment: Environment/climate risk & resilience')])))
weights_E_risk = [1]*len(list_E_risk)
list_negative_trend.append(['EN.CLC.CDDY.XD', 'EN.CLC.HEAT.XD', 'EN.CLC.MDAT.ZS', 'EN.CLC.PRCP.XD', 'EN.POP.DNST'])

list_E_food = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Environment: Food Security')])))
weights_E_food = [1]*len(list_E_food)

# Social #######################################################################

list_S_education = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Social: Education & skills')])))
weights_S_education = [1]*len(list_S_education)

list_S_employment = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Social: Employment')])))
weights_S_employment = [1]*len(list_S_employment)
list_negative_trend.append(['SL.TLF.0714.ZS', 'SL.UEM.TOTL.ZS'])

list_S_demography = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Social: Demography')])))
weights_S_demography = [1]*len(list_S_demography)

list_S_poverty = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Social: Poverty & Inequality')])))
weights_S_poverty = [1]*len(list_S_poverty)
list_negative_trend.append(list_S_poverty)

list_S_health = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Social: Health & Nutrition')])))
weights_S_health = [1]*len(list_S_health)
list_negative_trend.append(['SH.DTH.COMM.ZS', 'SH.DYN.MORT', 'SH.STA.OWAD.ZS', 'SN.ITK.DEFC.ZS'])

list_S_access = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Social: Access to Services')])))
weights_S_access = [1]*len(list_S_access)

# Governance ###################################################################

list_G_rights = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Governance: Human Rights')])))
weights_G_rights = [1]*len(list_G_rights)

list_G_effectiveness = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Governance: Government Effectiveness')])))
weights_G_effectiveness = [1]*len(list_G_effectiveness)

list_G_stability = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Governance: Stability & Rule of Law')])))
weights_G_stability = [1]*len(list_G_stability)

list_G_economic = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Governance: Economic Environment')])))
weights_G_economic = [1]*len(list_G_economic)

list_G_gender = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Governance: Gender')])))
weights_G_gender = [1]*len(list_G_gender)

list_G_innovation = sorted(list(set(df_merged['Indicator Code'][df_merged['Topic'].str.contains('Governance: Innovation')])))
weights_G_innovation = [1]*len(list_G_innovation)


# put -1 for indicators with negative impact on ESG 
list_negative_trend = list(flatten(list_negative_trend))
df_merged['Trend'][df_merged['Indicator Code'].isin(list_negative_trend)] = -1


# cleaning of dataframe - it contains the indicators, the trend and the country code, for all the countries 
df_merged = df_merged.drop(columns=['Country Name','Indicator Name','Short definition', 'Long definition', 'Unit of measure', 'Periodicity',
       'Base Period', 'Other notes', 'Aggregation method',
       'Limitations and exceptions', 'Notes from original source',
       'General comments', 'Source', 'Statistical concept and methodology',
       'Development relevance', 'Related source links', 'Other web links',
       'Related indicators', 'License Type'])


list_of_indicators = sorted(list(set(df_merged['Indicator Code'])))

dict_indicators = {}

# loop on indicators 
for i in range(len(list_of_indicators)):
    
    df_new = df_merged.copy()
    
    myind = list_of_indicators[i]
    
    # get one indicator 
    df_new = df_new[df_new['Indicator Code']==myind]
    df_new.index = df_new['Country Code']
    # trend is similar for all the countries - take only one value 
    indicator_trend = df_new['Trend'][0]

    # dataframe containing one indicator for all the countries - not normalized
    df_new = df_new.drop(columns=['Country Code','Indicator Code','Topic','2050','Trend'])
    
    
    # normalize the indicator values : use expanding window 
    # find max and min among countries for given year (expanding way)
    df_max = df_new.expanding(axis=1).max()
    df_min = df_new.expanding(axis=1).min()
    
    # get the min and max values for each column  
    df_max_new = df_max.max()
    df_min_new = df_min.min()
    
    # normalize the indicator to have it in the (0,1) range 
    df_new_norm = (df_new-df_min_new)/(df_max_new-df_min_new)
    
    # transpose results 
    df_new_norm = df_new_norm.T
    
    # linear interpolation to fill NaNs - test effect of this 
    df_new_norm = df_new_norm.interpolate()

    # transform negative indicator values : x --> 1-x
    if indicator_trend<0:
        df_new_norm = 1-df_new_norm

    # add information to dictionary 
    dict_indicators[myind] = df_new_norm

# dictionnary containing the E,S,G total index values   
dict_ESG = {}
list_ESG = ['environment','social','governance']
weights_ESG = [1,1,1]

# dictionary containig the E rating values for the Environement sub-categories 
dict_E = {}
list_E = ['emissions','endowment','energy','risk','food']
weights_E = [1,1,1,1,1]
dict_E['emissions'] = combine_indicators(list_E_emissions,dict_indicators, weights_E_emissions)  
dict_E['endowment'] = combine_indicators(list_E_endowment,dict_indicators, weights_E_endowment) 
dict_E['energy'] = combine_indicators(list_E_energy,dict_indicators, weights_E_energy)
dict_E['risk'] = combine_indicators(list_E_risk,dict_indicators, weights_E_risk)
dict_E['food'] = combine_indicators(list_E_food,dict_indicators, weights_E_food)
dict_ESG['environment'] = combine_indicators(list_E,dict_E,weights_E)

# dictionnary containing the S rating values for the Social sub-categories 
dict_S = {}
list_S = ['education','employment','demography','poverty','health','access']
weights_S = [1,1,1,1,1,1]
dict_S['education'] = combine_indicators(list_S_education,dict_indicators, weights_S_education)
dict_S['employment'] = combine_indicators(list_S_employment,dict_indicators, weights_S_employment)
dict_S['demography'] = combine_indicators(list_S_demography,dict_indicators, weights_S_demography)
dict_S['poverty'] = combine_indicators(list_S_poverty,dict_indicators, weights_S_poverty)
dict_S['health'] = combine_indicators(list_S_health,dict_indicators, weights_S_health)
dict_S['access'] = combine_indicators(list_S_access,dict_indicators, weights_S_access)
dict_ESG['social'] = combine_indicators(list_S,dict_S,weights_S)

# dictionnary containing the G rating values for the Governance sub-categories 
dict_G = {}
list_G = ['rights','effectiveness','stability','economic', 'gender', 'innovation']
weights_G = [1,1,1,1,1,1]
dict_G['rights'] = combine_indicators(list_G_rights,dict_indicators, weights_G_rights)
dict_G['effectiveness'] = combine_indicators(list_G_effectiveness,dict_indicators, weights_G_effectiveness)
dict_G['stability'] = combine_indicators(list_G_stability,dict_indicators, weights_G_stability)
dict_G['economic'] = combine_indicators(list_G_economic,dict_indicators, weights_G_economic)
dict_G['gender'] = combine_indicators(list_G_gender,dict_indicators, weights_G_gender)
dict_G['innovation'] = combine_indicators(list_G_innovation,dict_indicators, weights_G_innovation)
dict_ESG['governance'] = combine_indicators(list_G,dict_G,weights_G)

# Dataframe with the ESG rating values 
df_ESG = combine_indicators(list_ESG,dict_ESG,weights_ESG)

# filter to get the last result available
df_ESG_2020 = pd.DataFrame(df_ESG.loc['2020'].sort_values())
df_ESG_2020['iso_a3'] = df_ESG_2020.index

# to round values to two decimals 
# TODO : group results into (5) categories 
df_ESG_2020 = df_ESG_2020.round(2)

df_E_2020 = pd.DataFrame(dict_ESG['environment'].loc['2020'].sort_values())
df_E_2020['iso_a3'] = df_E_2020.index

df_S_2020 = pd.DataFrame(dict_ESG['social'].loc['2020'].sort_values())
df_S_2020['iso_a3'] = df_S_2020.index

df_G_2020 = pd.DataFrame(dict_ESG['governance'].loc['2020'].sort_values())
df_G_2020['iso_a3'] = df_G_2020.index

plot_world_map(df_ESG_2020,'ESG')
plot_world_map(df_E_2020,'Environment')
plot_world_map(df_S_2020,'Social')
plot_world_map(df_G_2020,'Governance')

print('2020 result :')
print(df_ESG_2020[-10:])


print('END OF PROGRAM')

