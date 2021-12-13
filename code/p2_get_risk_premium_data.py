# -*- coding: utf-8 -*-
"""
Spyder Editor

Get data from Worldbank for risk premium prediction 

"""

import wbdata    

################################################################################
# From https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp711.pdf : 

# GDP per capita (positive impact) - NY.GDP.PCAP.CD

# Real GDP growth (positive) - NY.GDP.MKTP.KD.ZG

# Inflation (uncertain)

# unemployment (negative) - SL.UEM.TOTL.NE.ZS, SL.UEM.TOTL.ZS

# government debt (negative) - GB.DOD.TOTL.CN
 
# fiscal balance (positive) - GC.BAL.CASH.CD

# government effectiveness (positive) # GE.EST

# external debt (negative) # DB.DOD.DLXF.CD 

# foreign reserves (positive) # IMPCOV

# current account balance (uncertain)

# default history (negative)

indicators = {"NY.GDP.PCAP.CD":"GPDpercapita",
              "NY.GDP.MKTP.KD.ZG":"GDPgrowth",
              "SL.UEM.TOTL.NE.ZS":"unemployment",
              ###"GB.DOD.TOTL.CN":"debt"
              ###"GC.BAL.CASH.CD":"fiscal_balance"
              #"GE.EST":"effectiveness",
              ##"DB.DOD.DLXF.CD":"external_debt"
              "IMPCOV":"foreign_reserves"
              #"EMBIG":"spread"
              }


df = wbdata.get_dataframe(indicators, country="BRA", freq='M',convert_date=True)   
print(df)
################################################################################
# From WorldBank 

#indicators = wbdata.get_indicator(source=15)
# CORENS         Core CPI,not seas.adj,,,
# CORESA         Core CPI,seas.adj,,,
# CPTOTNSXN      CPI Price, nominal
# CPTOTSAXMZGY   CPI Price, % y-o-y, median weighted, seas. adj.
# CPTOTSAXN      CPI Price, nominal, seas. adj.
# CPTOTSAXNZGY   CPI Price, % y-o-y, nominal, seas. adj.
# DMGSRMRCHNSCD  Imports Merchandise, Customs, current US$, millions
# DMGSRMRCHNSKD  Imports Merchandise, Customs, constant US$, millions
# DMGSRMRCHNSXD  Imports Merchandise, Customs, Price, US$
# DMGSRMRCHSACD  Imports Merchandise, Customs, current US$, millions, seas. adj.
# DMGSRMRCHSAKD  Imports Merchandise, Customs, constant US$, millions, seas. adj.
# DMGSRMRCHSAXD  Imports Merchandise, Customs, Price, US$, seas. adj.
# DPANUSLCU      Official exchange rate, LCU per USD, period average
# DPANUSSPB      Exchange rate, new LCU per USD extended backward, period average
# DPANUSSPF      Exchange rate, old LCU per USD extended forward, period average
# DSTKMKTXD      Stock Markets, US$
# DSTKMKTXN      Stock Markets, LCU
# DXGSRMRCHNSCD  Exports Merchandise, Customs, current US$, millions
# DXGSRMRCHNSKD  Exports Merchandise, Customs, constant US$, millions
# DXGSRMRCHNSXD  Exports Merchandise, Customs, Price, US$
# DXGSRMRCHSACD  Exports Merchandise, Customs, current US$, millions, seas. adj.
# DXGSRMRCHSAKD  Exports Merchandise, Customs, constant US$, millions, seas. adj.
# DXGSRMRCHSAXD  Exports Merchandise, Customs, Price, US$, seas. adj.
# EMBIG          J.P. Morgan Emerging Markets Bond Spread (EMBI+),,,,
# EMBIGI         J.P. Morgan Emerging Markets Bond Index(EMBI+),,,,
# IMPCOV         Foreign Reserves, Months Import Cover, Goods
# IPTOTNSKD      Industrial Production, constant US$
# IPTOTSAKD      Industrial Production, constant US$, seas. adj.
# NEER           Nominal Effecive Exchange Rate
# NYGDPMKTPSACD  GDP,current US$,millions,seas. adj.,
# NYGDPMKTPSACN  GDP,current LCU,millions,seas. adj.,
# NYGDPMKTPSAKD  GDP,constant 2010 US$,millions,seas. adj.,
# NYGDPMKTPSAKN  GDP,constant 2010 LCU,millions,seas. adj.,
# REER           Real Effective Exchange Rate
# RETSALESSA     Retail Sales Volume,Index,,,
# TOT            Terms of Trade
# TOTRESV        Total Reserves
# UNEMPSA_       Unemployment rate,Percent,,,

indicators = {"CORENS":"CPI"}
df2 = wbdata.get_dataframe(indicators, country="BRA", freq='M',convert_date=True)   
print(df2)
