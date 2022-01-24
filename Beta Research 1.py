#!/usr/bin/env python
# coding: utf-8

# In[15]:


#Importing all the libraries we will need

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
import requests
import io
import matplotlib.pyplot as plt
import scipy.linalg as la
from numpy.linalg import eig
get_ipython().run_line_magic('matplotlib', 'inline')



##Note to self: remember can delete extra lines with "escdd" and add lines with "escb"


# In[2]:


#1. Setting up basic yfinance dataframe that will be used to collect raw data

#Stock Ticker Data Import
ticker = "AAPL"
start_date = datetime.datetime(2021,1,1)
end_date = datetime.datetime(2021,12,31)

#Empty Data Frame
stock_df = pd.DataFrame()

#Downloading Stock Price
stock = []
stock = yf.download("AAPL", start = start_date, end = end_date, progress = False)
stock_df = stock_df.append(stock,sort = False)

#Stock Ticker Column, useful for looping
stock_df['Symbol'] = "AAPL"
stock_df

#All S&P500 I would like to collect returns on
tickers = ["AAPL","MSFT","AMZN","TSLA","GOOGL","FB","GOOG","NVDA","JPM","JNJ","UNH","HD","PG","V","BAC","MA","PFE","XOM","DIS","AVGO","CSCO","NFLX","TMO","ADBE","COST","PEP","ABBV","ACN","ABT","CVX","KO","CMCSA","PYPL","CRM","VZ","INTC","WFC","QCOM","LLY","NKE","WMT","MCD","MRK","DHR","T","LOW","LIN","TXN","NEE","INTU","AMD","UNP","UPS","PM","MS","HON","MDT","AMAT","ORCL","SCHW","BMY","RTX","CVS","GS","SBUX","C","BLK","AMGN","IBM","AMT","CAT","ISRG","BA","PLD","NOW","TGT","GE","SPGI","AXP","MU","ANTM","DE","COP","ZTS","MMM","ADP","BKNG","LRCX","F","MDLZ","PNC","ADI","GM","SYK","TJX","MO","GILD","LMT","CB","TFC","MMC","CSX","CCI","EL","CME","USB","SHW","DUK","CHTR","EW","MRNA","CI","ICE","NSC","SO","BDX","CL","FIS","ITW","EQIX","TMUS","ETN","KLAC","APD","FISV","FDX","COF","AON","D","WM","REGN","PGR","HCA","MCO","BSX","NXPI","NOC","FCX","ILMN","ADSK","EMR","ECL","JCI","VRTX","EOG","DG","PSA","EXC","SPG","TEL","SNPS","APH","INFO","XLNX","IQV","ROP","ATVI","AIG","IDXX","GD","MET","KMB","CDNS","SLB","MCHP","ORLY","HUM","APTV","NEM","DXCM","BK","CARR","MSCI","CTSH","TT","CMG","DLR","A","MAR","PXD","HPQ","AEP","CNC","GPN","MSI","DOW","BAX","AZO","SRE","MPC","TROW","SIVB","DD","PRU","LHX","EBAY","FTNT","HLT","PAYX","PH","ALGN","GIS","SYY","PPG","TRV","O","YUM","STZ","ROST","ROK","ADM","SBAC","WELL","MCK","AFL","WBA","MTD","EA","XEL","DFS","IFF","FRC","AMP","STT","OTIS","MTCH","MNST","FAST","CBRE","KEYS","BIIB","CTAS","PSX","RMD","AVB","CTVA","ALL","VRSK","AJG","EFX","TDG","AME","FITB","NUE","WMB","PEG","KMI","DHI","CMI","EPAM","VLO","PCAR","DLTR","ANSS","ODFL","KR","AWK","TWTR","SWK","EQR","ES","DVN","WEC","WY","ED","CPRT","WST","BLL","ANET","LEN","ARE","ZBRA","GLW","WLTW","EXR","OXY","HSY","CDW","RSG","LH","VMC","OKE","ALB","CERN","MLM","ZBH","TSN","TER","TSCO","NTRS","KHC","DOV","SWKS","SYF","EXPE","LUV","FTV","DAL","IT","MAA","CHD","HBAN","LYB","ETSY","HIG","KEY","EIX","IR","URI","MKC","VRSN","RF","HES","STE","VFC","DRE","DTE","BBY","PPL","PKI","CFG","HAL","BKR","NDAQ","ESS","AEE","STX","GWW","FE","HPE","MTB","SBNY","EXPD","ETR","ULTA","CLX","WAT","VTR","FANG","XYL","NTAP","POOL","TRMB","ENPH","BR","COO","TYL","GPC","GNRC","GRMN","TDY","CTLT","MPWR","WDC","VIAC","RJF","FLT","KMX","ABC","PEAK","DGX","DPZ","NVR","DRI","TTWO","CE","IP","CMS","PFG","AMCR","WAB","AKAM","CZR","J","HOLX","MGM","AVY","BXP","QRVO","IEX","VTRS","CNP","CINF","UDR","TXT","RCL","PAYC","MAS","FDS","JBHT","CCL","CRL","K","OMC","CTRA","TECH","EMN","BBWI","LKQ","CAG","BRO","AAP","NLOK","LYV","AES","EVRG","PWR","SJM","TFX","ABMD","LNT","CAH","KIM","CF","IPG","BIO","UAL","WHR","MOS","CHRW","FBHS","PHM","MRO","MKTX","FFIV","HRL","FMC","ATO","INCY","IRM","CBOE","HAS","SEDG","PKG","HWM","CMA","LNC","LDOS","RHI","BF-B","JKHY","LVS","FOXA","PTC","HST","XRAY","WRK","CDAY","L","LUMN","AAL","SNA","BWA","REG","CTXS","WRB","TPR","PNR","ALLE","APA","AOS","JNPR","ZION","RE","HSIC","NI","MHK","NRG","SEE","UHS","LW","FRT","TAP","BEN","GL","CPB","AIZ","WYNN","NWSA","NWL","PBCT","DXC","IVZ","DISH","PNW","PVH","NCLH","PENN","HII","ROL","DVA","DISCK","NLSN","VNO","ALK","IPGP","RL","FOX","DISCA","UAA","GPS","UA","NWS"]


# In[3]:


#2. Defining a new dataframe to populate with our raw returns data

returns_df = pd.DataFrame(index = stock_df.index) #setting the dataframe to contain stocks data, although this will be appended
returns_df #calling the dataframe

returns_df = returns_df.drop(returns_df.index[0]) #dropping first index


# In[5]:


#3. Creating the returns data frame for 250 days of returns for 503 stocks

print(len(returns_df.index)) #Using this to keep track of the data calculation

for i in tickers:
        #Create an empty DataFrame that holds each of our stocks information
        stock_df = pd.DataFrame()

        # Download the Stock Price for each symbol, by passing in i into yf.download
        stock = []
        stock = yf.download(i, start = start_date, end = end_date, progress = False)
        stock_df = stock_df.append(stock,sort = False)
        print(len(stock_df))

        # Add the symbol column to help with looping, and create an empty list called "rows"
        stock_df['Symbol'] = ticker
        rows = [] #create rows array that will be appended to in the below iteration
        ## Now let's create a loop that iterates over all of our date-time's in the specified stock, and calculates
        ## the return on that day based on the difference between adjClose the day of and the adjClose the day before
        ## divided by the adjClose the day of
        #below we will create an array of appended return values (js) and once finished add the our columns (i)
        for j in range(1, 251):
            daily_return = ((stock_df['Adj Close'][j-1] - stock_df['Adj Close'][j])/stock_df['Adj Close'][j])  #return value calculation for each date
            rows.append(daily_return) ## Append this result to our list
        returns_df[i] = rows ## Append the row to the column specified by our ticker symbol

returns_df


# In[6]:


#4. Defining our Y_pxn Matrix that will contain the mean less return values for each stock

Ypxn_df = pd.DataFrame(returns_df)


# In[7]:


#5.  Now creating the Y_pxn Matrix

#Collect mean value for each column of returns 
for i in returns_df.columns:
    mean = returns_df.mean(axis=0,skipna = True)[i] #operation for finding mean
    rows = []
    #Now for each column (i) take each row value (i)(j) and subtract by its column (i) mean from the above for loop
    for j in returns_df.index:
        MeanlessReturns = returns_df[i][j] - mean
        rows.append(MeanlessReturns) 
    Ypxn_df[i] = rows 
Ypxn_df


# In[8]:


#6. Defining and creating the transpose of our Y_pxn matrix, the Y^T_nxp

Ytnxp_df = Ypxn_df.T
Ytnxp_df


# In[16]:


#7. Defining and creating our Sample Covariance Matrix S_pxp

Spxp = np.dot(Ypxn_df,Ytnxp_df)/503
Spxp


# In[22]:


#8. Calculate Eigenvectors

w,v=eig(Spxp)

print('E-vector', v)


# In[ ]:


#9. Review Factor Models


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




