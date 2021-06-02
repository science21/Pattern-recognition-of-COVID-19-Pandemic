# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:38:41 2020

@author: Jianyong
"""

import os
path="D:\Data_analysis\Python\COVID19"
os.chdir(path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the original dataset
orgdata =pd.read_csv('time_series_covid19_confirmed_US.csv')
orgdata=orgdata.rename(columns={'Province_State': 'State'})

#get data from March 1th
npdata=orgdata.iloc[:,50:].values 

m,n=npdata.shape

# get daily new case data
npdailycase=np.zeros(npdata.shape) 
npdailycase[:,0] =npdata[:,0]
for i in list(range(m)):
    for j in list(range(1,n)):
        npdailycase[i,j]=npdata[i,j]-npdata[i, j-1]
        
# get the date from the column variables        
col=npdata=orgdata.iloc[:,50:].columns       
dailycase =pd.DataFrame(data =npdailycase, columns=col)
dailycase[dailycase<0]=0  #replace negative value as zero;

# First 11 columns are not covid-19 case data, but the information of COVID-19, such as locations
notedata=orgdata.iloc[:,0:11]

# Daily cases for whole US, including US territory 
dailycasefull=notedata.join(dailycase)

dailycasefull.to_csv('US_dailycase.csv')

# Create a spatial dataset using the cases in the last day of the dataset
lastcol=orgdata.iloc[:, -1]
spacedata=notedata.join(lastcol)
spacedata.to_csv('spacedata.csv')

# Get times series data for each states
timedata =dailycasefull.groupby('State').sum()
timedata.to_csv('timedata.csv')
# Remove the data from US territory, only states and DC data are included
nonstate=['American Samoa','Diamond Princess','Grand Princess', 'Guam','Northern Mariana Islands','Puerto Rico','Virgin Islands']

statetimedata=timedata[~timedata.index.isin(nonstate)]

statedaily=statetimedata.iloc[:,5:]
statedaily.to_csv('statedaily.csv')

############ K-means clustering

from sklearn.cluster import KMeans

# Find optimal K clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(statedaily)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method (Whole period)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(statedaily)

unique, counts = np.unique(y_kmeans, return_counts=True)
dict(zip(unique, counts))

#### PCA ###
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
pca = PCA(.78)
pca.fit(statedaily)
X = pca.transform(statedaily)
scaler=MinMaxScaler()
XS = scaler.fit_transform (X)
X=np.log10(XS+0.1) # scale the data for better visualization



#### Visualising the clusters

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'cyan', label = 'Cluster 3')



states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE ", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT ", "VA", "WA", "WV", " ", "WY"]
for i, txt in enumerate(states):
    plt.annotate(txt, (X[i,0], X[i,1]))


plt.annotate("WI", (X[49,0], X[49,1]-0.04))
plt.title('Clusters of COVID-19 curves by states (whole period)')
plt.xlabel('First principal component of PCA (scaled)')
plt.ylabel('Second principal component of PCA (scaled)')
plt.grid(color='y', linestyle='-', linewidth=0.2)
plt.legend()
plt.show()


####### Cluster analysis at Phase 1
phase1 =statedaily.loc[:, :'5/31/20']

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(phase1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(phase1)

unique, counts = np.unique(y_kmeans, return_counts=True)
dict(zip(unique, counts))

#### PCA ###
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
pca = PCA(.95)
pca.fit(phase1)
X = pca.transform(phase1)
scaler=MinMaxScaler()
XS = scaler.fit_transform (X)
X=np.log10(XS+0.1)


### Visualising the clusters

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'cyan', label = 'Cluster 3')
#plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 20, c = 'yellow', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 20, c = 'purple', label = 'Cluster 5')

states = ["AL", "AK", "AZ", "AR", " ", "CO", "CT", "DC", "DE ", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT ", "VA", "WA", "WV", "WI", "WY"]
for i, txt in enumerate(states):
    plt.annotate(txt, (X[i,0], X[i,1]))

plt.annotate("CA", (X[4,0], X[4,1]-0.04))

plt.title('Clusters of COVID-19 curves by states (the early phase)')
plt.xlabel('First principal component of PCA (scaled)')
plt.ylabel('Second principal component of PCA (scaled)')
plt.grid(color='y', linestyle='-', linewidth=0.2)
plt.legend()
plt.show()




###### Cluster analysis at Phase2
phase2 =statedaily.loc[:, '6/1/20':'9/30/20']

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(phase2)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(phase2)

unique, counts = np.unique(y_kmeans, return_counts=True)
dict(zip(unique, counts))

#### PCA ###
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
pca = PCA(.90)
pca.fit(phase2)
X = pca.transform(phase2)
scaler=MinMaxScaler()
XS = scaler.fit_transform (X)
X=np.log10(XS+0.1)

# Visualising the clusters

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 1')
#plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'green', label = 'Cluster 2')

states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE ", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", " ", 
          "SD", " ", "TX", "UT", "VT ", "VA", "WA", "WV", "WI", "WY"]
for i, txt in enumerate(states):
    plt.annotate(txt, (X[i,0], X[i,1]))

plt.annotate("TN", (X[42,0], X[42,1]-0.04))
plt.annotate("SC", (X[40,0], X[40,1]+0.04))
plt.title('Clusters of COVID-19 curves by states (the mid-phase)')
plt.xlabel('First principal component of PCA (scaled)')
plt.ylabel('Second principal component of PCA (scaled)')
plt.grid(color='y', linestyle='-', linewidth=0.2)
plt.legend()
plt.show()




###### Cluster analysis at Phase 3

phase3 =statedaily.loc[:, '10/1/20':]

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(phase3)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(phase3)

unique, counts = np.unique(y_kmeans, return_counts=True)
dict(zip(unique, counts))

#### PCA ###
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
pca = PCA(.82)
pca.fit(phase3)
X = pca.transform(phase3)
scaler=MinMaxScaler()
XS = scaler.fit_transform (X)
X=np.log10(XS+0.1)


# Visualising the clusters

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 20, c = 'red', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'cyan', label = 'Cluster 4')


states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE ", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT ", "VA", "WA", "WV", "WI", "WY"]
for i, txt in enumerate(states):
    plt.annotate(txt, (X[i,0], X[i,1]))

#plt.annotate("TN", (X[42,0], X[42,1]-0.04))

plt.title('Clusters of COVID-19 curves by states (the late phase)')
plt.xlabel('First principal component of PCA (scaled)')
plt.ylabel('Second principal component of PCA (scaled)')
plt.grid(color='y', linestyle='-', linewidth=0.2)
plt.legend()
plt.show()



###############
########## Time series analysis ##################
from statsmodels.tsa.seasonal import seasonal_decompose



Tstatedaily=statedaily.transpose()
Tstatedaily['US']=Tstatedaily.sum(axis=1)




# US total case---- ST decompositon
lg_us =np.log10(Tstatedaily['US'])
st_us = seasonal_decompose(lg_us, model='additive', period=7)
tus =st_us.trend.fillna(0)
us_trend=pd.concat([lg_us, tus ], axis=1).iloc[0:284, :]
us_trend1 =us_trend.iloc[15:284,:]



n=len(us_trend)
x=list(range(0, n))

Date=us_trend.index
Date1=[Date[0], Date[0], Date[50],Date[100], Date[150],Date[200],Date[250], '12/25/20']


fig = plt.figure()
ax = fig.add_axes([0.12, 0.1, 0.85, 0.8]) # main axes

ax.plot(x, us_trend)
ax.set_xlabel('Date')
ax.set_ylabel('The number of cases (log10-transformed)')
ax.set_title('Daily COVID-19 cases and trends in the US')
ax.set_xticklabels(Date1)
ax.set_ylim([4,5.5])
ax.legend( labels= ['Observed', 'Trend' ])
plt.show()


############# Trend for each state
Tstatedaily=statedaily.transpose()
Tstatedaily['US']=Tstatedaily.sum(axis=1)
Tstatedaily.replace(0,1, inplace=True)
Tstatedaily=np.log10(Tstatedaily)


# CA -1
st_ca = seasonal_decompose(Tstatedaily['California'], model='additive', period=7)

# TX -2
st_tx = seasonal_decompose(Tstatedaily['Texas'], model='additive', period=7)

# FL -3
st_fl = seasonal_decompose(Tstatedaily['Florida'], model='additive', period=7)

# IL -4
st_il = seasonal_decompose(Tstatedaily['Illinois'], model='additive', period=7)

# NY -5
st_ny = seasonal_decompose(Tstatedaily['New York'], model='additive', period=7)

# OH -6
st_oh = seasonal_decompose(Tstatedaily['Ohio'], model='additive', period=7)

#GA-7
st_ga = seasonal_decompose(Tstatedaily['Georgia'], model='additive', period=7)

#PA-8
st_pa = seasonal_decompose(Tstatedaily['Pennsylvania'], model='additive', period=7)

# TN-9
st_tn = seasonal_decompose(Tstatedaily['Tennessee'], model='additive', period=7)

# MI -10
st_mi = seasonal_decompose(Tstatedaily['Michigan'], model='additive', period=7)

# NC -11
st_nc = seasonal_decompose(Tstatedaily['North Carolina'], model='additive', period=7)


# IN -12
st_in = seasonal_decompose(Tstatedaily['Indiana'], model='additive', period=7)

# WI -13
st_wi = seasonal_decompose(Tstatedaily['Wisconsin'], model='additive', period=7)

# AZ -14
st_az = seasonal_decompose(Tstatedaily['Arizona'], model='additive', period=7)

# NJ -15
st_nj = seasonal_decompose(Tstatedaily['New Jersey'], model='additive', period=7)

# MN -16
st_mn = seasonal_decompose(Tstatedaily['Minnesota'], model='additive', period=7)

# MO -17
st_mo = seasonal_decompose(Tstatedaily['Missouri'], model='additive', period=7)

# AL -18
st_al = seasonal_decompose(Tstatedaily['Alabama'], model='additive', period=7)


# VA -19
st_va = seasonal_decompose(Tstatedaily['Virginia'], model='additive', period=7)

# MA -20
st_ma = seasonal_decompose(Tstatedaily['Massachusetts'], model='additive', period=7)

# CO -21
st_co = seasonal_decompose(Tstatedaily['Colorado'], model='additive', period=7)

# LA -22
st_la = seasonal_decompose(Tstatedaily['Louisiana'], model='additive', period=7)

# SC -23
st_sc = seasonal_decompose(Tstatedaily['South Carolina'], model='additive', period=7)

# IA -24
st_ia = seasonal_decompose(Tstatedaily['Iowa'], model='additive', period=7)

# OK -25
st_ok = seasonal_decompose(Tstatedaily['Oklahoma'], model='additive', period=7)

# MD -26
st_md = seasonal_decompose(Tstatedaily['Maryland'], model='additive', period=7)

# UT -27
st_ut = seasonal_decompose(Tstatedaily['Utah'], model='additive', period=7)

# KY -28
st_ky = seasonal_decompose(Tstatedaily['Kentucky'], model='additive', period=7)

# WA -29
st_wa = seasonal_decompose(Tstatedaily['Washington'], model='additive', period=7)

# NV -30
st_nv = seasonal_decompose(Tstatedaily['Nevada'], model='additive', period=7)


# AR -31
st_ar = seasonal_decompose(Tstatedaily['Arkansas'], model='additive', period=7)

# MS -32
st_ms = seasonal_decompose(Tstatedaily['Mississippi'], model='additive', period=7)

# KS -33
st_ks = seasonal_decompose(Tstatedaily['Kansas'], model='additive', period=7)

# CT -34
st_ct = seasonal_decompose(Tstatedaily['Connecticut'], model='additive', period=7)

# NE -35
st_ne = seasonal_decompose(Tstatedaily['Nebraska'], model='additive', period=7)


# NM -36
st_nm = seasonal_decompose(Tstatedaily['New Mexico'], model='additive', period=7)

# ID -37
st_id = seasonal_decompose(Tstatedaily['Idaho'], model='additive', period=7)


# OR -38
st_or = seasonal_decompose(Tstatedaily['Oregon'], model='additive', period=7)

# SD -39
st_sd = seasonal_decompose(Tstatedaily['South Dakota'], model='additive', period=7)

# ND -40
st_nd = seasonal_decompose(Tstatedaily['North Dakota'], model='additive', period=7)

# RI -41
st_ri = seasonal_decompose(Tstatedaily['Rhode Island'], model='additive', period=7)

# MT -42
st_mt = seasonal_decompose(Tstatedaily['Montana'], model='additive', period=7)

# WV -43
st_wv = seasonal_decompose(Tstatedaily['West Virginia'], model='additive', period=7)

# DE -44
st_de = seasonal_decompose(Tstatedaily['Delaware'], model='additive', period=7)

# AK -45
st_ak = seasonal_decompose(Tstatedaily['Alaska'], model='additive', period=7)

# WY -46
st_wy = seasonal_decompose(Tstatedaily['Wyoming'], model='additive', period=7)

# NH -47
st_nh = seasonal_decompose(Tstatedaily['New Hampshire'], model='additive', period=7)

# ME -48
st_me = seasonal_decompose(Tstatedaily['Maine'], model='additive', period=7)

# HI -49
st_hi = seasonal_decompose(Tstatedaily['Hawaii'], model='additive', period=7)

# VT -50
st_vt = seasonal_decompose(Tstatedaily['Vermont'], model='additive', period=7)




tca =st_ca.trend.fillna(0)
ttx =st_tx.trend.fillna(0)
tfl =st_fl.trend.fillna(0)
til =st_il.trend.fillna(0)
tny =st_ny.trend.fillna(0)

tga =st_ga.trend.fillna(0)
toh =st_oh.trend.fillna(0)
tpa =st_pa.trend.fillna(0)
tmi =st_mi.trend.fillna(0)
twi =st_wi.trend.fillna(0)

ttn =st_tn.trend.fillna(0)
tnc =st_nc.trend.fillna(0)
tin =st_in.trend.fillna(0)
tnj =st_nj.trend.fillna(0)
taz =st_az.trend.fillna(0)


tmn =st_mn.trend.fillna(0)
tmo =st_mo.trend.fillna(0)
tal =st_al.trend.fillna(0)
tco =st_co.trend.fillna(0)
tma =st_ma.trend.fillna(0)

tva =st_va.trend.fillna(0)
tla =st_la.trend.fillna(0)
tia =st_ia.trend.fillna(0)
tsc =st_sc.trend.fillna(0)
tok =st_ok.trend.fillna(0)

tmd =st_md.trend.fillna(0)
tut =st_ut.trend.fillna(0)
tky =st_ky.trend.fillna(0)
twa =st_wa.trend.fillna(0)
tks =st_ks.trend.fillna(0)


tar =st_ar.trend.fillna(0)
tnv =st_nv.trend.fillna(0)
tms =st_ms.trend.fillna(0)
tne =st_ne.trend.fillna(0)
tct =st_ct.trend.fillna(0)

tid =st_id.trend.fillna(0)
tnm =st_nm.trend.fillna(0)
tri =st_ri.trend.fillna(0)
tor =st_or.trend.fillna(0)
tsd =st_sd.trend.fillna(0)


tnd =st_nd.trend.fillna(0)
tmt =st_mt.trend.fillna(0)
twv =st_wv.trend.fillna(0)
tde =st_de.trend.fillna(0)
tak =st_ak.trend.fillna(0)

twy =st_wy.trend.fillna(0)
tnh =st_nh.trend.fillna(0)
thi =st_hi.trend.fillna(0)
tme =st_me.trend.fillna(0)
tvt =st_vt.trend.fillna(0)



trend5=pd.concat([tca, ttx, tfl, til,tny ], axis=1).iloc[0:284, :]
trend10=pd.concat([tga, toh,tpa, tmi,twi ], axis=1).iloc[0:284, :]
trend15=pd.concat([ttn, tnc,tin, tnj,taz ], axis=1).iloc[0:284, :]
trend20=pd.concat([tmn, tmo,tal, tco,tma ], axis=1).iloc[0:284, :]
trend25=pd.concat([tva, tla,tia, tsc,tok ], axis=1).iloc[0:284, :]

trend30=pd.concat([tmd, tut,tky, twa,tks ], axis=1).iloc[0:284, :]
trend35=pd.concat([tar, tnv,tms, tne,tct ], axis=1).iloc[0:284, :]
trend40=pd.concat([tid, tnm,tri, tor,tsd ], axis=1).iloc[0:284, :]
trend45=pd.concat([tnd, tmt,twv, tde,tak ], axis=1).iloc[0:284, :]
trend50=pd.concat([twy, tnh,thi,tme, tvt ], axis=1).iloc[0:284, :]

### Plot trend of COVID-19 IN 50 States

n=len(trend5)
x=list(range(0, n))
Date=trend5.index
Date1=[Date[0], Date[0], Date[50],Date[100], Date[150],Date[200],Date[250], '12/25/20']


fig, ([ax1,ax2], [ax3, ax4],[ax5,ax6], [ax7, ax8],[ax9,ax10]) = plt.subplots(nrows=5, ncols=2, sharex=False)
fig.text(0.5, 0.04, '         Date', ha='center')
fig.text(0.04, 0.5, 'The number of daily cases (log 10 transformed)', va='center', rotation='vertical')

ax1.plot(x, trend5)
ax1.legend(loc=0, ncol=3, prop={'size': 7},labels= ['CA', 'TX', 'FL','IL', 'NY'])


ax2.plot(x, trend10)
ax2.legend(loc=0, ncol=3, prop={'size': 7},labels= ['GA', 'OH', 'PA','MI', 'WI'])


ax3.plot(x, trend15)
ax3.legend(loc=0, ncol=3, prop={'size': 7},labels= ['TN','NC', 'IN', 'NJ', 'AZ'])

ax4.plot(x, trend20)
ax4.legend(loc=0, ncol=3, prop={'size': 7},labels= ['MN','MO', 'AL', 'CO', 'MA'])


ax5.plot(x, trend25)
ax5.legend(loc=0, ncol=3,prop={'size': 7}, labels= ['VA','LA', 'IA', 'SC', 'OK'])


ax6.plot(x, trend30)
ax6.legend(loc=0, ncol=3, prop={'size': 7},labels= ['MD','UT', 'KY', 'WA', 'KS'])

ax7.plot(x, trend35)
ax7.legend(loc=0, ncol=3,prop={'size': 7}, labels= ['AR','NV', 'MS', 'NE', 'CT'])

ax8.plot(x, trend40)
ax8.legend(loc=0, ncol=3,prop={'size': 7},labels= ['ID','NM', 'RI', 'OR', 'SD'])

ax9.plot(x, trend45)
ax9.legend(loc=0, ncol=3, prop={'size': 7}, labels= ['ND','MT', 'WV', 'DE', 'AK'])

ax10.plot(x, trend50)
ax10.legend(loc=0, ncol=3, prop={'size': 7},labels= ['WY','NH', 'HI', 'ME', 'VT'])

ax1.set_xticklabels(Date1, minor=False)
ax2.set_xticklabels(Date1, minor=False)
ax3.set_xticklabels(Date1, minor=False)
ax4.set_xticklabels(Date1, minor=False)
ax5.set_xticklabels(Date1, minor=False)
ax6.set_xticklabels(Date1, minor=False)
ax7.set_xticklabels(Date1, minor=False)
ax8.set_xticklabels(Date1, minor=False)
ax9.set_xticklabels(Date1, minor=False)
ax10.set_xticklabels(Date1, minor=False)


plt.show()


############################################################### NOT SCALE THE DATA #############


############# Trend for each state
Tstatedaily=statedaily.transpose()
Tstatedaily['US']=Tstatedaily.sum(axis=1)
Tstatedaily.replace(0,1, inplace=True)
Tstatedaily=Tstatedaily



# CA -1
st_ca = seasonal_decompose(Tstatedaily['California'], model='additive', period=7)

# TX -2
st_tx = seasonal_decompose(Tstatedaily['Texas'], model='additive', period=7)

# FL -3
st_fl = seasonal_decompose(Tstatedaily['Florida'], model='additive', period=7)

# IL -4
st_il = seasonal_decompose(Tstatedaily['Illinois'], model='additive', period=7)

# NY -5
st_ny = seasonal_decompose(Tstatedaily['New York'], model='additive', period=7)

# OH -6
st_oh = seasonal_decompose(Tstatedaily['Ohio'], model='additive', period=7)

#GA-7
st_ga = seasonal_decompose(Tstatedaily['Georgia'], model='additive', period=7)

#PA-8
st_pa = seasonal_decompose(Tstatedaily['Pennsylvania'], model='additive', period=7)

# TN-9
st_tn = seasonal_decompose(Tstatedaily['Tennessee'], model='additive', period=7)

# MI -10
st_mi = seasonal_decompose(Tstatedaily['Michigan'], model='additive', period=7)

# NC -11
st_nc = seasonal_decompose(Tstatedaily['North Carolina'], model='additive', period=7)


# IN -12
st_in = seasonal_decompose(Tstatedaily['Indiana'], model='additive', period=7)

# WI -13
st_wi = seasonal_decompose(Tstatedaily['Wisconsin'], model='additive', period=7)

# AZ -14
st_az = seasonal_decompose(Tstatedaily['Arizona'], model='additive', period=7)

# NJ -15
st_nj = seasonal_decompose(Tstatedaily['New Jersey'], model='additive', period=7)

# MN -16
st_mn = seasonal_decompose(Tstatedaily['Minnesota'], model='additive', period=7)

# MO -17
st_mo = seasonal_decompose(Tstatedaily['Missouri'], model='additive', period=7)

# AL -18
st_al = seasonal_decompose(Tstatedaily['Alabama'], model='additive', period=7)


# VA -19
st_va = seasonal_decompose(Tstatedaily['Virginia'], model='additive', period=7)

# MA -20
st_ma = seasonal_decompose(Tstatedaily['Massachusetts'], model='additive', period=7)

# CO -21
st_co = seasonal_decompose(Tstatedaily['Colorado'], model='additive', period=7)

# LA -22
st_la = seasonal_decompose(Tstatedaily['Louisiana'], model='additive', period=7)

# SC -23
st_sc = seasonal_decompose(Tstatedaily['South Carolina'], model='additive', period=7)

# IA -24
st_ia = seasonal_decompose(Tstatedaily['Iowa'], model='additive', period=7)

# OK -25
st_ok = seasonal_decompose(Tstatedaily['Oklahoma'], model='additive', period=7)

# MD -26
st_md = seasonal_decompose(Tstatedaily['Maryland'], model='additive', period=7)

# UT -27
st_ut = seasonal_decompose(Tstatedaily['Utah'], model='additive', period=7)

# KY -28
st_ky = seasonal_decompose(Tstatedaily['Kentucky'], model='additive', period=7)

# WA -29
st_wa = seasonal_decompose(Tstatedaily['Washington'], model='additive', period=7)

# NV -30
st_nv = seasonal_decompose(Tstatedaily['Nevada'], model='additive', period=7)


# AR -31
st_ar = seasonal_decompose(Tstatedaily['Arkansas'], model='additive', period=7)

# MS -32
st_ms = seasonal_decompose(Tstatedaily['Mississippi'], model='additive', period=7)

# KS -33
st_ks = seasonal_decompose(Tstatedaily['Kansas'], model='additive', period=7)

# CT -34
st_ct = seasonal_decompose(Tstatedaily['Connecticut'], model='additive', period=7)

# NE -35
st_ne = seasonal_decompose(Tstatedaily['Nebraska'], model='additive', period=7)


# NM -36
st_nm = seasonal_decompose(Tstatedaily['New Mexico'], model='additive', period=7)

# ID -37
st_id = seasonal_decompose(Tstatedaily['Idaho'], model='additive', period=7)


# OR -38
st_or = seasonal_decompose(Tstatedaily['Oregon'], model='additive', period=7)

# SD -39
st_sd = seasonal_decompose(Tstatedaily['South Dakota'], model='additive', period=7)

# ND -40
st_nd = seasonal_decompose(Tstatedaily['North Dakota'], model='additive', period=7)

# RI -41
st_ri = seasonal_decompose(Tstatedaily['Rhode Island'], model='additive', period=7)

# MT -42
st_mt = seasonal_decompose(Tstatedaily['Montana'], model='additive', period=7)

# WV -43
st_wv = seasonal_decompose(Tstatedaily['West Virginia'], model='additive', period=7)

# DE -44
st_de = seasonal_decompose(Tstatedaily['Delaware'], model='additive', period=7)

# AK -45
st_ak = seasonal_decompose(Tstatedaily['Alaska'], model='additive', period=7)

# WY -46
st_wy = seasonal_decompose(Tstatedaily['Wyoming'], model='additive', period=7)

# NH -47
st_nh = seasonal_decompose(Tstatedaily['New Hampshire'], model='additive', period=7)

# ME -48
st_me = seasonal_decompose(Tstatedaily['Maine'], model='additive', period=7)

# HI -49
st_hi = seasonal_decompose(Tstatedaily['Hawaii'], model='additive', period=7)

# VT -50
st_vt = seasonal_decompose(Tstatedaily['Vermont'], model='additive', period=7)





tca =st_ca.trend.fillna(0)
ttx =st_tx.trend.fillna(0)
tfl =st_fl.trend.fillna(0)
til =st_il.trend.fillna(0)
tny =st_ny.trend.fillna(0)

tga =st_ga.trend.fillna(0)
toh =st_oh.trend.fillna(0)
tpa =st_pa.trend.fillna(0)
tmi =st_mi.trend.fillna(0)
twi =st_wi.trend.fillna(0)

ttn =st_tn.trend.fillna(0)
tnc =st_nc.trend.fillna(0)
tin =st_in.trend.fillna(0)
tnj =st_nj.trend.fillna(0)
taz =st_az.trend.fillna(0)


tmn =st_mn.trend.fillna(0)
tmo =st_mo.trend.fillna(0)
tal =st_al.trend.fillna(0)
tco =st_co.trend.fillna(0)
tma =st_ma.trend.fillna(0)

tva =st_va.trend.fillna(0)
tla =st_la.trend.fillna(0)
tia =st_ia.trend.fillna(0)
tsc =st_sc.trend.fillna(0)
tok =st_ok.trend.fillna(0)

tmd =st_md.trend.fillna(0)
tut =st_ut.trend.fillna(0)
tky =st_ky.trend.fillna(0)
twa =st_wa.trend.fillna(0)
tks =st_ks.trend.fillna(0)


tar =st_ar.trend.fillna(0)
tnv =st_nv.trend.fillna(0)
tms =st_ms.trend.fillna(0)
tne =st_ne.trend.fillna(0)
tct =st_ct.trend.fillna(0)

tid =st_id.trend.fillna(0)
tnm =st_nm.trend.fillna(0)
tri =st_ri.trend.fillna(0)
tor =st_or.trend.fillna(0)
tsd =st_sd.trend.fillna(0)


tnd =st_nd.trend.fillna(0)
tmt =st_mt.trend.fillna(0)
twv =st_wv.trend.fillna(0)
tde =st_de.trend.fillna(0)
tak =st_ak.trend.fillna(0)

twy =st_wy.trend.fillna(0)
tnh =st_nh.trend.fillna(0)
thi =st_hi.trend.fillna(0)
tme =st_me.trend.fillna(0)
tvt =st_vt.trend.fillna(0)



trend5=pd.concat([tca, ttx, tfl, til,tny ], axis=1).iloc[0:284, :]
trend10=pd.concat([tga, toh,tpa, tmi,twi ], axis=1).iloc[0:284, :]
trend15=pd.concat([ttn, tnc,tin, tnj,taz ], axis=1).iloc[0:284, :]
trend20=pd.concat([tmn, tmo,tal, tco,tma ], axis=1).iloc[0:284, :]
trend25=pd.concat([tva, tla,tia, tsc,tok ], axis=1).iloc[0:284, :]

trend30=pd.concat([tmd, tut,tky, twa,tks ], axis=1).iloc[0:284, :]
trend35=pd.concat([tar, tnv,tms, tne,tct ], axis=1).iloc[0:284, :]
trend40=pd.concat([tid, tnm,tri, tor,tsd ], axis=1).iloc[0:284, :]
trend45=pd.concat([tnd, tmt,twv, tde,tak ], axis=1).iloc[0:284, :]
trend50=pd.concat([twy, tnh,thi,tme, tvt ], axis=1).iloc[0:284, :]

### Plot trend of COVID-19 IN 50 States

n=len(trend5)
x=list(range(0, n))
Date=trend5.index
Date1=[Date[0], Date[0], Date[50],Date[100], Date[150],Date[200],Date[250], '12/25/20']


fig, ([ax1,ax2], [ax3, ax4],[ax5,ax6], [ax7, ax8],[ax9,ax10]) = plt.subplots(nrows=5, ncols=2, sharex=False)
fig.text(0.5, 0.04, '         Date', ha='center')
fig.text(0.04, 0.5, 'The number of daily cases ', va='center', rotation='vertical')

ax1.plot(x, trend5)
ax1.legend(loc=0, ncol=3, prop={'size': 7},labels= ['CA', 'TX', 'FL','IL', 'NY'])


ax2.plot(x, trend10)
ax2.legend(loc=0, ncol=3, prop={'size': 7},labels= ['GA', 'OH', 'PA','MI', 'WI'])


ax3.plot(x, trend15)
ax3.legend(loc=0, ncol=3, prop={'size': 7},labels= ['TN','NC', 'IN', 'NJ', 'AZ'])

ax4.plot(x, trend20)
ax4.legend(loc=0, ncol=3, prop={'size': 7},labels= ['MN','MO', 'AL', 'CO', 'MA'])


ax5.plot(x, trend25)
ax5.legend(loc=0, ncol=3,prop={'size': 7}, labels= ['VA','LA', 'IA', 'SC', 'OK'])


ax6.plot(x, trend30)
ax6.legend(loc=0, ncol=3, prop={'size': 7},labels= ['MD','UT', 'KY', 'WA', 'KS'])

ax7.plot(x, trend35)
ax7.legend(loc=0, ncol=3,prop={'size': 7}, labels= ['AR','NV', 'MS', 'NE', 'CT'])

ax8.plot(x, trend40)
ax8.legend(loc=0, ncol=3,prop={'size': 7},labels= ['ID','NM', 'RI', 'OR', 'SD'])

ax9.plot(x, trend45)
ax9.legend(loc=0, ncol=3, prop={'size': 7}, labels= ['ND','MT', 'WV', 'DE', 'AK'])

ax10.plot(x, trend50)
ax10.legend(loc=0, ncol=3, prop={'size': 7},labels= ['WY','NH', 'HI', 'ME', 'VT'])

ax1.set_xticklabels(Date1, minor=False)
ax2.set_xticklabels(Date1, minor=False)
ax3.set_xticklabels(Date1, minor=False)
ax4.set_xticklabels(Date1, minor=False)
ax5.set_xticklabels(Date1, minor=False)
ax6.set_xticklabels(Date1, minor=False)
ax7.set_xticklabels(Date1, minor=False)
ax8.set_xticklabels(Date1, minor=False)
ax9.set_xticklabels(Date1, minor=False)
ax10.set_xticklabels(Date1, minor=False)


plt.show()


























