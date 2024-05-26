import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/earthquakes-2023-global/earthquakes_2023_global.csv")

df.info()

df.columns

#checking missing values
df.isnull().sum()

#check duplicate values
df.duplicated().sum()

duplicates = df[df.duplicated()]
print("Number of duplicate rows:", len(duplicates))

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Verify that duplicates have been removed
print("Number of duplicate rows after removal:", len(df[df.duplicated()]))

# fill the null value
df['nst'].fillna(df['nst'].mean(), inplace=True)
df['gap'].fillna(df['gap'].mean(), inplace=True)
df['dmin'].fillna(df['dmin'].mean(), inplace=True)
df['horizontalError'].fillna(df['horizontalError'].mean(), inplace=True)
df['magError'].fillna(df['magError'].mean(), inplace=True)
df['magNst'].fillna(df['magNst'].mean(), inplace=True)

# For 'place' column, fill missing values with 'Unknown'
df['place'].fillna('Unknown', inplace=True)

# Display the remaining missing values after handling
print(df.isnull().sum())

# Spatial Analysis
plt.figure(figsize=(40, 25))
sns.scatterplot(x='longitude', y='latitude', hue='mag', size='mag', data=df)
plt.title('Earthquake Distribution by Latitude and Longitude')
plt.show()

# Magnitude Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['mag'], bins=30, kde=True)
plt.title('Distribution of Earthquake Magnitudes')
plt.show()

# Status Analysis
plt.figure(figsize=(8, 6))
sns.countplot(x='status', data=df)
plt.title('Distribution of Earthquake Status')
plt.show()

!apt-get install -y libgeos-dev
!pip install basemap

from mpl_toolkits.basemap import Basemap

plt.rcParams["figure.figsize"]=13,13
m=Basemap()
m.drawcoastlines()
 
plt.show();

lats = df['latitude'].values
lons = df['longitude'].values
magnitudes = df['mag'].values

# Scatter plot earthquake locations on the map
x, y = m(lons, lats)
m.scatter(x, y, c=magnitudes, s=10, cmap='Reds', alpha=0.7, edgecolors='k', linewidth=0.5)
plt.colorbar(label='Magnitude')
plt.title('Spatial Distribution of Earthquakes')
plt.show()

plt.rcParams["figure.figsize"] = 13, 13
m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
m.drawcoastlines()
m.drawcountries()
# Scatter plot earthquake locations on the map
x, y = m(lons, lats)
m.scatter(x, y, c=magnitudes, s=10, cmap='Reds', alpha=0.7, edgecolors='k', linewidth=0.5)
plt.colorbar(label='Magnitude')
plt.title('Spatial Distribution of Earthquakes')
plt.show()

# earthquake depth analysis
plt.figure(figsize=(10, 6))
plt.hist(df['depth'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Earthquake Depths')
plt.xlabel('Depth (km)')
plt.ylabel('Frequency')
plt.show()

# magnitude vs depth analysis
fig = plt.figure(figsize=(30, 20))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['longitude'], df['latitude'], df['depth'], c=df['mag'], cmap='viridis', s=df['mag']*10, alpha=0.7)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Depth (km)')
ax.set_title('Magnitude vs Depth with Geographic Information')
cbar = plt.colorbar(scatter)
cbar.set_label('Magnitude')

plt.show()

plt.figure(figsize=(25, 20))
sns.jointplot(x='mag', y='depth', data=df, kind='scatter', marginal_kws=dict(bins=25, fill=False))
plt.suptitle('Magnitude vs Depth Relationship', y=1.02)
plt.show()

** end **




