---
layout: post
title: Querying Data and Creating Corresponding Data Visualizations
---

I will create several interesting, interactive data graphics using the NOAA climate data that we’ve explored in the first several weeks of lectures. The main libraries I will use are `sqlite3`, `pandas`, and `plotly`. Additional libraries will be used when needed.

## Create a Database

First, I will create a database with three tables: `temperatures`, `stations`, and `countries`. In order to do this, run the following code:

```python
import pandas as pd
import sqlite3
```

```python
temps = pd.read_csv("temps_stacked.csv")
temps.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>1</td>
      <td>-0.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>2</td>
      <td>2.36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>3</td>
      <td>4.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>4</td>
      <td>7.73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>5</td>
      <td>11.28</td>
    </tr>
  </tbody>
</table>
</div>




```python
countries = pd.read_csv('countries.csv')
# whitespaces in column names are bad for SQL
countries = countries.rename(columns = {"FIPS 10-4":"FIPS_10-4", "ISO 3166":"ISO_3166"})
countries.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FIPS_10-4</th>
      <th>ISO_3166</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>AF</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AX</td>
      <td>-</td>
      <td>Akrotiri</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>AL</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AG</td>
      <td>DZ</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AQ</td>
      <td>AS</td>
      <td>American Samoa</td>
    </tr>
  </tbody>
</table>
</div>




```python
stations = pd.read_csv('station-metadata.csv')
stations.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>STNELEV</th>
      <th>NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AE000041196</td>
      <td>25.3330</td>
      <td>55.5170</td>
      <td>34.0</td>
      <td>SHARJAH_INTER_AIRP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AEM00041184</td>
      <td>25.6170</td>
      <td>55.9330</td>
      <td>31.0</td>
      <td>RAS_AL_KHAIMAH_INTE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AEM00041194</td>
      <td>25.2550</td>
      <td>55.3640</td>
      <td>10.4</td>
      <td>DUBAI_INTL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AEM00041216</td>
      <td>24.4300</td>
      <td>54.4700</td>
      <td>3.0</td>
      <td>ABU_DHABI_BATEEN_AIR</td>
    </tr>
  </tbody>
</table>
</div>



```python
# open a connection to database called temps.db so that you can 'talk' to it using python
conn = sqlite3.connect("temps.db")

# add the tables to the database, replacing old tables if they already exist
temps.to_sql("temperatures", conn, if_exists = "replace", index = False)
countries.to_sql("countries", conn, if_exists = "replace", index = False)
stations.to_sql("stations", conn, if_exists = "replace", index = False)

# always close your connection to the database
conn.close()
```

## Write a Query Function

Next, I will write a Python function called `query_climate_database()` which accepts four arguments:

- `country`, a string giving the name of a country (e.g. ‘South Korea’) for which data should be returned.
- `year_begin` and `year_end`, two integers giving the earliest and latest years for which should be returned.
- `month`, an integer giving the month of the year for which should be returned.

The return value of `query_climate_database()` is a Pandas dataframe of temperature readings for the specified country, in the specified date range, in the specified month of the year. This dataframe will have columns for:

- The station name.
- The latitude of the station.
- The longitude of the station.
- The name of the country in which the station is located.
- The year in which the reading was taken.
- The month in which the reading was taken.
- The average temperature at the specified station during the specified year and month.

```python
def query_climate_database(country, year_begin, year_end, month):
    # connect to the database
    conn = sqlite3.connect("temps.db")
    
    # select columns from joined temperatures and stations tables with year between year_begin and year_end
    cmd = \
    """
    SELECT S.NAME, S.LATITUDE, S.LONGITUDE, SUBSTRING(T.ID, 1, 2) Country, T.Year, T.Month, T.Temp
    FROM temperatures T
    LEFT JOIN stations S ON T.ID = S.ID
    WHERE T.Month == """ + str(month) + " AND T.Year BETWEEN " + str(year_begin) + " AND " + str(year_end)
    
    # read query to create data frame, then close connection
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    
    # merge df with countries data frame on Country (2 letter code) and FIPS_10-4 (same 2 letter code)
    df = df.merge(countries, how = 'inner', left_on = 'Country', right_on = 'FIPS_10-4')
    
    # set Country column to full country name
    df["Country"] = df["Name"]
    
    # drop unnecessary columns
    df = df.drop(["Name", "FIPS_10-4", "ISO_3166"], axis = 1)
    
    # filter data frame to input country
    df = df[df["Country"] == country]
    
    # output resulting data frame
    return(df)
```


```python
query_climate_database(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
```

## Write a Geographic Scatter Function for Yearly Temperature Increases


```python
# import plotly express and a library for linear regression
from plotly import express as px 
from sklearn.linear_model import LinearRegression

# function that computes the slope of the linear regeression model based on temperature over years
def coef(data_group):
    x = data_group[["Year"]]
    y = data_group["Temp"]  
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]

def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs, **kwargs):
    # call query climate database to create data frame based on input
    df = query_climate_database(country, year_begin, year_end, month)

    # create column with number of observations per station
    df["obs"] = df.groupby(["NAME"])["Year"].transform(len)

    # filter data frame for stations with at least min_obs observations
    df = df[df["obs"] >= min_obs]

    # find annual increase in temp using linear regression
    coefs = df.groupby(["NAME", "LATITUDE", "LONGITUDE"]).apply(coef)

    # reset index
    coefs = coefs.reset_index()
    
    # rename column and round slope
    coefs = coefs.rename(columns = {0 : "Estimated Yearly Increase (°C)"})
    coefs["Estimated Yearly Increase (°C)"] = round(coefs["Estimated Yearly Increase (°C)"], 4)
    
    # create dict with names of months
    month_names = {1 : "January", 2 : "February", 3 : "March", 4 : "April", 5 : "May", 6 : "June", 7 : "July", 8 : "August", 9 : "September", 10 : "October", 11 : "November", 12 : "December"} 

    # create geographic scatterplot, with intensity of color reflecting the station's average yearly temperature increase relative to other stations
    fig = px.scatter_mapbox(coefs,
                            lat = "LATITUDE",
                            lon = "LONGITUDE",
                            hover_name = "NAME",
                            color = "Estimated Yearly Increase (°C)",
                            title = "Estimates of yearly increase in temperature in " + month_names[month] + " for Stations in " + country + ", years " + str(year_begin) + "-" + str(year_end),
                            color_continuous_midpoint = 0,
                            **kwargs)
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    return(fig)
```


```python
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style = "carto-positron",
                                   color_continuous_scale = color_map)

fig.show()
```


```python
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig = temperature_coefficient_plot("Brazil", 1970, 2000, 1, 
                                   min_obs = 5,
                                   zoom = 2,
                                   mapbox_style = "carto-positron",
                                   color_continuous_scale = color_map)

fig.show()
```

## Create Two More Interesting Figures


```python
def query_climate_database2(year_begin, year_end):
    conn = sqlite3.connect("temps.db")

    cmd = \
    """
    SELECT SUBSTRING(S.id,1,2) Country, S.name, ROUND(AVG(T.temp), 2) "Average Temperature", T.Year, S.latitude, S.longitude
    FROM temperatures T
    LEFT JOIN stations S ON T.id = S.id
    WHERE T.year BETWEEN """ + str(year_begin) + " AND " + str(year_end) + "\nGROUP BY Country"
    
    df = pd.read_sql_query(cmd, conn)
    conn.close()

    df = df.merge(countries, how = 'inner', left_on = 'Country', right_on = 'FIPS_10-4')
    
    # set Country column to full country name
    df["Country"] = df["Name"]

    df = df.drop(["Name", "FIPS_10-4", "ISO_3166"], axis = 1)
    
    return(df)
```


```python
query_climate_database2(year_begin = 1970, year_end = 2015)
```


```python
from urllib.request import urlopen
import json

def choropleth_plot(year_begin, year_end, **kwargs):
    countries_gj_url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/countries.geojson"

    with urlopen(countries_gj_url) as response:
        countries_gj = json.load(response)

    df = query_climate_database2(year_begin, year_end)

    # using a choropleth    
    fig = px.choropleth(df,
                        geojson = countries_gj,
                        locations = "Country",
                        locationmode = "country names",
                        color = "Average Temperature",
                        #scope = "world",
                        height = 300,
                        title = "Mean temperature across countries, years " + str(year_begin) + "-" + str(year_end),
                        **kwargs)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return(fig)

choropleth_plot(year_begin = 1970, year_end = 2015)
```


```python
import numpy as np

def z_score(x):
    m = np.mean(x)
    s = np.std(x)
    return (x - m)/s

def query_climate_database3(country1, country2, year_begin, year_end):
    conn = sqlite3.connect("temps.db")

    cmd = \
    """
    SELECT SUBSTRING(S.id,1,2) Country, S.name, T.Year, T.Month, T.Temp, S.latitude, S.longitude
    FROM temperatures T
    LEFT JOIN stations S ON T.id = S.id
    WHERE T.year BETWEEN """ + str(year_begin) + " AND " + str(year_end)
    
    df = pd.read_sql_query(cmd, conn)
    conn.close()

    df = df.merge(countries, how = 'inner', left_on = 'Country', right_on = 'FIPS_10-4')
    
    # set Country column to full country name
    df["Country"] = df["Name"]

    df = df.drop(["Name", "FIPS_10-4", "ISO_3166"], axis = 1)
    
    #df['Temperature z-score'] = df.groupby(["Country","NAME"])["Temp"].transform(z_score)
    df_country1 = df[df['Country'] == country1]
    df_country2 = df[df['Country'] == country2]
    df = pd.concat([df_country1, df_country2])
    
    return(df)
```


```python
query_climate_database3("France", "Argentina", 1980, 2020)
```


```python
def temp_line_plot(country1, country2, year_begin, year_end, **kwargs):
    
    df = query_climate_database3(country1, country2, year_begin, year_end)
    df = df.groupby(["Country","Year", "Month"])['Temp'].mean()
    df = df.reset_index()
    
    fig = px.line(df, 
                  x = "Year", 
                  y = "Temp", 
                  color = "Month", 
                  facet_col = "Country",
                  title = f"Temperature changes in {country1} and {country2} during {year_begin}-{year_end} by month",
                  height = 300,
                  **kwargs)

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
```


```python
temp_line_plot("Russia", "Argentina", 1980, 2020)
```
