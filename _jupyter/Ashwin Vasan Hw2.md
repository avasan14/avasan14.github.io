```python
import pandas as pd
import numpy as np
```


```python
movies = pd.read_csv("movies.csv")
movies
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
      <th>actor</th>
      <th>movie_or_TV_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jasper Linnewedel</td>
      <td>Inglourious Basterds</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Salvadore Brandt</td>
      <td>Heartbeat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Salvadore Brandt</td>
      <td>Sushi in Suhl</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Salvadore Brandt</td>
      <td>Europas letzter Sommer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Salvadore Brandt</td>
      <td>Das blaue Licht</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10191</th>
      <td>Michael Fassbender</td>
      <td>60 Minutes</td>
    </tr>
    <tr>
      <th>10192</th>
      <td>Michael Fassbender</td>
      <td>Celebrity Page</td>
    </tr>
    <tr>
      <th>10193</th>
      <td>Michael Fassbender</td>
      <td>Made in Hollywood</td>
    </tr>
    <tr>
      <th>10194</th>
      <td>Michael Fassbender</td>
      <td>Chelsea Lately</td>
    </tr>
    <tr>
      <th>10195</th>
      <td>Michael Fassbender</td>
      <td>1st AACTA Awards</td>
    </tr>
  </tbody>
</table>
<p>10196 rows × 2 columns</p>
</div>




```python
shared = movies.groupby(["movie_or_TV_name"]).aggregate(len)
df = shared.sort_values(by = ["actor"], ascending = False).reset_index().head(10)
df = df.rename({"movie_or_TV_name": "movie","actor": "number of shared actors"}, axis = 1)
df
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
      <th>movie</th>
      <th>number of shared actors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Inglourious Basterds</td>
      <td>99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tatort</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Oscars</td>
      <td>21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entertainment Tonight</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Police Call 110</td>
      <td>16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Grindhouse</td>
      <td>16</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Django Unchained</td>
      <td>16</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Saturday Night Live</td>
      <td>15</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Extra with Billy Bush</td>
      <td>14</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Today</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
```


```python
sns.barplot(x = "number of shared actors",
            y = "movie",
            data = df).set(title = "Top 10 Movies/TV Shows with most shared actors with Inglourious Basterds")
```




    [Text(0.5, 1.0, 'Top 10 Movies/TV Shows with most shared actors with Inglourious Basterds')]




    
![png](Ashwin%20Vasan%20Hw2_files/Ashwin%20Vasan%20Hw2_4_1.png)
    

