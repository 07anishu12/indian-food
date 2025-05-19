import numpy as np
import pandas as pd 
import networkx as nx

import geopandas as gpd 
import matplotlib.pyplot as plt
import seaborn as sns
import random
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots
init_notebook_mode(connected=True)
import cufflinks as cf

cf.go_offline()
from wordcloud import WordCloud , ImageColorGenerator
from PIL import Image

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/indian_food.csv") 
df=df.replace(-1,np.nan)
df=df.replace('-1',np.nan)


df.shape
df.isnull().sum()
df.describe()

#Diet
pie_df = df.diet.value_counts().reset_index()
pie_df.columns = ['diet','count']
fig = px.pie(pie_df, values='count', names='diet', title='Proportion of Vegetarian and Non-Vegetarian dishes',
             color_discrete_sequence=['green', 'red'])
fig.show()

#Region
reg_df = df.region.value_counts().reset_index()
reg_df.columns = ['region','count']
reg_df=reg_df.sample(frac=1)

fig = px.bar(reg_df, x='region', y='count', title='Number of dishes by region',
             color_discrete_sequence=['blue'])
fig.show()

#Course
course_df=df.course.value_counts().reset_index()
course_df.columns = ['course','count']
course_df=course_df.sample(frac=1)
fig = px.bar(course_df, x='course', y='count', title='Number of dishes by course',color_discrete_sequence=['red'])
fig.show()

#Flavor
pie_df = df.flavor_profile.value_counts().reset_index()
pie_df.columns = ['flavor','count']
fig=px.pie(pie_df,values='count',names='flavor',title='Proportion of flavors in Indian cuisine',color_discrete_sequence=['green','blue','yellow','purple','orange','pink','brown','gray','black','white'])
fig.show()

#Dessert
dessert_df=df[df['course']=='dessert'].reset_index(drop=True)
ingredients = []
for i in range(0,len(dessert_df)):
    text=dessert_df['ingredients'][i].split(',')
    text=",".join(text)
    ingredients.append(text)
    text="".join(ingredients)

wordcloud = WordCloud(width=800,height=400,colormap = 'seismic',background_color='white',min_font_size = 10).generate(text)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

#North Indian

north_df = df[df['region']=='North'].reset_index()

ingredients = []
for i in range(0,len(north_df)):
    text = north_df['ingredients'][i].split(',')
    text = ','.join(text)
    ingredients.append(text)
    text = ' '.join(ingredients)

wordcloud = WordCloud(width = 400, height = 400, colormap = 'winter',
                      background_color ='white', 
                min_font_size = 10).generate(text)                  
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.show()

#dishes
wordCloud = WordCloud(
    background_color='White',colormap = 'seismic',
    max_font_size = 50).generate(' '.join(df['name']))
plt.figure(figsize=(15,7))
plt.axis('off')
plt.imshow(wordCloud)
plt.show()


