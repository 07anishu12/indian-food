import numpy as np
import pandas as pd 
import networkx as nx
import os
import sys

def install_missing_packages():
    """Install required packages for image export if they're missing"""
    try:
        import plotly.io as pio
        # Check if kaleido is installed (required for static image export in plotly)
        try:
            pio.kaleido.scope
            print("Kaleido is already installed.")
        except:
            print("Installing kaleido for Plotly image export...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaleido"])
            print("Kaleido installation completed.")
    except ImportError:
        print("Installing required dependencies...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "kaleido"])
        print("Dependencies installation completed.")

# Call the function to install required packages
install_missing_packages()

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

save_path = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "indian food", "figures")
os.makedirs(save_path, exist_ok=True)
print(f"Saving images to: {save_path}")

def save_figure(fig, filepath):
    """Helper function to save figure with error handling"""
    try:
        fig.write_image(filepath)
        print(f"Successfully saved: {filepath}")
    except Exception as e:
        print(f"Error saving {filepath}: {str(e)}")
        # Try alternative method
        try:
            plt.savefig(filepath)
            print(f"Saved using matplotlib instead: {filepath}")
        except Exception as e2:
            print(f"Alternative method also failed: {str(e2)}")

# Diet Pie Chart
pie_df = df.diet.value_counts().reset_index()
pie_df.columns = ['diet', 'count']
fig = px.pie(pie_df, values='count', names='diet', title='Proportion of Vegetarian and Non-Vegetarian dishes',
             color_discrete_sequence=['green', 'red'])
save_figure(fig, os.path.join(save_path, "diet_pie_chart.png"))

# Region Bar Chart
reg_df = df.region.value_counts().reset_index()
reg_df.columns = ['region', 'count']
reg_df = reg_df.sample(frac=1)
fig = px.bar(reg_df, x='region', y='count', title='Number of dishes by region',
             color_discrete_sequence=['blue'])
save_figure(fig, os.path.join(save_path, "region_bar_chart.png"))

# Course Bar Chart
course_df = df.course.value_counts().reset_index()
course_df.columns = ['course', 'count']
course_df = course_df.sample(frac=1)
fig = px.bar(course_df, x='course', y='count', title='Number of dishes by course',
             color_discrete_sequence=['red'])
save_figure(fig, os.path.join(save_path, "course_bar_chart.png"))

# Flavor Pie Chart
pie_df = df.flavor_profile.value_counts().reset_index()
pie_df.columns = ['flavor', 'count']
fig = px.pie(pie_df, values='count', names='flavor', title='Proportion of flavors in Indian cuisine',
             color_discrete_sequence=['green','blue','yellow','purple','orange','pink','brown','gray','black','white'])
save_figure(fig, os.path.join(save_path, "flavor_pie_chart.png"))

# Dessert WordCloud
dessert_df = df[df['course'] == 'dessert'].reset_index(drop=True)
ingredients = []
for i in range(len(dessert_df)):
    text = dessert_df['ingredients'][i].split(',')
    ingredients.append(",".join(text))
text = " ".join(ingredients)

wordcloud = WordCloud(width=800, height=400, colormap='seismic', background_color='white', min_font_size=10).generate(text)
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig(os.path.join(save_path, "dessert_wordcloud_1.png"))
print(f"Saved dessert wordcloud to: {os.path.join(save_path, 'dessert_wordcloud_2.png')}")
plt.show()

# North Indian WordCloud
north_df = df[df['region'] == 'North'].reset_index()
ingredients = []
for i in range(len(north_df)):
    text = north_df['ingredients'][i].split(',')
    ingredients.append(",".join(text))
text = " ".join(ingredients)

wordcloud = WordCloud(width=400, height=400, colormap='winter', background_color='white', min_font_size=10).generate(text)
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig(os.path.join(save_path, "north_indian_wordcloud_1.png"))
print(f"Saved north indian wordcloud to: {os.path.join(save_path, 'north_indian_wordcloud_2.png')}")
plt.show()

# Dish Name WordCloud
wordCloud = WordCloud(background_color='White', colormap='seismic', max_font_size=50).generate(' '.join(df['name']))
plt.figure(figsize=(15, 7))
plt.axis('off')
plt.imshow(wordCloud)
plt.savefig(os.path.join(save_path, "dish_names_wordcloud_1.png"))
print(f"Saved dish names wordcloud to: {os.path.join(save_path, 'dish_names_wordcloud_2.png')}")
plt.show()


def state_infograph(state_name, title, save_path=None):
    state_df = df[df['state'] == state_name]

    if state_df.empty:
        print(f"No data available for state: {state_name}")
        return

    total_dishes = len(state_df)

    course_df = state_df['course'].value_counts().reset_index()
    course_df.columns = ['course', 'count']

    diet_df = state_df['diet'].value_counts().reset_index()
    diet_df.columns = ['diet', 'count']

    prep_time_df = state_df['prep_time'].value_counts().reset_index()
    prep_time_df.columns = ['prep_time', 'count']

    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Total Dishes', 'Dishes by Courses', 'Dishes by Preparation Time', '', '', ''),
        specs=[
            [{'type': 'indicator'}, {'type': 'bar', 'rowspan': 2}, {'type': 'bar', 'rowspan': 2}],
            [{'type': 'pie'}, None, None]
        ]
    )

    # Indicator for total dishes
    fig.add_trace(go.Indicator(
        mode='number',
        value=total_dishes,
        number={'font': {'color': '#270082', 'size': 50}},
    ), row=1, col=1)

    # Bar chart for course distribution
    fig.add_trace(go.Bar(
        x=course_df['course'],
        y=course_df['count'],
        marker_color='blue',
        text=course_df['count'],
        name='Courses',
        textposition='auto'
    ), row=1, col=2)

    # Pie chart for diet
    fig.add_trace(go.Pie(
        labels=diet_df['diet'],
        values=diet_df['count'],
        textinfo='percent+label',
        marker=dict(colors=['#00bd0d', '#fc0303']),
        name='Diet'
    ), row=2, col=1)

    # Bar chart for preparation time
    fig.add_trace(go.Bar(
        x=prep_time_df['prep_time'],
        y=prep_time_df['count'],
        marker_color='#fc0335',
        text=prep_time_df['count'],
        name='Preparation Time',
        textposition='auto'
    ), row=1, col=3)

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        template='plotly',
        height=600,
        width=1000,
        showlegend=False
    )

    fig.show()

    # Auto save image if no path given
    if save_path is None:
        folder = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "indian food", "figures")
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, f"{state_name.lower()}_infograph.png")
        print(f"Using default path: {save_path}")

    save_figure(fig, save_path)
    return fig


unique_states = df['state'].dropna().unique()

# Loop through each state and create infographic
for state in unique_states:
    title = f"{state} Food Infograph"
    print(f"Creating infographic for: {state}")
    state_infograph(state, title)