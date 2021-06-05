import numpy as np
import pandas as pd
# This library is used to ignore warnings
import warnings

warnings.filterwarnings('ignore')

# Now lets get the dataset
columns_names = ["user_id", "item_id", "rating", "timestamp"]  #Proj-MovieRecommendationSystem\ml-100k\u.data
df = pd.read_csv(r"Proj-MovieRecommendationSystem\ml-100k\u.data", sep='\t', names = columns_names)
# now one might think that how can we use read_csv if we are not using a csv file. In this case we are using tsv file(tab separated 
# file) that's why we use sep='\t' to inform about a tsv file being used

print(df.head())
print(df.shape) 

# This gives number of unique users
print(df['user_id'].nunique())
# This gives number of unique movies
print(df['item_id'].nunique())

# We used sep='\|' as this is a '|' seperated file
movie_titles = pd.read_csv(r'Proj-MovieRecommendationSystem\ml-100k\u.item', sep='\|', header=None)
print(movie_titles.shape)

# We need only the 1st 2 columns
movie_titles = movie_titles[[0, 1]]
# Naming the columns
movie_titles.columns = ['item_id', 'title']
print(movie_titles.head())

# We are merging df and movie_titles into one
# on='item_id' specifies on what basis are we trying to merge. So titles get added corresponding to item_id 
df = pd.merge(df, movie_titles, on='item_id')
print(df)

# Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# We are going to calculate the mean rating of every film as different users give different rating to a film

# groupby is used to group data.
# mean would calculate the mean of all other data
print(df.groupby('title').mean()['rating'].sort_values(ascending = False))

# number of times a movie has been watched in descending order
print(df.groupby('title').count()['rating'].sort_values(ascending=False))

# lets create a new dataframe
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['No. of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])
print(ratings)

print(ratings.sort_values(by='rating', ascending=False))
# But we can't rely on average ratings if the movie has been watched only by 2-3 people

# plt.figure(figsize=(10, 6))
# plt.hist(ratings['No. of ratings'], bins=70)
# plt.xlabel('No. of times a movie has been rated')
# plt.ylabel('Frequency of rating')
# # plt.show()

# plt.hist(ratings['rating'], bins = 70)
# plt.xlabel('Average Rating')
# plt.ylabel('Frequency of rating')
# plt.show()

# sns.jointplot(x='rating', y='No. of ratings', data=ratings, alpha=0.5)
# plt.show()

# print(df.head())

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
print(moviemat)

starwars_user_ratings = moviemat['Star Wars (1977)']
print(starwars_user_ratings)

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
print(corr_starwars)

print(corr_starwars.sort_values('Correlation', ascending=False).head(10))
# But one should keep in mind that we don't want movies with a correlation of 1. Thus these movies must be filtered out
# We must also consider only those movies that have atleast 100 ratings

corr_starwars = corr_starwars.join(ratings['No. of ratings'])
print(corr_starwars.head())
print(corr_starwars[corr_starwars['No. of ratings']>100].sort_values('Correlation', ascending=False))

# Predict Function
def predict_movies(movie_name):
    movie_user_ratings = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)

    corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)

    corr_movie = corr_movie.join(ratings['No. of ratings'])
    predictions = corr_movie[corr_movie['No. of ratings']>100].sort_values('Correlation', ascending = False)

    return predictions

predictions = predict_movies('12 Angry Men (1957)')
print(predictions.head())