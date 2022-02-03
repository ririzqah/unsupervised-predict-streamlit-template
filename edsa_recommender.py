"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from re import X
from tkinter import Y
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
import matplotlib
import seaborn as sns 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS



# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
ratings = pd.read_csv("resources/data/movies.csv")
movies = pd.read_csv("resources/data/ratings.csv")
df = movies.merge(ratings)
# Data Cleaning
df['genres'] = df.genres.astype(str)

df['genres'] = df['genres'].map(lambda x: x.lower().split('|'))
df['genres'] = df['genres'].apply(lambda x: " ".join(x))
st.set_page_config('centered')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Main Page", "EDA", "Models","Recommender System"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Main Page":
        
        st.image("resources/imgs/Main.jpg",  width= 800)
        st.markdown("""
        * **Joas Sebola Tsiri:** 
        * **Casper Kruger:** 
        * **Nthabiseng Moloisi:** 
        * **Rizqah Meniers:** 
        * **Tshiamo Nthite:** 
        """)

    if page_selection == "EDA":
        st.image("resources/imgs/b.jpg", width= 800 )
        st.info("""
        What we had to do:
        * Merge the dataset, allowing us to use both datasets.
        * Remove the pipes between genres, to be able to create graphs.
        * And convert the data type of genres to string for string handling.

        What we can see:
        * The title of the movies and their allocated ID's.
        * The genre category that each movie lies within.
        * And the ratings each movie recieved.
        """)


        genre = df['genres'].unique()
        rating = df['rating'].unique()
        select_genre = st.sidebar.selectbox('Select the Genre :', genre)
        select_rating = st.sidebar.selectbox('Select the Rating :', rating)


        
        st.image("resources/imgs/genre.jpg", width= 200)

        if st.button('Show raw data by Genre'):
            st.dataframe(df[df['genres'] == select_genre])


        st.image("resources/imgs/R.jpg", width= 200)

        if st.button('Show raw data by rating'):
            st.dataframe(df[df['rating'] == select_rating])

        
        
        st.title("Rating Distribution and Freqeuncy")

        grouped = pd.DataFrame(df.groupby(['rating'])['title'].count())
        grouped.rename(columns={'title':'rating_count'}, inplace=True)
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(122)
        labels = ['0.5 Star', '1 Stars', '1.5 Stars', '2 Stars', '2.5 Stars', '3 Star', '3.5 Stars', '4 Stars', '4.5 Stars', '5 Stars']
        theme = plt.get_cmap('Blues')
        ax.set_prop_cycle("color", [theme(1. * i / len(labels))
                                 for i in range(len(labels))])
        sns.set(font_scale=1.25)

        pie = ax.pie(grouped['rating_count'],
                 autopct='%1.1f%%',
                 shadow=True,
                 startangle=20,
                 pctdistance=1.115,
                 explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
                 
        
        plt.tight_layout()
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Most used ratings:")
            st.info("""
            * **4** - Stars was the highest with 28.8%
            * **3** - Stars consisted of 20.1%
            * **5** - Stars consisted of 15.1%
            * **3.5** - Stars consisted of 10.5%
            * **4.5** - Stars consisted of 7.7%
            """)

        with col2:
            st.header("Least used ratings:")
            st.info("""
            * **0.5** - Stars was the least used rating with 1.1%
            * **1.5** - Stars consisted of 1.7%
            * **1** - Stars consisted of 3.3%
            * **2.5** - Stars consisted of 4.4%
            * **2** - Stars consisted of 7.3%
            """)

        st.image('resources/imgs/rating.png')

        st.info("""
            * From the rating distribution and freuqency it gives us a better understanding of the data.
            * The distribution shows that 82.2% of all ratings given were above 3.
            * The freqeuncy also shows that only a small amount of movies were given a multiple ratings.
            """)



        st.title('Genres popularity:')   
        st.image("resources/imgs/genres.png", width= 850)
        
        st.header('What the graph shows:')
        st.info("""
        * Throughout the years the amount of times that movies was rated differs.
        * The graph also shows that certain Genres where rated more than others.
        * This could mean that the ratings poeple gave to movies was a factor of the type of movies that came out each year.
        * The top 4 most rated genres are Drama, Comedy, Action and Thriller.
        * Which means people were more intrested in rating movies with these types of Genres than any other type.
        """)

        



    if page_selection == "Models":
        st.title('Recommender systems:')
        
        st.image("resources/imgs/model.jpeg", width= 700 )
       
        
        st.title('Collaborative recommender systems we used:')
        st.image('resources/imgs/SVD.png', width= 500)
        st.info("""
            * Collaborative Filtering is the most common technique used when it comes to building intelligent recommender systems that can learn to give better recommendations as more information about users is collected.
            * It filters information by using the interactions and data collected by the system from other users.
            * From here we decided wich models to use. Our decision came from each models advantages, disadvantages and how they performed against each other.
            """)
        st.header('(SVD) Singular Value Decomposition :')
        st.latex(r'''
        A = UWV^T
        ''')
        

        col3, col4 = st.columns(2)

        with col3:
            st.header('Advantages')
            st.info("""
            * Can be applied to non-square matrices
            * Making the observation have the largest variance
            * SVD can be utilized to sully forth pseudo-inverses.
            """)
        with col4:
            st.header('Disadvantages')
            st.info("""
            * Computing is very slow
            * Computationally expensive
            * Requires care when dealing with missing data
            """)

        st.header('(KNN) K-Nearest Neighbor :')

        st.latex(r'''
        r_{ij} = \sum_k Similaries(u_i,u_k)r_{kj} / {number-of-ratings}
        ''')

        col5, col6 = st.columns(2)
        with col5:
            st.header('Advantages')
            st.info("""
            * It is simple to implement.
            * It is robust to the noisy training data.
            * It can be more effective if the training data is large.
            """)
        with col6:
            st.header('Disadvantages')
            st.info("""
            * Always needs to determine the value of K which may be complex some time.
            * The computation cost is high because of calculating the distance between the data points for all the training samples.
            """)

        st.header("Content recommender system:")
        st.image('resources/imgs/OIP.jpg', width= 400)

        st.info("""
            * A Content-Based Recommender works by the data that we take from the user, either explicitly (rating) or implicitly (clicking on a link).
            * By the data we create a user profile, which is then used to suggest to the user, as the user provides more input or take more actions on the recommendation, the engine becomes more accurate.
            """)

        st.subheader('2 Methods for content based filtering :')
        st.info("""
        Method 1 : Vector space method
        * Let us suppose you watch a crime thriller Movie, you review it on the internet. Also, you review one more Movie of the comedy genre with it and review the crime thriller Movie as good and the comedy one as bad. 
        Now, a rating system is made according to the information provided by you. In the rating system from 0 to 5, crime thriller genres are ranked as 5, and other movies lie from 5 to 0 and the comedy ones lie at the lowest.
        With this information, the next book recommendation you will get will be of crime thriller genres most probably as they are the highest rated genres for you.
        For this ranking system, a user vector is created which ranks the information provided by you. After this, an item vector is created where movies are ranked according to their genres on it.
        With the vector, every movie name is assigned a certain value by multiplying and getting the dot product of the user and item vector, and the value is then used for recommendation.
        Like this, the dot products of all the available movies searched by you are ranked and according to it the top 5 or top 10 movies are assigned.
        """)

        st.info("""
        Method 2 : 
        * The second method is the classification method. In it, we can create a decision tree and find out if the user wants to watch a movie or not.
        For example, a movie is considered, let it be Spider-man.
        Based on the user data, we see that the genre is not a crime thriller, nor is it the type of movie you ever reviewed. With these classifications, we conclude that this movie shouldn’t be recommended to you.
        """)

        col7, col8 = st.columns(2)
        with col7:
            st.header('Advantages')
            st.info("""
            * Because the recommendations are tailored to a person, the model does not require any information about other users. This makes scaling of a big number of people more simple.
            * The model can recognize a user's individual preferences and make recommendations for niche things that only a few other users are interested in.
            * New items may be suggested before being rated by a large number of users, as opposed to collective filtering.
            """)
        with col8:
            st.header('Disadvantages')
            st.info("""
            * This methodology necessitates a great deal of domain knowledge because the feature representation of the items is hand-engineered to some extent. As a result, the model can only be as good as the characteristics that were hand-engineered.
            * The model can only give suggestions based on the user's current interests. To put it another way, the model's potential to build on the users' existing interests is limited.
            * Since it must align the features of a user's profile with available products, content-based filtering offers only a small amount of novelty. 
            """)



            



    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
