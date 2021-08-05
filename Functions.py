import surprise
from surprise.prediction_algorithms import *
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import requests
import json
import math
import random
from surprise import Reader, Dataset
from surprise.dataset import DatasetAutoFolds
from surprise import SVD, accuracy

with open('.secrets/creds.json') as f:
    creds = json.load(f)


input_df = pd.read_csv('Data/saved_100000_calls.csv')
df_drop_nans = input_df[['user', 'pattern_id', 'rating']].dropna(subset = ['rating'])
users_list = list(df_drop_nans['user'].unique())

df_pattern_ids_and_categories = input_df[['pattern_id', 'categories']]
df_pattern_ids_and_categories = df_pattern_ids_and_categories.drop_duplicates(subset=['pattern_id'])
df_pattern_ids_and_categories['cat_list'] = ''
for pattern in list(df_pattern_ids_and_categories.index):
    df_pattern_ids_and_categories['cat_list'][pattern] = [category[1:-1] for category in df_pattern_ids_and_categories['categories'][pattern][1:-1].split(', ')]

best_model = SVD(n_factors = 15, n_epochs = 30, lr_all = 0.003, reg_all = 0.2)


def get_user_projects(user):
    
    # a data frame of projects tracked by a given user
    
    try:
   
        url ='https://api.ravelry.com/projects/' + user + '/list.json?sort=completed_'
        response = requests.get(url, auth=(creds['id'], creds['key']))
        projects = []
        try:
            for project in response.json()['projects']:
                if project['craft_name'] == 'Knitting': 
                    if project['pattern_id'] != None:
                        pattern_url ='https://api.ravelry.com/patterns.json?ids=' + str(int(project['pattern_id']))
                        pattern_response = requests.get(pattern_url, auth=(creds['id'], creds['key']))
                        project_tuple = (user, project['completed'], project['rating'], project['status_name'], 
                                         project['pattern_id'],
                                         pattern_response.json()['patterns'][str(int(project['pattern_id']))]['rating_average'],
                                         pattern_response.json()['patterns'][str(int(project['pattern_id']))]['rating_count'],
                                         [attribute['permalink'] for attribute in pattern_response.json()['patterns'][str(int(project['pattern_id']))]['pattern_attributes']],
                                         [category['permalink'] for category in pattern_response.json()['patterns'][str(int(project['pattern_id']))]['pattern_categories']])
                        projects.append(project_tuple)
            df = pd.DataFrame(projects, columns = ['user', 'completed', 'rating', 'status', 'pattern_id', 'average_rating', 'rating_count', 'attributes', 'categories'])

        except ValueError:
            pass
    
    except ValueError:
        df = pd.DataFrame[{}]
    return df

def get_user_projects_not_finished(user):
    
    # a list of ongoing, frogged, or hibernated projects for a given user - only if user is already in modelling data

    users_projects_not_completed = requests.get('https://api.ravelry.com/projects/' + user + '/list.json', 
                                                auth=(creds['id'], creds['key']))

    df = pd.DataFrame(users_projects_not_completed.json()['projects'])
    users_projects_not_completed = list(set(df[df['status_name'] != 'Finished']['pattern_id'].dropna()))
    
    return users_projects_not_completed

def get_user_queue(user):
    # a list of projects in a user's queue

    users_queue = requests.get('https://api.ravelry.com/people/' + user + '/queue/list.json?page_size=100', 
                                                auth=(creds['id'], creds['key']))
    
    users_queue = list(set(pd.DataFrame(users_queue.json()['queued_projects'])['pattern_id'].dropna()))

    return users_queue

def get_user_favorites(user):
    # a list of a patterns favourited by a given user
    users_favourites = requests.get('https://api.ravelry.com/people/' + user + '/favorites/list.json?page_size=100', 
                                    auth=(creds['id'], creds['key']))
    
    df = pd.DataFrame(users_favourites.json()['favorites'])
    users_favourites = list(pd.DataFrame(list(df[df['type'] == 'pattern']['favorited']))['id'])
    
    return users_favourites

def top_rated(user):
    # returns patterns predicted to earn a rating of 3 or more for a given user 
    # if the user is already in the data, no need to refit model
    
    if user in users_list:
        
        # make a list of patterns in modelling data, remove any the user has previously interacted with, generate 
        # predicted ratings for those patterns, output any greater than 3 to df
        
        patterns_list = list(input_df['pattern_id'].unique())
    
        predictions = []
        
        users_patterns = list(input_df[input_df['user'] == user]['pattern_id'])
        users_favourites = get_user_favorites(user)
        users_queue = get_user_queue(user)
        users_projects_not_completed = get_user_projects_not_finished(user)
        
        previously_interacted = users_patterns + users_favourites + users_queue + users_projects_not_completed
        
        remaining_patterns = [x for x in patterns_list if x not in previously_interacted]

    
        for pattern in remaining_patterns:
            x = best_model.predict(user, pattern)
            predictions.append(x)
        
        predictions_df = pd.DataFrame({"user": [prediction.uid for prediction in predictions],
                                       "item": [prediction.iid for prediction in predictions],
                                       "estimated" :[prediction.est for prediction in predictions]})
    
        predictions_df = predictions_df[predictions_df['estimated'] > 3]
        predictions_df = predictions_df.sort_values('estimated', ascending = False)
    
        return predictions_df
    
    elif user not in users_list:
        
        # get user data to match modelling data, transform to match, and refit model with that user included. 
        
        try: 
            
            new_user_ratings = get_user_projects(user)
            new_user_input_df = input_df.append(new_user_ratings).reset_index().drop(columns = 'index')
        
            df_drop_nans_new_user = new_user_input_df[['user', 'pattern_id', 'rating']].dropna(subset = ['rating'])
        
            reader = Reader()
            data_drop_new_user = Dataset.load_from_df(df_drop_nans_new_user, reader)
            trainset_new_user = DatasetAutoFolds.build_full_trainset(data_drop_new_user)
        
            best_model.fit(trainset_new_user)
    
            # make a list of patterns in modelling data, remove any the user has previously interacted with, generate 
            # predicted ratings for those patterns, output any greater than 3 to df
    
            patterns_list = list(new_user_input_df['pattern_id'].unique())
            predictions = []
        
            users_patterns = list(new_user_input_df[new_user_input_df['user'] == user]['pattern_id'])
            users_favourites = get_user_favorites(user)
            users_queue = get_user_queue(user)
            users_projects_not_completed = list(set(new_user_ratings[new_user_ratings['status'] != 'Finished']['pattern_id'].dropna()))
        
            previously_interacted = users_patterns + users_favourites + users_queue + users_projects_not_completed
        
            remaining_patterns = [pattern for pattern in patterns_list if pattern not in previously_interacted]
    
            for pattern in remaining_patterns:
                x = best_model.predict(user, pattern)
                predictions.append(x)
                
            predictions_df = pd.DataFrame({"user": [prediction.uid for prediction in predictions],
                                           "item": [prediction.iid for prediction in predictions],
                                           "estimated": [prediction.est for prediction in predictions]})
    
            predictions_df = predictions_df[predictions_df['estimated'] > 3]
            predictions_df = predictions_df.sort_values('estimated', ascending = False)
            
            return predictions_df
        
        except:
            
            patterns_list = list(input_df['pattern_id'].unique())
            df_non_user = []
            
            random_sample_patterns = random.sample(patterns_list, 6)

            for x in random_sample_patterns:
                pattern_url ='https://api.ravelry.com/patterns.json?ids=' + str(x)
                pattern_response = requests.get(pattern_url, auth=(creds['id'], creds['key']))
                photo_urls = [photo['medium_url'] for photo in pattern_response.json()['patterns'][str(x)]['photos'][0:3]]
                st.image(photo_urls, width = 200)
        
                pattern_name = pattern_response.json()['patterns'][str(x)]['name']

                label = 'How do you rate ' + pattern_name + '?'
                rating = st.radio(label = label, options=[0,1,2,3,4])
                df_non_user.append({'user': 'new_user', 'pattern_id': x, 'rating': rating})
        
            df_non_user = pd.DataFrame(df_non_user)
            df_drop_nans.append(df_non_user).reset_index().drop(columns = 'index')
            
            reader = Reader()
            data_drop_non_user = Dataset.load_from_df(df_drop_nans, reader)
            trainset_non_user = DatasetAutoFolds.build_full_trainset(data_drop_non_user)
        
            best_model.fit(trainset_non_user)
            # generate predicted ratings for those patterns, output any greater than 3 to df
        
            previously_interacted = random_sample_patterns
            remaining_patterns = [pattern for pattern in patterns_list if pattern not in previously_interacted]
            
            predictions = []
            
            for pattern in remaining_patterns:
                x = best_model.predict('new_user', pattern)
                predictions.append(x)
                
            predictions_df = pd.DataFrame({"user": [prediction.uid for prediction in predictions],
                                           "item": [prediction.iid for prediction in predictions],
                                           "estimated": [prediction.est for prediction in predictions]})
    
            predictions_df = predictions_df[predictions_df['estimated'] > 3]
            predictions_df = predictions_df.sort_values('estimated', ascending = False)
            
            return predictions_df

def user_fave_categories(user):
    # return a list of the users most frequently knitted types of patterns 
    # (i.e. scarves, toys, cardigans...)
    try:
    
        user_projects = get_user_projects(user)

        user_projects['cat'] = ''
        for project in range(0,len(user_projects)):
            user_projects['cat'][project] = user_projects['categories'][project].sort()
            for category in range(0,len(user_projects['categories'][project])):
                user_projects['cat'][project] = user_projects['categories'][project][category]

        df_count_categories = user_projects.groupby('cat').count().sort_values('user', ascending = False)
        df_count_categories = df_count_categories.reset_index()[['cat', 'user']]
        if len(df_count_categories) <= 5:
            favorite_categories = list(df_count_categories['cat'])
        elif len(df_count_categories) <=20:
            favorite_categories = list(df_count_categories.head(5)['cat'])
        elif len(df_count_categories) > 20:
            favorite_categories = list(df_count_categories.head(math.ceil(len(df_count_categories)/5))['cat'])

    except:
        favorite_categories = []
    
    return favorite_categories

def get_recommendations(user):
    
    if len(user_fave_categories(user)) != 0: 

        fave_categories = user_fave_categories(user)
        # merge df of user recommendations with input df containing item categories
        
        recs = top_rated(user)
        recs['pattern_id'] = recs['item']
        result = pd.merge(df_pattern_ids_and_categories, recs, how="inner", on=["pattern_id"])
        
        
        # drop any recommendations not corresponding to users top categories
        
        result['favourites_list'] = ''
        for rec in list(result.index):
            if len(list(set(result['cat_list'][rec]).intersection(set(fave_categories)))) != 0:
                result['favourites_list'][rec] = 1
            else: 
                result['favourites_list'][rec] = 0
                
        result = result[result['favourites_list'] != 0]
        result = result.sort_values('estimated', ascending = False).head(15)
        
        recommendations = []
        
        # get pattern name and generate url
    
        for pattern in list(result['item']):
        
            pattern_url ='https://api.ravelry.com/patterns.json?ids=' + str(pattern)
            pattern_response = requests.get(pattern_url, auth=(creds['id'], creds['key']))
            
            st.write(pattern_response.json()['patterns'][str(pattern)]['name'])
            st.image([photo['square_url'] for photo in pattern_response.json()['patterns'][str(pattern)]['photos'][0:3]], width = 200)
            st.write("          ")

            recommendations.append('ravelry.com/patterns/library/' + str(pattern_response.json()['patterns'][str(pattern)]['permalink']))
    
        return recommendations
    
    elif len(user_fave_categories(user)) == 0: 
      
        recs = top_rated(user)
        recs['pattern_id'] = recs['item']
        result = pd.merge(df_pattern_ids_and_categories, recs, how="inner", on=["pattern_id"])
                
        # drop any recommendations not corresponding to users top categories
        
        result = result.sort_values('estimated', ascending = False).head(15)
        
        recommendations = []
        
        # get pattern name and generate url
    
        for pattern in list(result['item']):
        
            pattern_url ='https://api.ravelry.com/patterns.json?ids=' + str(pattern)
            pattern_response = requests.get(pattern_url, auth=(creds['id'], creds['key']))
            recommendations.append('ravelry.com/patterns/library/' + str(pattern_response.json()['patterns'][str(pattern)]['permalink']))
    
        return recommendations
        
        