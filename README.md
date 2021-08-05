## 1. Business Understanding
Ravelry.com is a database-driven website where users can browse and download knitting and crochet patterns, track their progress on a given project, and review the patterns. 

It currently has a "your pattern highlights" recommender system. Compared to the front-and-centre recommendations that Netflix or Amazon make to their users, it's tucked away at the bottom of the patterns search page, displaying only thumbnail images of the recommended pattterns. 

![alt text](https://github.com/clareadunne/PatternRecommender/blob/main/Vizualisations/Rav1.png)

The recommendations generated appear to be based on clicks and/or favourites, rather than a more comprehensive examination of what projects users actually work on and rate positively, having experienced making them. 

The aim of this project is to provide more tailored recommendations for knitting patterns to users of Ravelry.com, based on patterns they have worked on and rated already. 
## 2. Data Understanding
The data has all been obtained through the Ravelry.com API. Since the website does not have an app, it makes all of its content available through APIs, and at present there are 41 apps which make use of some or all of the websites functionality.

The modelling features for a collaborative filtering recommender system are users, items and ratings. the model finds similarities between users based on their ratings of items, and uses these similarities to predict ratings for items for users who have not already rated them.

![alt text](https://github.com/clareadunne/PatternRecommender/blob/main/Vizualisations/Capture1.PNG)

![alt text](https://github.com/clareadunne/PatternRecommender/blob/main/Vizualisations/Capture2.PNG)

![alt text](https://github.com/clareadunne/PatternRecommender/blob/main/Vizualisations/Capture3.PNG)

## 3. Data Preparation
The model was tested on the data with missing ratings removed, and then with missing ratings replaced with pattern averages. Replacing missing values with pattern averages negatively impacted the RMSE for SVD, so proceeded with the missing values dropped. 

Also, one series of the dataframe contained lists which converted to strings after reading from the CSV. These were converted back to lists by slicing and splitting. 

Lastly, the sk surprise package requires a specific version of the data to run its models, generated by passing the dataframe containing only user, item and rating (in that order) to the <code>load_from_df</code> and then <code>train_test_split</code> (for evaluation) or <code>build_full_trainset</code> (for final model) methods. 
## 4. Modelling
Many of the KNN, Matrix Factorization, Slope One, and Co-Clustering modelling methods within the sk surprise package were used, and evaluated based on their RMSE on predicted ratings, using a 25% train test split. 
## 5. Evaluation
The models were tested on the ratings only for projects marked "finished", and then on ratings for projects of all statuses: "finished", "in-progress", "hibernating", and "frogged" $^{1}$. The SVD model achieves the lowest RMSE in either case. While the RMSE is lower where only ratings for completed projects are input, this can lead to imbalanced data, as users are more likely to rate higeher projects which they have finished. 

$^{1}$ "Frogged" refers to a project which was started and then un-knit, or ripped out. The word refers to the sound a frog makes: "ribbit, ribbit", or "Rip it, rip it".
## 6. Generate Predictions
The model is not yet deployed to a user interface. The functions below generate ratings predictions for items that users have not yet interacted with, either by tracking a project, adding a pattern to their queue, or favouriting a pattern. 

Utilising only predicted ratings resulted in almost all users being recommended the same patterns: those that were highly rated in all cases. This did not achieve the type of tailored recommendations anticipated. 

The categories of the user's projects: i.e. sweater, soft-toy, ankle-socks are obtained using the API and only those items matching their most frequently knit categories are returned from the function. 
## Next Steps
1. User interface using streamlit
2. Cold start recommendations for non-users or users with no existing ratings
3. Expand to crocheters, weavers.
4. Layer more content based filtration: attributes (v-neck, seamless, toddler-sized) in addition to categories. 
5. Keep tweaking models to improve RMSE. 

## Repository Directory
```
├── Data
│   └── saved_100000_calls.csv
├── PDFs
│   ├── Capstone Project Technical Notebook - Knitting Pattern Recommender.pdf
│   └── Proposal - Ravelry Project Recommender.pdf
├── Visualizations
│   ├── Capture1.PNG
│   ├── Capture2.PNG
│   ├── Capture3.PNG
│   └── Rav1.png
├── Pattern Recommender - Technical Notebook.ipynb
├── Ravelry API Calls - Users and Projects.ipynb
└── README.md
```
## Reproduction Instructions
This repository contains two jupyter notebooks. The first, Ravelry API Calls - Users and Projects contains the code for making requests of Ravelry's API. It makes 100s of thousands of calls and so unless you need to change it, the data from it is all available in the Data folder, stored in a .csv file. 
The second notebook contains the modelling for the recommendation system, along with the CRISP-DM proces steps. 
The second notebook also makes API calls, although they are user specific and take far less time. You will still need your own keys, see: https://www.ravelry.com/api.

## Link to Presentation
https://drive.google.com/file/d/1HxtcwWsDtODf_WTFMNXM971OFQpBQfvQ/view?usp=sharing
