# movie_recommender
 
This project is based on a give dataset from an online movie streaming company which includes movie information, user IDs and user ratings. 

Source of data for this project is [Kaggle Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Aim of this project is to 
1) recommend top 5 popular movies for given genres, and 

2) to recommend top 3 movies for 3 given user IDs. 

In tackling the first problem, a simple Weighted Rating Rank method is proposed. For the second problem, a collaborative filtering technique is chosen, in particular Probabilistic Matrix Factorization algorithm. Both methods are implemented in Python. 

## Dataset Introduction
The given dataset for this project consists of 7 files in csv format.   
In `movies_metadata`, the file contains basic information of `45,418` movies including `movie id`, `imdb ID`, `movie title`, `genre`, `budget`, `popularity score`, `average votes`, etc.   
In `keywords`, the file allocates keywords to each movie id.   
File `links` references corresponding movie ID, IMDB ID and TMBD ID for all movies, whereas file `links_small` extracts id information of 9,000 movies.   
The `rating` file shows record of all movie ratings in range of 1-5 by each user ID at different time stamps.   
In correspondence to the `links_small` file, the `ratings_small` file extracts ratings of 700 users for the 9,000 movies.  

## Problem One: Top 5 Popular Movies by Genre
### 1.1 Assumptions 
Item popularity is widely used as strong signal and metrics for recommender system. In *(Ji, Sun, Zhang & Li, 2020)*, popularity is defined in three groups, namely `MostPop`, `RecentPop` and `DecayPop`. `MostPop` refers to a global popularity baseline, where items to be recommended are the ones with highest interactions. `RecentPop` extends the timestamp as a factor which considers the time point of interaction. `DecayPop` includes further long slots even before the time stamp of interactions.
In this problem, we consider only the mainstream `MostPop`, as it provides simple and easy computation for large dataset, and is universally applied for tackling cold-start problems. *(Arapakis, Cambazoglu & Lalmas, 2016) (Sedhain et al., 2014)* Although `MostPop` has certain fallbacks, such as popularity bias of active/inactive users and long-tail items, it could be moderated with proper techniques in below section.

### 1.2 Solutions
For this solution, a popularity recommender system using Bayesian Weighted Rank is considered. In the original IMBD genre movie ranking, similar algorithm is applied. Here we will construct a simple ranking process then compare with the original IMBD popularity ranking. Formula is as follows:


<img src="https://render.githubusercontent.com/render/math?math=e^{i%20\pi}%20=%20-1">



