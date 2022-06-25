# movie_recommender
 
This project is based on a give dataset from an online movie streaming company which includes movie information, user IDs and user ratings. Aim of this project is to 
1) recommend top 5 popular movies for given genres, and 

2) to recommend top 3 movies for 3 given user IDs. 

In tackling the first problem, a simple Weighted Rating Rank method is proposed. For the second problem, a collaborative filtering technique is chosen, in particular Probabilistic Matrix Factorization algorithm. Both methods are implemented in Python. The results are evaluated in RMSE and MAE, which presented satisfactory performance by benchmarking given values. However, in comparison with other empirical findings, there is still room to improve the accuracy. For this purpose, further studies and techniques could be experimented, such as incorporating more user-item information, and applying hybrid filtering techniques.