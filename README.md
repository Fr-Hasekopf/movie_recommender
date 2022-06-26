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
In *(Ji, Sun, Zhang & Li, 2020)*, popularity is defined in three groups, namely `MostPop`, `RecentPop` and `DecayPop`.   
`MostPop` refers to a global popularity baseline, where items to be recommended are the ones with highest interactions.   
`RecentPop` extends the timestamp as a factor which considers the time point of interaction.  
 `DecayPop` includes further long slots even before the time stamp of interactions.  
In this problem, we consider only the mainstream `MostPop`, as it provides simple and easy computation for large dataset, and is universally applied for tackling cold-start problems. *(Arapakis, Cambazoglu & Lalmas, 2016) (Sedhain et al., 2014)*   
Although `MostPop` has certain fallbacks, such as popularity bias of active/inactive users and long-tail items, it could be moderated with proper techniques.

### 1.2 Solutions
For this solution, a popularity recommender system using __Bayesian Weighted Rank__ is considered. In the original IMBD genre movie ranking, similar algorithm is applied. Here we will construct a simple ranking process then compare with the original IMBD popularity ranking. Formula is as follows:  
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?WR%20%3D%20%5Cfrac%7Bv%7D%7Bv&plus;m%7DR%20&plus;%20%5Cfrac%7Bm%7D%7Bv&plus;m%7DC">
</p>

where WR denotes weighted Rank, R denotes user average vote, v denotes vote count for specific movie, m denotes minimum votes to be listed and C denotes overall average vote of all movies. *(Rajarajeswari et al., 2019)*  

The Bayesian Weighted Rank algorithm takes into consideration the comparability of ratings for popular movies and niche / long-tailed movies, where the former receives considerable number of votes and the latter very limited. If *Wilson interval score* is applied, votes of the latter will be significantly degraded. Therefore, the votes of long-tailed movies should be compensated with more voters.   
In the WR algorithm, each movie will be allocated m number of votes with C scores, then the v*R/(v+m) will compensate movies which receive low vote counts and improves the given votes as in a more fair scenario.

### 1.3 Results
The result is shown in (Fig 5) for top 5 movies in each genre, so that given a preferred genre, we could recommend the top movies for specific user.   
Examples include:   
`Top 5 Animation movies are:['The Lion King', 'Spirited Away', "Howl's Moving Castle", 'Princess Mononoke', 'My Neighbor Totoro'] Top 5 Comedy movies are:['Dilwale Dulhania Le Jayenge', 'Forrest Gump', 'Back to the Future', 'The Intouchables', 'The Grand Budapest Hotel']`

## Problem Two: Movie Recommender
### 1.1 Assumptions and solution
*(Sunilkumar, 2020)* summaries movie recommender systems based four methods, namely `content based filtering`, `collaborative filtering`, `hybrid method` and `deep learning` approach. Among them, `collaborative filtering` is considered advantageous in coping with data sparsity problems.  
Here we also need to tackle with __data sparsity__. In the file `ratings`, we have identified that despite of a large movie dataset, ratings of certain movies are largely skewed, which would result in a sparse user-item matrix.    
In this case, collaborative filtering methods could contribute to computation reduction. In particular, PMF __(Probabilistic Matrix Factorization)__ proves to be efficient and accurate for Dimensionality Reduction.   
  
The PMF predicts item rating 𝑟̂𝑢𝑖 of a given user with the formula (1) , in addition, regularized squared error is minimized to estimate the unknown in formula (2). *(Salakhutdinov, 2007)(Hug, 2020a)*  
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Chat%7Br%7D_%7Bui%7D%20%3D%20%5Cmu%20&plus;%20b_u%20&plus;%20b_i%20&plus;%20q_i%5ETp_u">
</p>
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Csum%5Cleft%28r_%7Bui%7D%20-%20%5Chat%7Br%7D_%7Bui%7D%20%5Cright%29%5E2%20&plus;%20%5Clambda%5Cleft%28b_i%5E2%20&plus;%20b_u%5E2%20&plus;%20%7C%7Cq_i%7C%7C%5E2%20&plus;%20%7C%7Cp_u%7C%7C%5E2%5Cright%29">
</p>
  
Then stochastic gradient descent is applied in (3) – (6) to update the values.  
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <semantics>
    <mtable columnalign="right left" rowspacing="3pt" columnspacing="0em" displaystyle="true">
      <mtr>
        <mtd>
          <msub>
            <mi>b</mi>
            <mi>u</mi>
          </msub>
        </mtd>
        <mtd>
          <mi></mi>
          <mo stretchy="false">&#x2190;<!-- ← --></mo>
          <msub>
            <mi>b</mi>
            <mi>u</mi>
          </msub>
        </mtd>
        <mtd>
          <mo>+</mo>
          <mi>&#x03B3;<!-- γ --></mi>
          <mo stretchy="false">(</mo>
          <msub>
            <mi>e</mi>
            <mrow class="MJX-TeXAtom-ORD">
              <mi>u</mi>
              <mi>i</mi>
            </mrow>
          </msub>
          <mo>&#x2212;<!-- − --></mo>
          <mi>&#x03BB;<!-- λ --></mi>
          <msub>
            <mi>b</mi>
            <mi>u</mi>
          </msub>
          <mo stretchy="false">)</mo>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>b</mi>
            <mi>i</mi>
          </msub>
        </mtd>
        <mtd>
          <mi></mi>
          <mo stretchy="false">&#x2190;<!-- ← --></mo>
          <msub>
            <mi>b</mi>
            <mi>i</mi>
          </msub>
        </mtd>
        <mtd>
          <mo>+</mo>
          <mi>&#x03B3;<!-- γ --></mi>
          <mo stretchy="false">(</mo>
          <msub>
            <mi>e</mi>
            <mrow class="MJX-TeXAtom-ORD">
              <mi>u</mi>
              <mi>i</mi>
            </mrow>
          </msub>
          <mo>&#x2212;<!-- − --></mo>
          <mi>&#x03BB;<!-- λ --></mi>
          <msub>
            <mi>b</mi>
            <mi>i</mi>
          </msub>
          <mo stretchy="false">)</mo>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>p</mi>
            <mi>u</mi>
          </msub>
        </mtd>
        <mtd>
          <mi></mi>
          <mo stretchy="false">&#x2190;<!-- ← --></mo>
          <msub>
            <mi>p</mi>
            <mi>u</mi>
          </msub>
        </mtd>
        <mtd>
          <mo>+</mo>
          <mi>&#x03B3;<!-- γ --></mi>
          <mo stretchy="false">(</mo>
          <msub>
            <mi>e</mi>
            <mrow class="MJX-TeXAtom-ORD">
              <mi>u</mi>
              <mi>i</mi>
            </mrow>
          </msub>
          <mo>&#x22C5;<!-- ⋅ --></mo>
          <msub>
            <mi>q</mi>
            <mi>i</mi>
          </msub>
          <mo>&#x2212;<!-- − --></mo>
          <mi>&#x03BB;<!-- λ --></mi>
          <msub>
            <mi>p</mi>
            <mi>u</mi>
          </msub>
          <mo stretchy="false">)</mo>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>q</mi>
            <mi>i</mi>
          </msub>
        </mtd>
        <mtd>
          <mi></mi>
          <mo stretchy="false">&#x2190;<!-- ← --></mo>
          <msub>
            <mi>q</mi>
            <mi>i</mi>
          </msub>
        </mtd>
        <mtd>
          <mo>+</mo>
          <mi>&#x03B3;<!-- γ --></mi>
          <mo stretchy="false">(</mo>
          <msub>
            <mi>e</mi>
            <mrow class="MJX-TeXAtom-ORD">
              <mi>u</mi>
              <mi>i</mi>
            </mrow>
          </msub>
          <mo>&#x22C5;<!-- ⋅ --></mo>
          <msub>
            <mi>p</mi>
            <mi>u</mi>
          </msub>
          <mo>&#x2212;<!-- − --></mo>
          <mi>&#x03BB;<!-- λ --></mi>
          <msub>
            <mi>q</mi>
            <mi>i</mi>
          </msub>
          <mo stretchy="false">)</mo>
        </mtd>
      </mtr>
    </mtable>
    <annotation encoding="application/x-tex">\begin{split}b_u &amp;\leftarrow b_u &amp;+ \gamma (e_{ui} - \lambda b_u)\\
b_i &amp;\leftarrow b_i &amp;+ \gamma (e_{ui} - \lambda b_i)\\
p_u &amp;\leftarrow p_u &amp;+ \gamma (e_{ui} \cdot q_i - \lambda p_u)\\
q_i &amp;\leftarrow q_i &amp;+ \gamma (e_{ui} \cdot p_u - \lambda q_i)\end{split}</annotation>
  </semantics>
</math>







