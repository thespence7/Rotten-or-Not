# Rotten-or-Not

Early on in my childhood, I loved watching movies. I remember watching The Lion King on VHS (I know, I sound terribly old with that reference), so many times that I somehow burned the actual film strip in my copy. As I matured, I critiqued movies based on their content and how well they were made. Relating to my love of movies and being able to review them, I wanted to pursue a data science project involving a process where I predicted a critic's rating of a movie based on their review. In this project, I explore how to determine a critic's freshness or rottenness using Natural Language Processing techniques. 

# About the data
This dataset was scraped from Rottentomatoes.com and placed in rotten_tomatoes_reviews.csv

There was relatively little cleaning that I had to do for this dataset, but I did implement an easier rating system. If a review was 60% or greater (fresh), they received a 1. If a review was less than 60%(rotten), they received a 0. The 2 features are listed below:
- `Freshness`: as described above, this column contains 1s and 0s pertaining to the critics score
- `Review`: critics review in string format

# Part 1: Predicting Critics Freshness
## Initial EDA
In the reviews column i found that there were multiple duplicate reviews or reviews that contained a link to a review not on Rottentomatoes.com, which I decided to solely remove the duplicates. 

After this, my data was decently balanced with 180k Fresh reviews and 150k rotten reviews

### Modeling
Initially created test, train sets for my data. Then I used TFIDF to vectorize the data along with a MLB labeler due to my data being multiple words in each review.

Initial cross validation accuracy, precision, and recall scores using various models:

**Logistic Regression** | **Random Forest** | **Gradient Boosting**
----------------------- | ----------------- | --------------------
Train Set accuracy: 78.6% | Train Set accuracy: 71.3% | Train Set accuracy: 63.2%



