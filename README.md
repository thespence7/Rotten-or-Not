# Rotten-or-Not

## Initial EDA
In the reviews column i found that there were multiple duplicate reviews or reviews that contained a link to a review not on Rottentomatoes.com, which I decided to solely remove the duplicates. 

After this, my data was decently balanced with 180k Fresh reviews and 150k rotten reviews

### Modeling
Initially created test, train sets for my data. Then I used TFIDF to vectorize the data along with a MLB labeler due to my data being multiple words in each review.

