# High/low income classification from US Census data

The data used in this analysis can be found in the following zip file: http://thomasdata.s3.amazonaws.com/ds/us_census_full.zip. This US Census dataset contains detailed but anonymized information for approximately 300,000 people.

The archive contains 3 files: 
1) A large training file (csv)
2) Another test file (csv)
3) A metadata file (txt) describing the columns of the two csv files (identical for both)

The [Jupyter notebook](census_income.ipynb) contains all the code along with explanatory comments. Please refer to this notebook for full details of my analysis. In summary, I have performed some EDA, cleaned the data, and created several models to predict whether or not someone earns greater than $50,000 per year.

Initially, I tried Logistic Regression, K-Nearest Neighbors, Random Forest, and XGBoost models on the cleaned data. Class imbalance was quite high so I applied SMOTE rebalancing and re-ran the models without seeing significant improvement; I was using the F1 score to tune the models which is a robust metric with imbalanced datasets.

Next, I applied Principle Component Analysis and determined that 10 components could capture 99.9% of the variance, so used these 10 features in new models. Performance did not improve for the Random Forest or XGBoost, which were already the highest-performing models, so I didn't feel it necessary to use PCA in the final model (training time significantly improved, however, so in a production setting if the models will be frequently retrained, PCA is probably a good idea).

I determined the most influential features to be:

1. 8.49% - dividends_from_stocks
2. 8.01% - capital_gains
3. 7.81% - age
4. 6.94% - detailed_occupation_recode
5. 3.83% - detailed_industry_recode
6. 3.63% - num_persons_worked_for_employer
7. 3.18% - weeks_worked_in_year
8. 2.93% - sex_ Male
9. 2.60% - capital_losses
10. 2.36% - major_occupation_code_ Executive admin and managerial

Finally, I used an unseen test data set to measure the performance of the XGBoost model:

- 94.7% - Accuracy
- 59.5% - F1 score
- 74.4% - Precision
- 49.6% - Recall