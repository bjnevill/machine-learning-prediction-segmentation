# Submitted report — text edition

Original Semester 3 report narrative and values, with cover sheets, identifiers and submission forms omitted. Figures are not reproduced in this text edition. The author acknowledged generative AI support for learning and explanations.

2. Summary / Overview

I used both supervised and unsupervised machine learning techniques to solve two different business problems using real-world datasets. The first task was on predicting wine quality scores using supervised learning, while the second task focused on customer segmentation using unsupervised learning. These show how machine learning can be used to support business decision-making through prediction and discovering patterns.

The task of the first task was to build a regression model which is able to predict  the quality score of white wine based on its physicochemical properties, including acidity, residual sugar, chlorides, sulphates, density and alcohol content. Predicting wine quality using measurable characteristics allows wine producers to assess product quality more consistently before manual sensory evaluation, improving quality control and supporting production decisions.

I explored the Wine Quality dataset using summary statistics, histograms, boxplots and a correlation heatmap to understand the characteristics of the data. The dataset contained no missing values but several variables contained outliers. Hence, IQR capping was applied to reduce the effect of extreme values while preserving the rows. As several predictors were strongly right-skewed, Power Transformation was applied to selected variables to reduce skewness before standardisation. StandardScaler was then applied to place the features on a comparable scale. This was particularly important for K-Nearest Neighbours because it calculates distances between observations. Three supervised regression models were developed Linear Regression, Decision Tree Regressor and K-Nearest Neighbours Regressor. Their performances were evaluated using RMSE and R2. Decision Tree Regressor was further improved using GridSearchCV, and the tuned model was selected as the final model because it provided a good balance between predictive performance and generalisation.

The second task involved customer segmentation using the Online Retail dataset. Unlike supervised learning, this task did not have a target variable. Instead, the objective was to discover meaningful customer groups based on purchasing behaviour that could support marketing and customer management strategies. Since clustering should represent customer behaviour rather than individual transactions, the transactional data was aggregated to customer level.

The Online Retail dataset was cleaned by removing records with missing CustomerID values, cancelled invoices, returned items, negative quantities and invalid unit prices. A TotalPrice feature was created by multiplying Quantity and UnitPrice. Customer-level features were then engineered using the RFM framework given in brief, consisting of Recency, Frequency and Monetary value, together with AverageBasketValue to capture average spending per invoice. Boxplots were used to identify extreme customer values, followed by IQR capping to reduce their influence. Finally, StandardScaler was applied because both K-means and Hierarchical Clustering are distance-based algorithms that require features to be on a comparable scale.

Two clustering models were developed and compared. Four clusters were selected for K-means using the Elbow Method, while two clusters were selected for Hierarchical Clustering using the dendrogram. K-means achieved a Silhouette Score of 0.409, while Hierarchical Clustering achieved a higher score of 0.448. Although Hierarchical Clustering produced slightly better cluster separation, its two clusters only provided a broad division between active higher-value customers and less-active lower-value customers. K-means was selected as the preferred model because its four segments provided more detailed and actionable customer insights.3. Task 1: Wine Quality (Quality Score Prediction)
(Word count: Min 1000 / Max 3000 words)

3.1 Problem Understanding

The objective of this task was to develop a supervised machine learning regression model to predict the quality score of white wine using its physicochemical properties. The dataset contains various chemical measurements, including fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, sulphates, density, pH, alcohol content and sulphur dioxide levels. The target variable is the wine quality score, which ranges from 3 to 9.

Usually, wine quality is assessed by experienced wine tasters. However, this process can be subjective, time-consuming and costly. By using measurable chemical properties, machine learning can provide a more consistent and objective method of estimating wine quality before manual tasting. This can help wine producers improve quality control, identify products that may require further inspection, and support production and pricing decisions or even speed the process up.

Since the quality score is a continuous numerical value rather than a category, this problem is a regression task. Hence, supervised learning algorithms were used because the dataset contains historical records with known quality scores that can be used to train predictive models. In this assignment, Linear Regression, Decision Tree Regressor and K-Nearest Neighbours Regressor were developed and compared to determine which model produced the most accurate predictions.
The success of the models was evaluated using RMSE R2. A lower RMSE indicates that the predicted quality scores are closer to the actual values, while a higher R2 indicates that the model explains a larger proportion of the variation in wine quality. The model with the best balance between prediction accuracy and generalisation to unseen data was selected as the final model.


3.2 Data Exploration

I explored the dataset to understand its structure, data quality and feature distributions before any preprocessing or model development was carried out. Basic inspection using head(), shape(), info() and describe() gave me a overview of the dataset and confirmed that it contained 4,898 observations and 12 variables. Eleven variables represented the physicochemical properties of white wine, while the last variable, quality, was the target variable to be predicted.

A missing value check showed that all variables contained complete data. Hence, I used no missing value treatment such as deletion or imputation was required. This simplified the data preparation process and allowed all observations to be retained for model training.

Figure 1:Distribution of Wine Quality variables
Histograms were then used to examine the distributions of the numerical variables. Several features, including volatile acidity, citric acid, residual sugar, chlorides, free sulphur dioxide and sulphates, showed noticeable right-skewed distributions. In contrast, variables such as fixed acidity, pH, density and total sulphur dioxide appeared more evenly distributed. The target variable, quality, was concentrated mainly around scores of 5, 6 and 7, showing that very low and very high quality wines were uncommon.

Figure 2: Boxplots showing potential outliers in the Wine Quality predictor variables
Boxplots were used to identify potential outliers in each feature. Several variables contained extreme observations, particularly residual sugar, chlorides, free sulphur dioxide and sulphates. These outliers were not immediately removed because they may represent valid wine samples. Hence, they were considered during the preprocessing stage, where an appropriate outlier treatment method was applied.

This process provided clear understanding for me of the dataset and highlighted several characteristics that influenced the the next preprocessing steps. In particular, the presence of skewed variables and extreme values justified the use of data transformation and outlier treatment before building the machine learning models.


3.3 Data Cleansing and Transformation

Before developing the model,  I used several preprocessing techniques to improve the quality of the input data and ensure that the machine learning models could learn more effectively. Since the dataset did not contain any missing values, no imputation or row removal was required.
I first divided the dataset into training and testing sets using an 80:20 ratio. The training set was used to fit all preprocessing methods, while the same fitted transformations were applied to the testing set. This prevents data leakage, where information from the testing data could unknowingly influence the model during training.

Although the dataset did not contain missing values, the boxplots showed that several variables contained extreme values. To reduce the influence of these outliers without removing observations, IQR capping was applied using the Winsorizer with IQR method. This approach limits extreme values to the calculated upper and lower IQR boundaries while keeping all samples in the dataset. Keeping all observations was preferred because the extreme values may still represent valid wine samples.

The distribution plots also showed that several variables, including volatile acidity, citric acid, residual sugar, chlorides, free sulphur dioxide and sulphates, were noticeably right-skewed. To reduce skewness, I applied a Power Transformation to these selected variables after IQR capping. The transformation was only applied to variables with clear skewed distributions, while variables that already had relatively balanced distributions were left unchanged. Applying the transformation after outlier treatment helped produce more balanced feature distributions while reducing the influence of long-tailed values for Linear models such as Linear Regression.

Finally, StandardScaler was applied before developing the Linear Regression and KNN models. Standardisation ensures that variables measured on different scales contribute equally during model training. This preprocessing step is particularly important for K-Nearest Neighbours because it relies on distance calculations. The Decision Tree was trained using the capped but unscaled features because tree-based models are not sensitive to feature scaling.





3.4 Correlation Analysis

Figure 3: Correlation heatmap of the Wine Quality variables

I used a correlation heatmap generated to examine the relationships between the physicochemical features and the target variable, quality, and the relationships of the predictor variables. This analysis helped me identify which features are more strongly associated with wine quality and highlights possible multicollinearity between predictors.

The heatmap showed that alcohol had the strongest positive correlation with wine quality approximately 0.44, indicating that wines with higher alcohol content generally received higher quality scores. In contrast, density, chlorides, and volatile acidity showed moderate negative correlations with quality, suggesting that higher values of these variables were generally associated with lower quality wines. This shows that that some chemical properties have a greater influence on wine quality than others.

The correlation analysis also showed that some predictor variables were strongly correlated with each other. Residual sugar and density had a high positive correlation of approximately 0.84, while density and alcohol showed a strong negative correlation of approximately -0.78. These relationships indicate possible multicollinearity, where two predictors contain similar information.
Although multicollinearity was identified, the correlated features were not removed in the final dataset. Correlation alone does not necessarily reduce prediction accuracy, and removing features without evaluating the impact on model performance may result in the loss of useful information and since testing will take up a lot more time, it was just kept.

The correlation heatmap also showed that citric acid had only a very weak relationship with wine quality. This observation supported further experimentation with feature selection using p-value, although the final comparison showed that removing the feature did not improve the predictive performance of the Linear Regression model. Therefore, the original feature set was retained for the final analysis.

3.5 Modeling (At Least Two Supervised Models)


I made three supervised regression models to predict wine quality: Linear Regression, Decision Tree Regressor, and K-Nearest Neighbours (KNN) Regressor. These models were selected because they are different ML models and allow their predictive performance to be compared on the same dataset.

Linear Regression was selected as one of the regression models because it can model and evaluate linear relationships between the physicochemical properties and wine quality. Its performance also provided a useful comparison with the non-linear Decision Tree and distance-based KNN models.

Decision Tree Regressor was used because it can model non-linear relationships between the physicochemical properties and wine quality. Unlike Linear Regression, Decision Trees continuously split the data into smaller groups based on the feature values, allowing them to capture more complex relationships  between variables. In addition, Decision Trees are generally less sensitive to multicollinearity and feature scaling.

K-Nearest Neighbours (KNN) Regressor was included as an additional distance-based regression model. Instead of learning an explicit mathematical relationship, KNN predicts the quality score based on the values of the nearest neighbouring observations. Because KNN relies on distance calculations, the predictor variables were standardised before model training to ensure that all features contributed fairly to the distance measurement.

To improve the performance of the Decision Tree model, GridSearchCV was used to perform hyperparameter tuning. Different combinations of maximum tree depth, minimum samples required to split a node, and minimum samples required at each leaf node were evaluated using cross-validation. The combination that achieved the best cross-validation performance was selected as the final Decision Tree model. The optimal hyperparameters obtained were:
Maximum depth = 6
Minimum samples split = 2
Minimum samples leaf = 10
GridSearchCV systematically compared different hyperparameter combinations using cross-validation and selected the combination with the lowest average prediction error. The tuned Decision Tree model was then compared with the baseline Decision Tree, Linear Regression and KNN Regressor using RMSE and R² on both the training and testing datasets



3.6 Performance Analysis


Figure 4: Performance results of the original Linear Regression, Decision Tree and KNN models

Figure 5: Performance results after Linear Regression feature selection and Decision Tree hyperparameter tuning

The supervised learning models were evaluated RMSE and the R2. RMSE measures the average size of the prediction errors, where a lower value indicates better predictive accuracy. R2 measures the proportion of variation in wine quality explained by the model, where a higher value indicates better performance.

The Linear Regression model achieved a training RMSE of 0.745 and a testing RMSE of 0.746. Its training and testing R² values were 0.295 and 0.282, respectively. The very small difference between the training and testing results shows that the model generalised consistently and did not show signs of overfitting. However, its relatively low R² indicates that the linear model could explain only around 28% of the variation in the wine quality scores on the testing data. This suggests that the relationship between the physicochemical properties and wine quality may not be entirely linear.

The original Decision Tree Regressor achieved a training RMSE of 0.772 and a testing RMSE of 0.776, with training and testing R² values of 0.242 and 0.223. Although the training and testing results were similar, the model performed worse than Linear Regression on both evaluation measures. Therefore, the original Decision Tree did not provide a strong predictive model for this dataset. Its similar training and testing performance also indicates that the main issue was underfitting rather than overfitting, as it was unable to capture enough of the patterns in either dataset.

The K-Nearest Neighbours Regressor achieved the best testing performance among the initial models. It obtained a training RMSE of 0.575 and a testing RMSE of 0.714, while its training and testing R² values were 0.579 and 0.341, respectively. The testing RMSE was the lowest and the testing R² was the highest among all models developed. This shows that KNN captured the relationships in the dataset more effectively than Linear Regression and the original Decision Tree. However, the noticeable difference between its training and testing results suggests some overfitting, as the model performed considerably better on the training data.

Feature selection was also tested for Linear Regression using statistically significant variables. However, the selected-feature model achieved a testing RMSE of 0.746 and a testing R² of 0.281, which were almost identical to and slightly worse than the original Linear Regression model. This suggests that removing the less significant features did not improve predictive performance. Therefore, the full Linear Regression model was more suitable for comparison.

GridSearchCV was used to tune the Decision Tree hyperparameters. The tuned Decision Tree Regressor achieved a training RMSE of 0.676 and a testing RMSE of 0.725, with training and testing R² values of 0.420 and 0.322. These results were a clear improvement over the original Decision Tree. Hyperparameter tuning allowed the model to capture more useful patterns and reduced its testing error. However, its testing performance remained slightly weaker than KNN, which achieved a lower testing RMSE of 0.714 and a higher testing R² of 0.341.

Although the KNN Regressor achieved the lowest testing RMSE and the highest testing R², it also exhibited a noticeably larger gap between its training and testing performance, suggesting a greater degree of overfitting. In contrast, the tuned Decision Tree Regressor achieved slightly lower predictive performance but demonstrated a better balance between training and testing results, indicating stronger generalisation to unseen data. As the objective of machine learning is to develop models that perform reliably on new data rather than memorise the training data, the tuned Decision Tree Regressor was selected as the final model. In addition, the Decision Tree provides better interpretability through its tree structure, making it easier to understand how predictions are generated.

4. Task 2: Online Retail (Customer Segmentation)
(Word count: Min 1000 / Max 3000 words)
4.1 Problem Understanding
The objective of this task was to perform customer segmentation using the Online Retail dataset. Unlike the previous task, where the target variable was known, this dataset does not contain given customer groups. Hence, an unsupervised learning approach was used to identify natural patterns and similarities among customers based on their purchasing behaviour.

Customer segmentation is an important business application because customers often have different purchasing habits, spending levels and shopping frequencies. By grouping customers with similar characteristics, businesses can better understand their customer base and develop targeted marketing strategies. For example, loyal high-value customers can be rewarded through loyalty programmes, while inactive customers can be targeted with promotional campaigns to encourage repeat purchases. Effective customer segmentation can improve customer satisfaction, increase sales and support more informed business decision-making.

Since no labelled target variable was available, clustering techniques were selected to discover hidden patterns within the data. Customer purchasing behaviour was represented using Recency, Frequency, Monetary (RFM) values and Average Basket Value, which provide a meaningful summary of each customer's shopping activity as recommended in the brief. Customers with similar values across these features are expected to belong to similar behavioural groups.

I made 2 clustering algorithms, K-Means Clustering and Hierarchical Clustering in this assignment. Their performance was evaluated using the Silhouette Score, which measures how well each customer fits within its assigned cluster compared to other clusters. In addition to quantitative evaluation, the resulting customer segments were interpreted based on their purchasing characteristics to determine which clustering approach produced the most meaningful and actionable business insights.





4.2 Data Exploration and Preparation

I explored the Online Retail dataset to understand its structure, data quality and suitability for customer segmentation. Basic inspection using head(), shape(), info() and describe() provided an overview of the dataset, including the number of records, variable types and summary statistics. The dataset contained transaction-level information such as invoice numbers, product descriptions, quantities purchased, unit prices, invoice dates and customer identifiers.
A missing value analysis showed that the CustomerID field contained missing values. Since customer segmentation requires grouping transactions by individual customers, records without a valid CustomerID could not be assigned to a customer and were therefore removed. This ensured that every transaction could be accurately aggregated during feature engineering.
The dataset was then cleaned by removing cancelled transactions and invalid records. Cancelled invoices, identified by invoice numbers beginning with the letter 'C', were excluded because they do not represent completed purchases. Transactions with non-positive Quantity or UnitPrice values were also removed, as they represent returns, corrections or invalid sales records that do not accurately reflect customer purchasing behaviour.
The InvoiceDate variable was converted from a text format into a datetime format. This allowed transaction dates to be processed correctly and was necessary for calculating customer recency during feature engineering.

Figure 6: Boxplot of UnitPrice and Quantity in the original Online Retail dataset

Boxplots of Quantity and UnitPrice were created to examine the transaction data and identify extreme values. Both variables contained several unusually large observations. These values were not automatically removed because they may represent genuine bulk purchases or expensive products. Invalid transactions, such as cancelled invoices, non-positive quantities and non-positive unit prices, were removed before customer-level features were created. The remaining extreme customer values were handled later using IQR capping after the data had been aggregated by customer.

Figure 8: Distributions of Quantity and UnitPrice after excluding extreme values for visualisation
Histograms were also created to examine the distributions of Quantity and UnitPrice after excluding invalid values and the most extreme observations for visualisation purposes. Both variables remained right-skewed, showing that most transactions involved small quantities of lower-priced products, while larger quantities and higher unit prices occurred less frequently. The extreme observations were excluded only to make the histograms easier to interpret and were not removed from the cleaned dataset at this stage. They were only used to observe. NO transformations needed as models used they do not benefit by tranforming skewness.
Overall, I removed incomplete and invalid records while preserving meaningful customer transactions during the data preparation process. This ensured that the dataset was clean, consistent and suitable for generating reliable customer segments in the subsequent clustering analysis.


4.3 Feature Engineering for Clustering

Since clustering is performed at the customer level rather than the transaction level, the cleaned transaction data was aggregated to create meaningful customer features that summarise purchasing behaviour. Four features were engineered to represent each customer's shopping activity: Recency, Frequency, Monetary (RFM) and Average Basket Value as stated in the brief.

Recency measures the number of days since a customer's most recent purchase. Customers with lower recency values have purchased more recently and are generally considered to be more engaged with the business.

Frequency represents the number of unique purchases made by each customer. Customers with higher frequency values tend to shop more regularly and may indicate stronger customer loyalty.

Monetary measures the total amount spent by each customer and was calculated by summing the transaction values, where each transaction value was obtained by multiplying Quantity by UnitPrice. Customers with higher monetary values contribute more revenue to the business.

An additional feature, Average Basket Value, was also created by dividing the total monetary value by the purchase frequency. This feature provides an indication of the average amount a customer spends per purchase and helps distinguish customers with similar total spending but different purchasing patterns.

Figure 9: Boxplot of RFM including AverageBasketValue
After feature engineering, boxplots were used to examine the distributions of the customer-level features. As shown in Figure 9, Monetary, Frequency and Average Basket Value contained several extreme values due to a small number of customers making exceptionally large or high-value purchases. Since these customers may represent genuine purchasing behaviour, the records were retained. Instead, IQR Winsorization was applied to cap the extreme values while preserving all customer records for clustering.

Finally, the engineered features were standardised using StandardScaler. Since K-Means and Hierarchical Clustering are distance-based algorithms, variables measured on larger scales could otherwise dominate the clustering process. Standardisation ensured that all customer features contributed equally when calculating distances between customers.

Overall, the engineered features captured different aspects of customer purchasing behaviour and provided a suitable numerical representation for the clustering models to identify meaningful customer segments.



4.4 Build TWO Different Clustering Models

Two clustering algorithms, K-Means Clustering and Hierarchical Clustering, were developed to group customers based on their purchasing behaviour. Both models were trained using the standardised customer features to ensure that all variables contributed equally to the distance calculations.



Figure 10: Elbow Method for selecting the number of K-means clusters

The first model, K-Means Clustering, partitions customers into a predefined number of clusters by assigning each customer to the nearest cluster centroid. The algorithm repeatedly updates the centroid locations and customer assignments until the clusters become stable. To determine an appropriate number of clusters, the Elbow Method was applied by plotting the Within-Cluster Sum of Squares against different values of k. Based on the elbow plot, four clusters were selected as they provided a good balance between reducing cluster variation and avoiding unnecessary model complexity.


Figure 11: Dendrogram for selecting the number of Hierarchical clusters

Figure 12: Silhouette Scores for different numbers of Hierarchical clusters
The second model, Hierarchical Clustering, groups customers by progressively merging the most similar clusters based on the distance between observations. Unlike K-Means, Hierarchical Clustering does not require cluster assignments to be updated iteratively. Instead, the relationships between customers are visualised using a dendrogram, which illustrates how clusters are merged at different distance levels. The dendrogram was analysed to determine a suitable cutting point, resulting in two customer clusters. As cutting might have some bias , silhouette score graph to reinforce it later on.


Figure 13: Three-dimensional visualisation of the four K-means customer clusters


Figure 14: Three-dimensional visualisation of the two Hierarchical customer clusters
After both clustering models were built, cluster labels were assigned to each customer. The resulting clusters were visualised using three-dimensional scatter plots and summarised using descriptive statistics. These visualisations provided an initial understanding of how customers were grouped before formally evaluating the clustering performance in the next section.


4.5 Evaluate and Compare the Models

Figure 15: Summary statistics and customer counts for the K-means clusters

Figure 16: Summary statistics and customer counts for the Hierarchical clusters

Figure 17: K-means and Hierarchical Clustering Silhouette Score results
The K-means and Hierarchical Clustering models were evaluated using the Silhouette Score. The score measures how similar customers are to other customers in the same cluster compared with customers in different clusters. A higher score indicates better cluster separation.

K-means produced four customer segments and achieved a Silhouette Score of 0.409. The four clusters contained 2,016, 715, 962 and 645 customers respectively. The cluster summary showed meaningful differences in Recency, Frequency, Monetary value and Average Basket Value, allowing each group to be interpreted based on its purchasing behaviour.

Hierarchical Clustering produced two clusters and achieved a higher Silhouette Score of 0.448. Cluster 0 contained 915 more-active and higher-value customers, while Cluster 1 contained 3,423 less-active and lower-value customers. The higher score indicated that the two Hierarchical clusters had slightly better separation than the four K-means clusters.

Although Hierarchical Clustering achieved the higher Silhouette Score, its two clusters provided only a broad division of the customer base. K-means was selected as the preferred model because its four clusters provided more detailed and actionable customer segments. Therefore, the final selection considered both the evaluation score and the business usefulness of the resulting clusters.


4.6 Segment Naming and Interpretation


Figure 18: Boxplots comparing Recency, Frequency, Monetary value and Average Basket Value across the K-means clusters

After selecting the K-means model, the four customer clusters were analysed based on their Recency, Frequency, Monetary and Average Basket Value characteristics. Each cluster was assigned a descriptive name based on its purchasing behaviour and business value.

Cluster 0 – Recent Low-Value Customers
This was the largest segment, containing 2,016 customers. These customers purchased relatively recently, with an average Recency of 45.27 days. However, their average Frequency of 2.65, Monetary value of 633.63 and Average Basket Value of 246.27 were relatively low. The business could encourage these customers to purchase more frequently through membership promotions, product recommendations and bundle offers.

Cluster 1 – High-Value Loyal Customers
This segment contained 715 customers. They had the lowest average Recency of 20.79 days, highest Frequency of 14.07 and highest Monetary value of 7,855.01. This indicates that they purchased recently, bought frequently and spent the most overall. The business should retain these valuable customers through loyalty rewards, personalised offers and early access to promotions.

Cluster 2 – At-Risk Low-Value Customers
This segment contained 962 customers. They had the highest average Recency of 249.21 days, lowest Frequency of 1.46 and lowest Monetary value of 331.27. This means they had not purchased for a long time and contributed relatively little revenue. Reminder emails, win-back discounts and limited-time promotions could be used to encourage them to return.

Cluster 3 – High-Basket Occasional Customers
This segment contained 645 customers. They had moderate Recency and Frequency values but the highest Average Basket Value of 1,167.54. This indicates that they purchased occasionally but spent considerably more during each purchase. The business could target them with premium product recommendations, high-value bundles and personalised offers.
For Hierarchical Clustering, Cluster 0 contained 915 more-active and higher-value customers, while Cluster 1 contained 3,423 less-active and lower-value customers. These two segments provided a simple overview of the customer base but were less detailed than the four K-means segments.

Overall, Hierarchical Clustering achieved a higher Silhouette Score of 0.448 compared with 0.409 for K-means. However, K-means was selected as the preferred model because its four customer segments were more detailed and supported more specific marketing and customer-management strategies

5. Summary and Further Improvements
(Word count: Min 800 / Max 2000 words)
This assignment allowed me to apply both supervised and unsupervised machine learning techniques to solve two different business problems using real-world datasets. Task 1 focused on predicting wine quality using regression models, while Task 2 focused on grouping customers according to their purchasing behaviour. Throughout both tasks, I followed the main machine learning workflow, including understanding the problem, exploring and cleaning the data, preparing the features, building different models and evaluating the results. This helped me understand that machine learning is not only about training a model, as many decisions have to be made before and after the modelling stage.

For the Wine Quality dataset, the objective was to predict the quality score of white wine using its physicochemical properties. The dataset was first explored using summary statistics, histograms, boxplots and a correlation heatmap. There were no missing values, but several variables contained outliers and skewed distributions. I used IQR capping to reduce the influence of extreme values without removing the observations completely. The Yeo-Johnson Power Transformation was then applied to selected skewed features, followed by standardisation. All preprocessing methods were fitted using only the training data before being applied to the testing data, which helped prevent data leakage.

Three regression models were developed and compared: Linear Regression, Decision Tree Regressor and K-Nearest Neighbours Regressor. Linear Regression achieved a testing RMSE of 0.746 and testing R² of 0.282. Feature selection was also tested by removing citric acid based on its high p-value, but this did not improve the model. KNN achieved the best testing performance, with an RMSE of 0.714 and R² of 0.341. However, its training performance was noticeably better than its testing performance, showing some signs of overfitting.

The original Decision Tree did not perform well because its low maximum depth caused it to underfit the data. GridSearchCV was used to test different combinations of maximum depth, minimum samples split and minimum samples leaf. After tuning, the Decision Tree achieved a testing RMSE of 0.725 and R² of 0.322. Although KNN was slightly more accurate on the testing data, the tuned Decision Tree had a smaller difference between its training and testing results. It was also easier to interpret because its prediction process can be represented using decision rules. Therefore, the tuned Decision Tree was selected as the preferred final model based on its balance between predictive performance, generalisation and interpretability. This decision showed me that the model with the best testing score is not always automatically the most suitable model.

For the Online Retail dataset, the objective was to discover meaningful customer groups without using a target variable. The dataset contained missing CustomerID values, cancelled invoices, returns and invalid transactions. Records without a CustomerID were removed because they could not be linked to a customer. Cancelled invoices, non-positive quantities and non-positive unit prices were also removed because they did not represent completed purchases. A TotalPrice feature was created by multiplying Quantity and UnitPrice.

The cleaned transactions were then grouped by CustomerID to create Recency, Frequency and Monetary features. Average Basket Value was also included to represent how much each customer spent per purchase. Since the customer-level features contained extreme values, IQR capping was applied before standardisation. StandardScaler was important because K-means and Hierarchical Clustering use distances to group similar customers, and variables with larger values could otherwise influence the clustering more strongly.

Four clusters were selected for K-means using the Elbow Method, while two clusters were selected for Hierarchical Clustering using the dendrogram. K-means achieved a Silhouette Score of 0.409, while Hierarchical Clustering achieved a higher score of 0.448. This indicated that the two Hierarchical clusters had slightly better separation. However, they only divided the customers into two broad groups consisting of more-active higher-value customers and less-active lower-value customers.

K-means produced four more detailed groups: Recent Low-Value Customers, High-Value Loyal Customers, At-Risk Low-Value Customers and High-Basket Occasional Customers. These segments gave a clearer understanding of the different purchasing behaviours and allowed more specific business actions to be recommended. For example, loyal customers could receive rewards, while at-risk customers could be targeted with win-back promotions. Therefore, K-means was selected as the preferred clustering model even though its Silhouette Score was slightly lower. This taught me that evaluation metrics are important, but the usefulness of the results in the business context should also be considered.

There are several possible improvements for Task 1. The KNN model was only developed using five neighbours, so different values of n_neighbors could be tested using GridSearchCV. This may reduce overfitting and improve its generalisation. A wider range of Decision Tree hyperparameters could also be tested instead of limiting the maximum depth to seven. Repeated cross-validation may produce a more reliable evaluation because the model would be tested across several different data splits. The dataset also contained fewer wines with extremely low or high quality scores. Collecting more samples for these less common scores may help the models learn their characteristics more effectively. More complex models could also be used and tested.

Task 2 could also be improved by including more customer behaviour features. Examples include the types of products purchased, return rate, seasonal purchasing patterns and customer lifetime value. These features may reveal customer differences that cannot be identified using RFM values alone. The stability of the clusters could also be checked by running the models using different samples or random states. It may also be useful to compare the results before and after IQR capping because capping could reduce the differences of genuine high-value customers.

One of my main takeaways from this assignment was the importance of checking whether each preprocessing step was actually used by the model. During Task 2, I created the capped customer data as X_capped using the Winsorizer. However, I initially passed the original X into StandardScaler instead of passing X_capped. This meant that although the capping code ran without an error, the clustering models were still trained using the original extreme values. As a result, K-means produced highly uneven clusters, including one cluster with only two extremely high-value customers. After identifying the mistake, I changed the code from scaler.fit_transform(X) to scaler.fit_transform(X_capped) and reran all the clustering cells. The corrected model produced more reasonably sized clusters containing 2,016, 715, 962 and 645 customers, and the cluster interpretations also changed. This experience taught me that code running successfully does not always mean that the entire workflow is correct. I need to trace which version of the data is passed into each step and verify whether the final outputs make sense. It also showed me why model results and interpretations must be updated whenever the preprocessing process changes.

Generative AI was used as a learning and support tool during the assignment. It helped me understand unfamiliar concepts, check my explanations and improve the structure of some sections. However, I also noticed that AI suggestions could sometimes be inaccurate or may not match the actual notebook output. Because of this, I had to compare its suggestions against my code, model results and assignment requirements before using them. In my opinion, Generative AI is useful for supporting learning, but it should not replace understanding the code or making my own decisions. Overall, this assignment strengthened my ability to apply the full machine learning process and explain the results based on both technical evidence and the business problem.
