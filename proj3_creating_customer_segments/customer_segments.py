
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Unsupervised Learning
# ## Project 3: Creating Customer Segments

# Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
# 
# The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.
# 
# Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[1]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
get_ipython().magic(u'matplotlib inline')

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"


# ## Data Exploration
# In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.
# 
# Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.

# In[2]:

# Display a description of the dataset
display(data.describe())


# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

# In[3]:

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [7, 77, 177]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)


# ### Question 1
# Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
# *What kind of establishment (customer) could each of the three samples you've chosen represent?*  
# **Hint:** Examples of establishments include places like markets, cafes, and retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant.

# **Answer:**
# 
# For index 0, it could be a medium-size grocery store because it requires every thing in a balanced way: The fresh and frozen cost is slightly smaller than median, milk, detergents_paper and delicatessen cost is close to  mean, grocery cost is close to 75% of total samples.  
# 
# For index 1, it could be a large-size grocery because fresh, milk, grocery and detergents paper cost is much larger than 75% of total samples. Delicatessen is to the median level. Frozen cost is near 25% of total samples, which is possible for a grocery store which does not focus on frozen product market.
# 
# For index 2, it could be an elegant cafe restaurant. The detergents paper it requires is significantly lower than the samples (close to min) and an elegant restaurant is highly possible to not consume much paper. Fresh, milk and delicatessen are near 75% of samples and frozen is to the median-level, which proves the restaurant focus on cooking fresh food with good quality latte coffee. 

# ### Implementation: Feature Relevance
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.
# 
# In the code block below, you will need to implement the following:
#  - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
#  - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
#    - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
#  - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
#  - Report the prediction score of the testing set using the regressor's `score` function.

# In[32]:

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
#print data.head()
keyName = 'Detergents_Paper'
new_label = data[keyName]
new_data = data.drop(keyName, axis = 1)
#print new_data.tail();

# TODO: Split the data into training and testing sets using the given feature as the target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, new_label, test_size=0.25, random_state=42)

# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0, min_samples_split=50)
regressor.fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print "training score = ", regressor.score(X_train, y_train)
print "prediction score = ", score


# ### Question 2
# *Which feature did you attempt to predict? What was the reported prediction score? Is this feature is necessary for identifying customers' spending habits?*  
# **Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data.

# **Answer:**
#  I attempt to predict detergents paper. Because intuitively, detergents paper is part of grocery demand. 
#  
#  The reported prediction score is 0.58.
#  
#  This feature is not necessary for identifying customer's spending habits. Because other features in the data has relation and are relevant to detergents paper quantities. The feature's information can be mostly represented by the remaining features. However it depends on the situation. The dropping detergents paper may lose some extent of information. If the amount of feature is limiting the resouces, it should be dropped. But if more complete information is required, the feature can be kept.

# ### Visualize Feature Distributions
# To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.

# In[33]:

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Question 3
# *Are there any pairs of features which exhibit some degree of correlation? Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? How is the data for those features distributed?*  
# **Hint:** Is the data normally distributed? Where do most of the data points lie? 

# **Answer:**
# 
# There are 3 pairs. Detergents paper and grocery have a strong relevance. Grocery and milk also have a decent relation. Therefore milk and detergent paper also have a weak relation.
# 
# Yes it does confirm my attempted feature relation prediction.
# 
# The data are not normally distributed from the observation of the figure above. The data mostly focus at the left lower corner. Logrithmic features scaling could be applied to find an another perspective to view the data.

# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.
# 
# In the code block below, you will need to implement the following:
#  - Assign a copy of the data to `log_data` after applying a logarithm scaling. Use the `np.log` function for this.
#  - Assign a copy of the sample data to `log_samples` after applying a logrithm scaling. Again, use `np.log`.

# In[35]:

# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Observation
# After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).
# 
# Run the code below to see how the sample data has changed after having the natural logarithm applied to it.

# In[36]:

# Display the log-transformed sample data
display(log_samples)


# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
#  - Assign the calculation of an outlier step for the given feature to `step`.
#  - Optionally remove data points from the dataset by adding indices to the `outliers` list.
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

# In[81]:

# For each feature find the data points with extreme high or low values
nSamples = len(log_data.index)
noOutlier = [True] * nSamples

for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3 - Q1) * 1.5
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
    # update no-outlier-labeling array
    t = log_data.index[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    for i in t :
        noOutlier[i] = False
    
# OPTIONAL: Select the indices for data points you wish to remove
outliers  = [i for i in range(nSamples) if not noOutlier[i]]
print "outliers: ", outliers

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
#print "good_data: \n", good_data


# ### Question 4
# *Are there any data points considered outliers for more than one feature? Should these data points be removed from the dataset? If any data points were added to the `outliers` list to be removed, explain why.* 

# **Answer:**
# Yes there are.
# 
# Yes any data points considered outliers for no less than one features should be removed from the dataset. Because the learning is going to evaluate all feature values in order to be trained correctly. For example customer No.154 is considered outlier for milk, grocery and delicatessen and therefore should not be considered to be qualified data for training. And there are many examples like No.154 that has multiple outlier features in the samples. They should be all eliminated. 

# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of the sample log-data `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[54]:

from sklearn.decomposition import PCA
# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components = 6)
pca.fit(good_data)
#print "PCA explained variance ratio = ", pca.explained_variance_ratio_

# TODO: Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = rs.pca_results(good_data, pca)


# ### Question 5
# *How much variance in the data is explained* ***in total*** *by the first and second principal component? What about the first four principal components? Using the visualization provided above, discuss what the first four dimensions best represent in terms of customer spending.*  
# **Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the indivdual feature weights.

# **Answer:**
# 
# 0.532 + 0.191 = 0.723; 0.723 variance in the data is exampled in total by first and second pricepal component (PC).
# 
# The first four dimensions in total 92.7% variance of data (by summing explained variance of first four dimensions). The portion of variance over all data is large enough to ensure trivial information loss. Therefore it is reasonable to let the first four dimensions represent.
# 
# Interpretation
# 
# Rule of thumb: A correlation value above 0.5 is deemed important.
# 
# - The 1st PC has 0.75 correlation with detergents paper. So the 1st PC is primarily a measure of detergents paper. Milk and grocery cost also has a small portion in 1st PC (0.4 and 0.45 respectively), indicating that milk and grocery cost will be highly possibly to vary with detergents paper.
# 
# - The 2nd PC has 0.55 fresh and 0.65 frozen. 2nd PC mainly measures these two features. 2nd PC increases with increase of fresh and frozen. A customer buys more frozen will be likely to buy fresh too.
# 
# - The 3rd PC has 0.7 frozen and -0.55 delicatessen. 3rd PC mainly measures frozen and delicatessen. And the 3rd increases with the increase of frozen and decrease of delicatessen.
# 
# - The 4rd PC has 0.75 fresh and -0.5 delicatessen. 3rd PC mainly measures fresh and delicatessen. And the 3rd increases with the increase of fresh and decrease of delicatessen.

# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.

# In[55]:

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementation: Dimensionality Reduction
# When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.
# 
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the reuslts to `reduced_data`.
#  - Apply a PCA transformation of the sample log-data `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[59]:

# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components = 2)
pca.fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.

# In[60]:

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# ## Clustering
# 
# In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

# ### Question 6
# *What are the advantages to using a K-Means clustering algorithm? What are the advantages to using a Gaussian Mixture Model clustering algorithm? Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?*

# **Answer:**
# Advantage of using a K-Mean is that it is a simple model. It assigns each data to exactly one cluster. It is fast and computationally cheap.
# 
# Advantage of using Gaussian Mixture Model (GMM):
# 
# 1. Softly cluster data. Data are assumed to have several Guassian-distribution clusters and each data point has probability to be assigned to one cluster.
# 2. The likelihood of data clustering decreases monotonically. It's well behaved, and will not diverge becasue it is probability driven.
# 3. With different initial clustering condision, GMM converges to results less variant than K-Mean
# 
# I would use GMM because in reality data is not hard-clustered and probalistical clustering is more practical. 

# ### Implementation: Creating Clusters
# Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.
# 
# In the code block below, you will need to implement the following:
#  - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
#  - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
#  - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
#  - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
#  - Import sklearn.metrics.silhouette_score and calculate the silhouette score of `reduced_data` against `preds`.
#    - Assign the silhouette score to `score` and print the result.

# In[76]:

# TODO: Apply your clustering algorithm of choice to the reduced data 
from sklearn import mixture
clusterer = mixture.GMM(n_components=2)
clusterer.fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)
#print preds

# TODO: Find the cluster centers
centers = clusterer.means_
print centers

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
from sklearn.metrics import silhouette_score
score = silhouette_score(reduced_data, preds)
print score


# ### Question 7
# *Report the silhouette score for several cluster numbers you tried. Of these, which number of clusters has the best silhouette score?* 

# **Answer:**
# the silhouette score = 0.46, with number of clusters = 2, which is the best sihouette score

# ### Cluster Visualization
# Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 

# In[77]:

# Display the results of the clustering from implementation
rs.cluster_results(reduced_data, preds, centers, pca_samples)


# ### Implementation: Data Recovery
# Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.
# 
# In the code block below, you will need to implement the following:
#  - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
#  - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.
# 

# In[78]:

# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)


# ### Question 8
# Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project. *What set of establishments could each of the customer segments represent?*  
# **Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`.

# **Answer:**
# 
# Segment 0 could represent a grocery. Because it has reasonable amount of demand on all things. Also an outstanding feature is it demands a large amount of milk, grocery and detergents paper, which further proves the establishment.
# 
# Segment 1 could represent a restaurant. It does not demand much milk, grocery or detergents paper, while has a large consumption on fresh.

# ### Question 9
# *For each sample point, which customer segment from* ***Question 8*** *best represents it? Are the predictions for each sample point consistent with this?*
# 
# Run the code block below to find which cluster each sample point is predicted to be.

# In[79]:

# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred


# **Answer:**
# 
# - Sample point 0 predicted to be in Cluster 0 (grocery)
# - Sample point 1 predicted to be in Cluster 0 (grocery)
# - Sample point 2 predicted to be in Cluster 1 (restaurant)
# 
# Yes the predictions for each sample point consistent with this. 
# - Sample No.0 has fresh, milk, grocery, frozen and detergents paper cost very close to segment 0 center point. Delicatessen is double of segment 0 center point (1223) while even farther than segment 1 (816), which gains larger possibility for sample No.0 to lie in segment 0.
# - Sample No.1 has milk, grocery and detergents paper (12697, 28540, 12034) way more than signment 0 signature cost (6794, 9529, 3316), while those cost of segment 1 is even lower (1962, 2464, 330). Therefore sample No.1 has more change to lie in segment 0 too. Sample No.1's delicatessen (1009) is also close to segment 0 center (1123) while farther to segment 1 center point (816). Fresh of sample no.1 is close to segment 1, but other features' similarity to segment 0 leads the sample to be clustered to segment 0.
# - Sample No. 2's evidence of being segment 1 is mainly its detergents paper cost (20) while segment 1 center has 330 and segment 0 has 3316. Since detergents paper is 1st principle component, the feature plays a very important role for it to be clustered.
# 

# ## Conclusion

# ### Question 10
# *Companies often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services. If the wholesale distributor wanted to change its delivery service from 5 days a week to 3 days a week, how would you use the structure of the data to help them decide on a group of customers to test?*  
# **Hint:** Would such a change in the delivery service affect all customers equally? How could the distributor identify who it affects the most?

# **Answer:**
# 
# Since we have figured out there are mainly two categories of customers: grocery and restaurant. We can test delivery service change for the two categories respectively to evaluate the effect on customers.
# 
# More exactly, for restaurant customers, the company should first try to change delivery service from 5 days/week to 3 days/week for half of restaurant customers. The restaurant customers withiout delivery service change is called group A. The other half with delivery service change is called group B. When group A and B are tested in the market, evaluate if group B is saving more money for chaning to a cheaper delivery service or group is losing money because large amount of restaurant customers does not satisfy with the service any more. By comparing test result A/B, practical decision can be made to whether change the delivery service or not, for restaurant customers. Then apply the similar A/B test strategy for grocery customers.

# ### Question 11
# *Assume the wholesale distributor wanted to predict a new feature for each customer based on the purchasing information available. How could the wholesale distributor use the structure of the clustering data you've found to assist a supervised learning analysis?*  
# **Hint:** What other input feature could the supervised learner use besides the six product features to help make a prediction?

# **Answer:**
# 
# For example the distributor want to predict a type (Chinese, Thai, Italian, American) of a restaurant based on purchasing information.
# 
# It can first perform a questionaire and collect restaurant type label based on the companies existing customer. Then specifically for restaurant-cluster data, apply the supervised learning analysis with collected labels. If the learner is well-trained, it will be able to predict. 

# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier on to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

# In[80]:

# Display the clustering results based on 'Channel' data
rs.channel_results(reduced_data, outliers, pca_samples)


# ### Question 12
# *How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? Would you consider these classifications as consistent with your previous definition of the customer segments?*

# **Answer:**
# 
# The clustering algorithm and number of clusters (which equals two) I've chosen fits the underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers very well. The cluster number is correctly identified to be two. The Guassian Mixture Model assumes guassian distribution for each cluster, which is practical in reality because in the figures it showes there is no clear clustering boundary. Data points close to one cluster center may still be a member of another cluster, with a lower possibility.
# 
# Customer segments are not classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution. I would consider these classifications as consistent with my previous definition of the customer segments. Because customers are probalistically clustered. Customers who are quite similar to one cluster may still be possible to belong to another cluster.  

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
