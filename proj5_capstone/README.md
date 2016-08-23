# Degree Capstone Project

## Predicting Stock Market

### Project Requirement

In this capstone project, you will leverage what you’ve learned throughout the Nanodegree program to solve a problem of your choice by applying machine learning algorithms and techniques. You will first define the problem you want to solve and investigate potential solutions and performance metrics. Next, you will analyze the problem through visualizations and data exploration to have a better understanding of what algorithms and features are appropriate for solving it.
You will then implement your algorithms and metrics of choice, documenting the preprocessing, refinement, and postprocessing steps along the way. Afterwards, you will collect results about the performance of the models used, visualize significant quantities, and validate/justify these values. Finally, you will construct conclusions about your results, and discuss whether your implementation adequately solves the problem.
Suggested application areas include:
- Deep Learning
- Robot Motion Planning
- Healthcare
- Education
- Computer Vision
- Investment and Trading


### Capstone Project Highlights

This project is designed to prepare you for delivering a polished, end-to-end solution report of a real-world problem in a field of interest. When developing new technology, or deriving adaptations of previous technology, properly documenting your process is critical for both validating and replicating your results.

Things you will learn by completing this project:

- How to research and investigate a real-world problem of interest.
- How to accurately apply specific machine learning algorithms and techniques.
- How to properly analyze and visualize your data and results for validity.
- How to document and write a report of your work.


### Project Overview

Investment firms, hedge funds and even individuals have been using financial models to better understand market behavior and make profitable investments and trades. Predicting the stock price trend by interpreting the seemly chaotic market data has always been an attractive topic to both investors and researchers. Among those popular methods that have been employed, Machine Learning techniques are very popular due to the capacity of identifying stock trend from massive amounts of data that capture the underlying stock price dynamics. 

This project builds a stock price trend estimate tool for a given query date. The estimator leverages state-of-art machine learning techniques, and learns from large amounts of a wide variety of historic stock market data, which is not able to be achieved by human. It is going to provide a valuable prospective to predict stocks and assist stock investments. Further, it plays an essential role for constructing an automatic trading system.
Fortunately, stock market data can be easily accessed from a number of APIs or websites on the Internet. For instance, Google Finance, Bloomberg or Yahoo! Finance interface historic stock data containing multiple metrics, such as:

- Open: price of the stock at the opening of the trading.
- High: highest price of the stock during the trading day.
- Low: lowest price of the stock during the trading day.
- Volume: amount of stocks traded during the day.


### Problem Statement

Specifically, the stock trend estimator in this project is defined as:

- Input: daily trading data over a certain date range.
- Query: the date and the company symbol of which the estimator is required to predict the stock trend.
- Output: estimate of stock adjusted close price trend with confidence value (rise or fall). 
- Methodology: supervised learning.

The following problems are to be solved:
- Stock data acquisition and selection. There are numerous data on the Internet. It is significant to select important datasets that affect the prediction most and get cleaned and formatted data so that program is able to process.
- Feature generation. Features dimension can easily reach the scale of thousands. Generating a limited number of features that successfully capture most information and stock dynamics to feed the machine-learning model is the key of this project.
- Classification model training and comparison. Metrics should be selected to effectively compare performance of different algorithms. Appropriate parameters should be searched to get best training result and prevent over-fitting.
- User interface design. It is important to land the design and make it handy for users and provide valuable investment suggestions for them.


### Deliverables

- Project report in `./Capstone Project Report.pdf`
- Ipython notebook model creation in `./Stock_Market.ipynb` and `./stock_market.html`


