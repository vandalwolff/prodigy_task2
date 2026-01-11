Task-02: Customer Segmentation using K-Means Clustering
Objective
To group retail store customers into meaningful segments based on their purchase behavior using the K-Means clustering algorithm.
This helps businesses understand customer patterns and improve marketing strategies.

Dataset
Source: Kaggle – Customer Segmentation Tutorial in Python

File Used: Mall_Customers.csv

Features Used
Annual Income (k$)
Spending Score (1–100)

Approach
Load customer purchase dataset
Select relevant numerical features
Scale features using StandardScaler

Use the Elbow Method to find optimal number of clusters
Apply K-Means clustering
Visualize customer segments

Technologies Used

Python 3
pandas
scikit-learn
matplotlib

Project Structure
prodigy/
│── task2.py
│── Mall_Customers.csv
│── README.md
│── .venv/

How to Run
.\.venv\Scripts\activate
python task2.py

Output
Elbow Method graph
Cluster visualization plot
Dataset with assigned cluster labels

Conclusion

The K-Means algorithm successfully groups customers with similar purchasing behavior, enabling data-driven customer segmentation.
