





















































# 1. Perform the following operations using Python on a data set : read data from different formats(like csv, xls),indexing and selecting data, sort data, describe attributes of data, checking data types of each column. (Use Titanic Dataset)

# -->
# import pandas as pd

# # 1. Read data from CSV
# df = pd.read_csv("titanic.csv")   # replace with your file name/path

# # 2. (Theory/demo) Read from Excel
# df_excel = pd.read_excel("titanic.xlsx", sheet_name="Sheet1")

# # 3. View first few records
# print("First 5 rows:")
# print(df.head())

# # 4. Indexing and Selecting data
# print("\nSingle column - Age:")
# print(df["Age"].head())

# print("\nMultiple columns - Name, Sex, Age:")
# print(df[["Name", "Sex", "Age"]].head())

# print("\nFirst 5 rows using iloc:")
# print(df.iloc[0:5, :])

# print("\nPassengers older than 50:")
# print(df.loc[df["Age"] > 50].head())

# # 5. Sorting data
# print("\nSorted by Age:")
# print(df.sort_values(by="Age")[["Name", "Age"]].head())

# print("\nSorted by Fare (descending):")
# print(df.sort_values(by="Fare", ascending=False)[["Name", "Fare"]].head())

# # 6. Describe attributes
# print("\nSummary statistics (numeric):")
# print(df.describe())

# print("\nSummary statistics (all columns):")
# print(df.describe(include="all"))

# # 7. Data types of each column
# print("\nData types:")
# print(df.dtypes)

# print("\nDetailed info:")
# df.info()

# 2. Perform the following operations using Python on the Telecom_Churn dataset. Compute and display summary statistics for each feature available in the dataset using separate commands for each statistic. (e.g. minimum value, maximum value, mean, range, standard deviation, variance and percentiles)

# # -----------------------------------------------
# # Q2: Summary Statistics on Telecom Churn Dataset
# # -----------------------------------------------

# import pandas as pd

# # Step 1: Load the dataset
# path = "/mnt/data/891ac7d1-bcf2-4475-b776-9f9f7dcb69e4.csv"
# df = pd.read_csv(path)

# # Step 2: Select only numeric columns (exclude boolean columns)
# numeric_cols = df.select_dtypes(include=["number"]).columns

# # Step 3: Compute summary statistics using separate commands

# # (a) Minimum values
# print("Minimum values:\n", df[numeric_cols].min(), "\n")

# # (b) Maximum values
# print("Maximum values:\n", df[numeric_cols].max(), "\n")

# # (c) Mean values
# print("Mean values:\n", df[numeric_cols].mean(), "\n")

# # (d) Range (Max - Min)
# print("Range:\n", df[numeric_cols].max() - df[numeric_cols].min(), "\n")

# # (e) Standard deviation
# print("Standard deviation:\n", df[numeric_cols].std(), "\n")

# # (f) Variance
# print("Variance:\n", df[numeric_cols].var(), "\n")

# # (g) Percentiles (25th, 50th, 75th)
# print("25th percentile:\n", df[numeric_cols].quantile(0.25), "\n")
# print("50th percentile (Median):\n", df[numeric_cols].quantile(0.5), "\n")
# print("75th percentile:\n", df[numeric_cols].quantile(0.75), "\n")

# # Step 4: Create a combined summary table
# summary = pd.DataFrame({
#     "Min": df[numeric_cols].min(),
#     "Max": df[numeric_cols].max(),
#     "Mean": df[numeric_cols].mean(),
#     "Range": df[numeric_cols].max() - df[numeric_cols].min(),
#     "Std Dev": df[numeric_cols].std(),
#     "Variance": df[numeric_cols].var(),
#     "25%": df[numeric_cols].quantile(0.25),
#     "50% (Median)": df[numeric_cols].quantile(0.5),
#     "75%": df[numeric_cols].quantile(0.75)
# })

# print("\n===== Summary Statistics for Telecom Churn Dataset =====\n")
# print(summary)

# # Step 5: Display column data types
# print("\nData types of all columns:\n", df.dtypes)

# 3. Perform the following operations using Python on the data set House_Price Prediction dataset. Compute standard deviation, variance and percentiles using separate commands, for each feature. Create a histogram for each feature in the dataset to illustrate the feature distributions
# # -------------------------------------------------------
# # Q3: Summary Stats + Histograms on House_Price dataset
# # Dataset file: "House Data.csv"
# # -------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt

# # 1. Load the dataset
# # If your file is in the same folder as your .py/.ipynb, this is enough:
# # df = pd.read_csv("House Data.csv")

# # For this environment (as given in the question bank):
# df = pd.read_csv("House Data.csv")   # <-- change path if needed

# # 2. (Optional but useful) – Show first few rows and column info
# print("First 5 rows of dataset:\n", df.head(), "\n")
# print("Column data types:\n", df.dtypes, "\n")

# # 3. Select only numeric columns (features on which stats make sense)
# numeric_cols = df.select_dtypes(include=["number"]).columns
# print("Numeric feature columns used for statistics:\n", numeric_cols, "\n")

# # 4. STANDARD DEVIATION for each numeric feature
# print("Standard Deviation for each numeric feature:\n")
# std_values = df[numeric_cols].std()
# print(std_values, "\n")

# # 5. VARIANCE for each numeric feature
# print("Variance for each numeric feature:\n")
# var_values = df[numeric_cols].var()
# print(var_values, "\n")

# # 6. PERCENTILES for each numeric feature
# # 25th percentile (Q1)
# print("25th Percentile (Q1) for each numeric feature:\n")
# q25 = df[numeric_cols].quantile(0.25)
# print(q25, "\n")

# # 50th percentile (Median, Q2)
# print("50th Percentile (Median, Q2) for each numeric feature:\n")
# q50 = df[numeric_cols].quantile(0.50)
# print(q50, "\n")

# # 75th percentile (Q3)
# print("75th Percentile (Q3) for each numeric feature:\n")
# q75 = df[numeric_cols].quantile(0.75)
# print(q75, "\n")

# # (Optional) – Combined summary table for your record
# summary = pd.DataFrame({
#     "Std Dev": std_values,
#     "Variance": var_values,
#     "25%": q25,
#     "50% (Median)": q50,
#     "75%": q75
# })

# print("===== Summary table for numeric features =====\n")
# print(summary)

# # 7. HISTOGRAMS for each numeric feature
# # This will open one histogram per numeric column

# for col in numeric_cols:
#     plt.figure()                     # new figure for each feature
#     df[col].hist()                   # histogram for the feature
#     plt.title(f"Histogram of {col}")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.tight_layout()
#     plt.show()
# 4. Write a program to do: A dataset collected in a cosmetics shop showing details of customers and whether or not they responded to a special offer to buy a new lip-stick is shown in table below. (Implement step by step using commands - Dont use library) Use this dataset to build a decision tree, with Buys as the target variable, to help in buying lipsticks in the future. Find the root node of the decision tree.
# # ---------------------------------------------
# # Q4: Decision Tree Root Node for Lipstick Data
# # No ML libraries used (manual entropy & IG)
# # ---------------------------------------------

# import csv
# import math

# # ----- 1. Load dataset from CSV -----
# filename = "Lipstick.csv"   # change path if needed

# data = []
# with open(filename, newline='', encoding="utf-8") as f:
#     reader = csv.reader(f)
#     header = next(reader)           # ['Id','Age','Income','Gender','Ms','Buys']
#     # Fix BOM in first header if present
#     if header[0].startswith("\ufeff"):
#         header[0] = header[0].replace("\ufeff", "")
#     for row in reader:
#         record = dict(zip(header, row))
#         data.append(record)

# print("First few records:")
# for d in data[:5]:
#     print(d)
# print()

# # Target and attributes
# target_attr = "Buys"
# attributes = ["Age", "Income", "Gender", "Ms"]  # Ignore Id

# # ----- 2. Helper: compute entropy of a dataset -----
# def entropy(dataset):
#     total = len(dataset)
#     if total == 0:
#         return 0.0
    
#     # Count Yes/No in Buys
#     counts = {}
#     for row in dataset:
#         label = row[target_attr]
#         counts[label] = counts.get(label, 0) + 1
    
#     ent = 0.0
#     for count in counts.values():
#         p = count / total
#         ent -= p * math.log2(p)
#     return ent

# # ----- 3. Helper: compute information gain of an attribute -----
# def information_gain(dataset, attr):
#     total = len(dataset)
#     base_entropy = entropy(dataset)
    
#     # Partition dataset based on attribute values
#     partitions = {}   # value -> list of rows
#     for row in dataset:
#         key = row[attr]
#         partitions.setdefault(key, []).append(row)
    
#     # Expected entropy after split
#     expected_entropy = 0.0
#     for subset in partitions.values():
#         weight = len(subset) / total
#         expected_entropy += weight * entropy(subset)
    
#     gain = base_entropy - expected_entropy
#     return gain, partitions, expected_entropy

# # ----- 4. Compute base entropy -----
# base_ent = entropy(data)
# print(f"Overall entropy of Buys: {base_ent:.4f}\n")

# # ----- 5. Compute information gain for each attribute -----
# best_attr = None
# best_gain = -1

# for attr in attributes:
#     gain, parts, exp_ent = information_gain(data, attr)
#     print(f"Attribute: {attr}")
#     print(f"  Information Gain = {gain:.6f}")
#     print(f"  Expected Entropy after split = {exp_ent:.6f}")
#     print("  Partitions:")
#     for val, subset in parts.items():
#         # Count Yes/No in each subset
#         yes = sum(1 for r in subset if r[target_attr] == "Yes")
#         no = sum(1 for r in subset if r[target_attr] == "No")
#         print(f"    {attr} = {val}: {len(subset)} records  (Yes={yes}, No={no})")
#     print()

#     if gain > best_gain:
#         best_gain = gain
#         best_attr = attr

# # ----- 6. Print root node of decision tree -----
# print("========================================")
# print(f"Root node of the Decision Tree: {best_attr}")
# print(f"Information Gain at root: {best_gain:.6f}")
# print("========================================")
# 5. Write a program to do: A dataset collected in a cosmetics shop showing details of customers and whether or not they responded to a special offer to buy a new lip-stick is shown in table below. (Use library commands) According to the decision tree you have made from the previous training data set, what is the decision for the test data: [Age < 21, Income = Low, Gender = Female, Marital Status = Married]?
# # ---------------------------------------------------------------
# # Q5: Decision Tree using Library (scikit-learn) for Lipstick Data
# # ---------------------------------------------------------------

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier, export_text

# # 1️⃣ Load dataset
# path = "Lipstick.csv"      # change to your actual path if needed
# df = pd.read_csv(path)

# # Fix column name if it contains BOM
# if df.columns[0].startswith("\ufeff"):
#     df = df.rename(columns={df.columns[0]: "Id"})

# # 2️⃣ Select relevant features and target
# X = df[["Age", "Income", "Gender", "Ms"]].copy()   # copy() avoids warnings
# y = df["Buys"]

# # 3️⃣ Convert categorical values to numeric using LabelEncoder
# encoders = {}   # store separate encoder for each column

# for col in X.columns:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])
#     encoders[col] = le          # save encoder for later (test data)

# target_le = LabelEncoder()
# y_enc = target_le.fit_transform(y)   # Yes/No -> 1/0

# # 4️⃣ Train Decision Tree model
# clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
# clf.fit(X, y_enc)

# # 5️⃣ Visualize tree structure (text form)
# tree_rules = export_text(clf, feature_names=list(X.columns))
# print("Decision Tree Rules:\n", tree_rules)

# # 6️⃣ Prepare test data as given in question
# test_data = pd.DataFrame({
#     "Age": ["<21"],
#     "Income": ["Low"],
#     "Gender": ["Female"],
#     "Ms": ["Married"]
# })

# # 7️⃣ Apply SAME encoders used for training
# for col in test_data.columns:
#     test_data[col] = encoders[col].transform(test_data[col])

# # 8️⃣ Predict the result for test data
# prediction_enc = clf.predict(test_data)[0]
# prediction_label = target_le.inverse_transform([prediction_enc])[0]

# # 9️⃣ Display the prediction
# print("\nTest Data: [Age <21, Income=Low, Gender=Female, Ms=Married]")
# print(f"Decision (Buys Lipstick?): {prediction_label}")
# 6. Write a program to do: A dataset collected in a cosmetics shop showing details of customers and whether or not they responded to a special offer to buy a new lip-stick is shown in table below. (Use library commands) According to the decision tree you have made from the previous training data set, what is the decision for the test data: [Age > 35, Income = Medium, Gender = Female, Marital Status = Married]?
# # ---------------------------------------------------------------
# # Q6: Decision Tree using Library (scikit-learn) for Lipstick Data
# #      Test: [Age >35, Income=Medium, Gender=Female, Ms=Married]
# # ---------------------------------------------------------------

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier, export_text

# # 1️⃣ Load dataset
# path = "Lipstick.csv"      # change path if needed
# df = pd.read_csv(path)

# # Fix column name if it contains BOM
# if df.columns[0].startswith("\ufeff"):
#     df = df.rename(columns={df.columns[0]: "Id"})

# # 2️⃣ Select relevant features and target
# X = df[["Age", "Income", "Gender", "Ms"]].copy()
# y = df["Buys"]

# # 3️⃣ Encode categorical features and target using LabelEncoder
# encoders = {}

# for col in X.columns:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])
#     encoders[col] = le        # store encoder for each column

# target_le = LabelEncoder()
# y_enc = target_le.fit_transform(y)   # Yes / No → numeric

# # 4️⃣ Train Decision Tree model
# clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
# clf.fit(X, y_enc)

# # (Optional) View tree rules
# tree_rules = export_text(clf, feature_names=list(X.columns))
# print("Decision Tree Rules:\n", tree_rules)

# # 5️⃣ Prepare test data for Q6
# test_data = pd.DataFrame({
#     "Age": [">35"],
#     "Income": ["Medium"],
#     "Gender": ["Female"],
#     "Ms": ["Married"]
# })

# # 6️⃣ Encode test data using SAME encoders
# for col in test_data.columns:
#     test_data[col] = encoders[col].transform(test_data[col])

# # 7️⃣ Predict Buys / Not Buys
# prediction_enc = clf.predict(test_data)[0]
# prediction_label = target_le.inverse_transform([prediction_enc])[0]

# # 8️⃣ Print final decision
# print("\nTest Data: [Age >35, Income=Medium, Gender=Female, Ms=Married]")
# print(f"Decision (Buys Lipstick?): {prediction_label}")
# 7. Write a program to do: A dataset collected in a cosmetics shop showing details of customers and whether or not they responded to a special offer to buy a new lip-stick is shown in table below. (Use library commands) According to the decision tree you have made from the previous training data set, what is the decision for the test data: [Age > 35, Income = Medium, Gender = Female, Marital Status = Married]?
# # ---------------------------------------------------------------
# # Q7: Decision Tree using Library (scikit-learn) for Lipstick Data
# #     Test: [Age >35, Income=Medium, Gender=Female, Ms=Married]
# # ---------------------------------------------------------------

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier, export_text

# # 1️⃣ Load dataset
# path = "Lipstick.csv"   # change path if required
# df = pd.read_csv(path)

# # Fix BOM in first column name if present
# if df.columns[0].startswith("\ufeff"):
#     df = df.rename(columns={df.columns[0]: "Id"})

# # 2️⃣ Select feature columns (inputs) and target column (output)
# X = df[["Age", "Income", "Gender", "Ms"]].copy()
# y = df["Buys"]

# # 3️⃣ Encode all categorical columns using LabelEncoder
# encoders = {}   # store encoders for each feature

# for col in X.columns:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])
#     encoders[col] = le          # store encoder for later use (test data)

# # Encode target (Buys)
# target_le = LabelEncoder()
# y_enc = target_le.fit_transform(y)   # "No"/"Yes" -> 0/1

# # 4️⃣ Train Decision Tree model (using entropy criterion)
# clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
# clf.fit(X, y_enc)

# # (Optional) print tree rules for understanding
# tree_rules = export_text(clf, feature_names=list(X.columns))
# print("Decision Tree Rules:\n", tree_rules)

# # 5️⃣ Prepare the test data given in the question
# #    [Age > 35, Income = Medium, Gender = Female, Ms = Married]
# test_data = pd.DataFrame({
#     "Age": [">35"],
#     "Income": ["Medium"],
#     "Gender": ["Female"],
#     "Ms": ["Married"]
# })

# # 6️⃣ Encode test data using SAME encoders as training
# for col in test_data.columns:
#     test_data[col] = encoders[col].transform(test_data[col])

# # 7️⃣ Predict using the trained model
# predicted_numeric = clf.predict(test_data)[0]
# predicted_label = target_le.inverse_transform([predicted_numeric])[0]

# # 8️⃣ Final output
# print("\nTest Data: [Age >35, Income=Medium, Gender=Female, Ms=Married]")
# print(f"Decision (Buys Lipstick?): {predicted_label}")
# 8. Write a program to do: A dataset collected in a cosmetics shop showing details of customers and whether or not they responded to a special offer to buy a new lip-stick is shown in table below. (Use library commands) According to the decision tree you have made from the previous training data set, what is the decision for the test data: [Age = 21-35, Income = Low, Gender = Male, Marital Status = Married]?
# # ---------------------------------------------------------------
# # Q8: Decision Tree using Library (scikit-learn) for Lipstick Data
# #     Test: [Age = 21-35, Income = Low, Gender = Male, Ms = Married]
# # ---------------------------------------------------------------

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier, export_text

# # 1️⃣ Load dataset
# path = "Lipstick.csv"   # change path if needed
# df = pd.read_csv(path)

# # Fix column name if it contains BOM
# if df.columns[0].startswith("\ufeff"):
#     df = df.rename(columns={df.columns[0]: "Id"})

# # 2️⃣ Select relevant features and target
# X = df[["Age", "Income", "Gender", "Ms"]].copy()
# y = df["Buys"]

# # 3️⃣ Encode categorical values using LabelEncoder
# encoders = {}

# for col in X.columns:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])
#     encoders[col] = le  # save encoder for later use (test data)

# # Encode target column
# target_le = LabelEncoder()
# y_enc = target_le.fit_transform(y)

# # 4️⃣ Train the Decision Tree Classifier
# clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
# clf.fit(X, y_enc)

# # (Optional) View tree structure for understanding
# tree_rules = export_text(clf, feature_names=list(X.columns))
# print("Decision Tree Rules:\n", tree_rules)

# # 5️⃣ Prepare test data for question
# test_data = pd.DataFrame({
#     "Age": ["21-35"],
#     "Income": ["Low"],
#     "Gender": ["Male"],
#     "Ms": ["Married"]
# })

# # 6️⃣ Encode test data using SAME encoders
# for col in test_data.columns:
#     test_data[col] = encoders[col].transform(test_data[col])

# # 7️⃣ Predict the result
# prediction_num = clf.predict(test_data)[0]
# prediction_label = target_le.inverse_transform([prediction_num])[0]

# # 8️⃣ Print final decision
# print("\nTest Data: [Age=21-35, Income=Low, Gender=Male, Ms=Married]")
# print(f"Decision (Buys Lipstick?): {prediction_label}")
# 9. Write a program to do the following: You have given a collection of 8 points. P1=[0.1,0.6] P2=[0.15,0.71] P3=[0.08,0.9] P4=[0.16, 0.85] P5=[0.2,0.3] P6=[0.25,0.5] P7=[0.24,0.1] P8=[0.3,0.2]. Perform the k-mean clustering with initial centroids as m1=P1 =Cluster#1=C1 and m2=P8=cluster#2=C2. Answer the following 1] Which cluster does P6 belong to? 2] What is the population of a cluster around m2? 3] What is the updated value of m1 and m2?
# # ----------------------------------------------------
# # Q9: One iteration of K-Means with k=2 on 8 points
# # Initial centroids: m1 = P1, m2 = P8
# # ----------------------------------------------------

# import math

# # 1. Define points
# points = {
#     "P1": (0.1, 0.6),
#     "P2": (0.15, 0.71),
#     "P3": (0.08, 0.9),
#     "P4": (0.16, 0.85),
#     "P5": (0.2, 0.3),
#     "P6": (0.25, 0.5),
#     "P7": (0.24, 0.1),
#     "P8": (0.3, 0.2),
# }

# # 2. Initial centroids
# m1 = points["P1"]  # (0.1, 0.6)
# m2 = points["P8"]  # (0.3, 0.2)

# def dist(a, b):
#     return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# # 3. Assign each point to nearest cluster
# C1 = []   # cluster around m1
# C2 = []   # cluster around m2

# print("Distances and cluster assignment:")
# for name, p in points.items():
#     d1 = dist(p, m1)
#     d2 = dist(p, m2)
#     if d1 <= d2:
#         C1.append(name)
#         assigned = "C1"
#     else:
#         C2.append(name)
#         assigned = "C2"
#     print(f"{name}: dist to m1={d1:.4f}, dist to m2={d2:.4f} -> {assigned}")

# print("\nCluster C1 (around m1):", C1)
# print("Cluster C2 (around m2):", C2)

# # 4. Update centroids m1 and m2 as mean of their clusters
# def mean_point(cluster):
#     xs = [points[p][0] for p in cluster]
#     ys = [points[p][1] for p in cluster]
#     return (sum(xs) / len(xs), sum(ys) / len(ys))

# new_m1 = mean_point(C1)
# new_m2 = mean_point(C2)

# print("\nUpdated centroid m1:", new_m1)
# print("Updated centroid m2:", new_m2)

# # 5. Specific answers
# print("\nAnswers:")
# print("1) P6 belongs to:", "C1" if "P6" in C1 else "C2")
# print("2) Population of cluster around m2 (C2):", len(C2))
# print("3) Updated m1 =", new_m1, ", Updated m2 =", new_m2)


# ✅ Final concise answers (for your writeup)
# 	P6 belongs to: Cluster C1
# 	Population of cluster around m2 (C2): 3 points
# 	Updated centroids:
# 	m1=(0.148,0.712)
# 	m2=(0.2467,0.2)

# 10. Write a program to do the following: You have given a collection of 8 points. P1=[2, 10] P2=[2, 5] P3=[8, 4] P4=[5, 8] P5=[7,5] P6=[6, 4] P7=[1, 2] P8=[4, 9]. Perform the k-mean clustering with initial centroids as m1=P1 =Cluster#1=C1 and m2=P4=cluster#2=C2, m3=P7 =Cluster#3=C3. Answer the following 1] Which cluster does P6 belong to? 2] What is the population of a cluster around m3? 3] What is the updated value of m1, m2, m3?
# # ------------------------------------------------------------
# # Q10: One iteration of K-Means with k=3 on 8 points
# # Initial centroids:
# #   m1 = P1 = (2,10)
# #   m2 = P4 = (5,8)
# #   m3 = P7 = (1,2)
# # ------------------------------------------------------------

# import math

# # 1. Define all points
# points = {
#     "P1": (2, 10),
#     "P2": (2, 5),
#     "P3": (8, 4),
#     "P4": (5, 8),
#     "P5": (7, 5),
#     "P6": (6, 4),
#     "P7": (1, 2),
#     "P8": (4, 9)
# }

# # 2. Initial centroids
# m1 = points["P1"]  # (2,10)
# m2 = points["P4"]  # (5,8)
# m3 = points["P7"]  # (1,2)

# def dist(a, b):
#     return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# # 3. Assign points to nearest centroid
# C1, C2, C3 = [], [], []

# print("Distances and assignments:")
# for name, p in points.items():
#     d1 = dist(p, m1)
#     d2 = dist(p, m2)
#     d3 = dist(p, m3)

#     # Find nearest centroid
#     min_d = min(d1, d2, d3)
#     if min_d == d1:
#         C1.append(name)
#         cluster = "C1"
#     elif min_d == d2:
#         C2.append(name)
#         cluster = "C2"
#     else:
#         C3.append(name)
#         cluster = "C3"

#     print(f"{name} {p}: d(m1)={d1:.3f}, d(m2)={d2:.3f}, d(m3)={d3:.3f} -> {cluster}")

# print("\nCluster C1 around m1:", C1)
# print("Cluster C2 around m2:", C2)
# print("Cluster C3 around m3:", C3)

# # 4. Update centroids as mean of points in each cluster
# def mean_point(cluster_names):
#     xs = [points[p][0] for p in cluster_names]
#     ys = [points[p][1] for p in cluster_names]
#     return (sum(xs) / len(xs), sum(ys) / len(ys))

# new_m1 = mean_point(C1)
# new_m2 = mean_point(C2)
# new_m3 = mean_point(C3)

# print("\nUpdated centroids:")
# print("m1' =", new_m1)
# print("m2' =", new_m2)
# print("m3' =", new_m3)

# # 5. Specific answers
# print("\nAnswers:")
# print("1) P6 belongs to:", "C1" if "P6" in C1 else ("C2" if "P6" in C2 else "C3"))
# print("2) Population of cluster around m3 (C3):", len(C3))
# print("3) Updated m1, m2, m3:", new_m1, new_m2, new_m3)
# 11. Use Iris flower dataset and perform following : 1. List down the features and their types (e.g., numeric, nominal) available in the dataset. 2. Create a histogram for each feature in the dataset to illustrate the feature distributions. 
# # -------------------------------------------------------------
# # Q11: Iris Dataset Feature Types & Histograms (Using CSV file)
# # -------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 1️⃣ Load the dataset
# path = "IRIS.csv"   # change path if needed
# df = pd.read_csv(path)

# # 2️⃣ Display first few rows
# print("First 5 records of the dataset:\n")
# print(df.head(), "\n")

# # 3️⃣ List all features and their data types
# print("Features and their types:\n")
# for col in df.columns:
#     # If dtype is float or int -> Numeric, else Nominal
#     if df[col].dtype in ["float64", "int64"]:
#         dtype = "Numeric"
#     else:
#         dtype = "Nominal (Categorical)"
#     print(f"{col:25s} → {dtype}")

# # 4️⃣ Create histograms for each numeric feature
# numeric_features = df.select_dtypes(include=["float64", "int64"]).columns

# df[numeric_features].hist(
#     figsize=(10, 8), bins=10, color='lightblue', edgecolor='black'
# )
# plt.suptitle("Histograms for Each Feature (IRIS Dataset)", fontsize=14)
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

# # 5️⃣ (Optional - For better visualization)
# for feature in numeric_features:
#     plt.figure(figsize=(6,4))
#     sns.histplot(data=df, x=feature, kde=True, color="skyblue")
#     plt.title(f"Distribution of {feature}")
#     plt.xlabel(feature)
#     plt.ylabel("Frequency")
#     plt.tight_layout()
#     plt.show()
# 12.Use Iris flower dataset and perform following : 1. Create a box plot for each feature in the dataset. 2. Identify and discuss distributions and identify outliers from them. 
# # -------------------------------------------------------------
# # Q12: Box Plot & Outlier Detection using Iris Dataset (CSV file)
# # -------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 1️⃣ Load the dataset
# path = "IRIS.csv"  # change path if needed
# df = pd.read_csv(path)

# # 2️⃣ Display first few records
# print("First 5 records of the Iris dataset:\n")
# print(df.head(), "\n")

# # 3️⃣ Identify numeric features
# numeric_features = df.select_dtypes(include=["float64", "int64"]).columns
# print("Numeric features found in the dataset:\n", list(numeric_features), "\n")

# # 4️⃣ Create box plots for each numeric feature
# plt.figure(figsize=(10, 6))
# df.boxplot(column=list(numeric_features))
# plt.title("Boxplots of Numeric Features in Iris Dataset", fontsize=14)
# plt.ylabel("Centimeters (cm)")
# plt.show()

# # 5️⃣ Create individual box plots with Seaborn for better visuals
# for feature in numeric_features:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(data=df, x=feature, color="lightblue")
#     plt.title(f"Boxplot of {feature}")
#     plt.xlabel(feature)
#     plt.tight_layout()
#     plt.show()

# # 6️⃣ Outlier detection using IQR method
# print("Outlier detection using IQR (Interquartile Range):\n")

# for col in numeric_features:
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

#     print(f"{col}:")
#     print(f"  Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
#     print(f"  Lower bound = {lower_bound:.2f}, Upper bound = {upper_bound:.2f}")
#     if outliers.empty:
#         print("  ✅ No outliers detected.")
#     else:
#         print(f"  ⚠️ Outliers detected at values: {list(outliers.values)}")
#     print()
# 13. Use the covid_vaccine_statewise.csv dataset and perform the following analytics. a. Describe the dataset b. Number of persons state wise vaccinated for first dose in India c. Number of persons state wise vaccinated for second dose in India 
# import pandas as pd

# # Load the dataset
# path = "/mnt/data/Covid Vaccine Statewise.csv"
# df = pd.read_csv(path)

# # a) Describe the dataset
# print("First 5 records of the dataset:\n", df.head(), "\n")
# print("Dataset Info:\n")
# print(df.info(), "\n")
# print("Statistical Summary of Numeric Columns:\n", df.describe(), "\n")

# # b) Number of persons state wise vaccinated for first dose in India
# if 'First Dose Administered' in df.columns:
#     first_dose = df.groupby("State")["First Dose Administered"].max().sort_values(ascending=False)
#     print("Number of persons state-wise vaccinated for FIRST DOSE:\n", first_dose, "\n")
# else:
#     print("Column 'First Dose Administered' not found in dataset.\n")

# # c) Number of persons state wise vaccinated for second dose in India
# if 'Second Dose Administered' in df.columns:
#     second_dose = df.groupby("State")["Second Dose Administered"].max().sort_values(ascending=False)
#     print("Number of persons state-wise vaccinated for SECOND DOSE:\n", second_dose, "\n")
# else:
#     print("Column 'Second Dose Administered' not found in dataset.\n")

# # Combine summary for both doses
# if 'First Dose Administered' in df.columns and 'Second Dose Administered' in df.columns:
#     summary = pd.DataFrame({
#         "First Dose": first_dose,
#         "Second Dose": second_dose
#     })
#     print("State-wise Vaccination Summary:\n", summary)
# 14. Use the covid_vaccine_statewise.csv dataset and perform the following analytics. A. Describe the dataset. B. Number of Males vaccinated C.. Number of females vaccinated 
# import pandas as pd

# # Load dataset
# path = "/mnt/data/Covid Vaccine Statewise.csv"
# df = pd.read_csv(path)

# # A) Describe the dataset
# print("First 5 records:\n", df.head(), "\n")
# print("Dataset Info:\n")
# print(df.info(), "\n")
# print("Statistical Summary:\n", df.describe(), "\n")

# # B) Number of Males vaccinated (state-wise maximum to get cumulative count)
# if 'Male(Individuals Vaccinated)' in df.columns:
#     male_vaccinated = df.groupby("State")["Male(Individuals Vaccinated)"].max().sort_values(ascending=False)
#     print("Number of males vaccinated (state-wise):\n", male_vaccinated, "\n")
# else:
#     print("⚠️ Column 'Male(Individuals Vaccinated)' not found. Please check dataset.\n")

# # C) Number of Females vaccinated (state-wise maximum to get cumulative count)
# if 'Female(Individuals Vaccinated)' in df.columns:
#     female_vaccinated = df.groupby("State")["Female(Individuals Vaccinated)"].max().sort_values(ascending=False)
#     print("Number of females vaccinated (state-wise):\n", female_vaccinated, "\n")
# else:
#     print("⚠️ Column 'Female(Individuals Vaccinated)' not found. Please check dataset.\n")

# # Combined summary
# if 'Male(Individuals Vaccinated)' in df.columns and 'Female(Individuals Vaccinated)' in df.columns:
#     summary = pd.DataFrame({
#         "Males Vaccinated": male_vaccinated,
#         "Females Vaccinated": female_vaccinated
#     })
#     print("State-wise Summary of Male & Female Vaccinations:\n", summary)

# 15. Use the dataset 'titanic'. The dataset contains 891 rows and contains information about the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to see if we can find any patterns in the data.
# # ---------------------------------------------------------------
# # Q15: Titanic Dataset Analysis using Seaborn Visualizations
# # ---------------------------------------------------------------

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 1️⃣ Load the dataset
# path = "Titanic.csv"   # update path if needed
# df = pd.read_csv(path)

# # 2️⃣ Display basic dataset info
# print("First 5 records of Titanic dataset:\n")
# print(df.head(), "\n")

# print("Dataset Info:\n")
# print(df.info(), "\n")

# print("Statistical Summary:\n")
# print(df.describe(), "\n")

# # 3️⃣ Clean or check missing values
# print("Missing values in each column:\n", df.isnull().sum(), "\n")

# # 4️⃣ Basic Overview using Seaborn pairplot (optional)
# # sns.pairplot(df, hue='Survived')  # uncomment if you want full multi-feature plot
# # plt.show()

# # 5️⃣ Survival count visualization
# plt.figure(figsize=(6,4))
# sns.countplot(x='Survived', data=df, palette='Set2')
# plt.title("Overall Survival Count (0 = Died, 1 = Survived)")
# plt.show()

# # 6️⃣ Survival by Gender
# plt.figure(figsize=(6,4))
# sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1')
# plt.title("Survival Rate by Gender")
# plt.show()

# # 7️⃣ Survival by Passenger Class
# plt.figure(figsize=(6,4))
# sns.countplot(x='Pclass', hue='Survived', data=df, palette='coolwarm')
# plt.title("Survival by Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)")
# plt.show()

# # 8️⃣ Age distribution by Survival
# plt.figure(figsize=(8,5))
# sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=20, palette='husl')
# plt.title("Age Distribution by Survival")
# plt.show()

# # 9️⃣ Fare vs Class Visualization
# plt.figure(figsize=(7,5))
# sns.boxplot(x='Pclass', y='Fare', data=df, palette='pastel')
# plt.title("Fare Distribution by Passenger Class")
# plt.show()

# # 🔟 Survival probability by Embarkation Port
# if 'Embarked' in df.columns:
#     plt.figure(figsize=(6,4))
#     sns.countplot(x='Embarked', hue='Survived', data=df, palette='muted')
#     plt.title("Survival by Port of Embarkation")
#     plt.show()

# # 11️⃣ Correlation Heatmap for Numeric Features
# plt.figure(figsize=(8,6))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Heatmap for Numeric Features")
# plt.show()

# 16. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information about the passengers who boarded the unfortunate Titanic ship. Write a code to check how the price of the ticket (column name: 'fare') for each passenger is distributed by plotting a histogram. 
# # ---------------------------------------------------------------
# # Q16: Titanic Dataset - Fare Distribution Histogram
# # ---------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 1️⃣ Load the dataset
# path = "Titanic.csv"   # update path if required
# df = pd.read_csv(path)

# # 2️⃣ Display dataset information
# print("First 5 rows of dataset:\n")
# print(df.head(), "\n")

# print("Dataset Info:\n")
# print(df.info(), "\n")

# # 3️⃣ Check for missing values in 'Fare'
# print("Missing values in Fare column:", df['Fare'].isnull().sum(), "\n")

# # 4️⃣ Basic description of 'Fare'
# print("Statistical summary of Fare column:\n", df['Fare'].describe(), "\n")

# # 5️⃣ Plot histogram for Fare distribution
# plt.figure(figsize=(8,5))
# sns.histplot(df['Fare'], bins=30, kde=True, color='skyblue', edgecolor='black')
# plt.title("Distribution of Ticket Fare Among Titanic Passengers")
# plt.xlabel("Fare (Ticket Price)")
# plt.ylabel("Number of Passengers")
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
# 17. Compute Accuracy, Error rate, Precision, Recall for following confusion matrix ( Use formula for each) True Positives (TPs): 1 False Positives (FPs): 1 False Negatives (FNs): 8 True Negatives (TNs): 90 
# Easy
# 18. Use House_Price prediction dataset. Provide summary statistics (mean, median, minimum, maximum, standard deviation) of variables (categorical vs quantitative) such as- For example, if categorical variable is age groups and quantitative variable is income, then provide summary statistics of income grouped by the age groups. 
# import pandas as pd

# # Load dataset
# path = "/mnt/data/House Data.csv"
# df = pd.read_csv(path)

# # Display dataset info
# print("First 5 records:\n", df.head(), "\n")
# print("Dataset Info:\n")
# print(df.info(), "\n")

# # Identify categorical and quantitative variables
# categorical = df.select_dtypes(include=["object"]).columns.tolist()
# quantitative = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
# print("Categorical variables:", categorical)
# print("Quantitative variables:", quantitative, "\n")

# # Example: Group quantitative variable (Price) by a categorical variable (HouseType or City if available)
# # Automatically pick first categorical & first quantitative column for demo
# if len(categorical) > 0 and len(quantitative) > 0:
#     cat_col = categorical[0]
#     num_col = quantitative[0]

#     print(f"Summary statistics of '{num_col}' grouped by '{cat_col}':\n")
#     grouped_summary = df.groupby(cat_col)[num_col].agg(
#         ['mean', 'median', 'min', 'max', 'std']
#     )
#     print(grouped_summary)
# else:
#     print("Dataset does not contain both categorical and quantitative variables for grouping.")
# 19. Write a Python program to display some basic statistical details like percentile, mean, standard deviation etc (Use python and pandas commands) the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’ of iris.csv dataset. 
# # ---------------------------------------------------------------
# # Q19: Statistical Summary of Each Iris Species
# # ---------------------------------------------------------------

# import pandas as pd

# # 1️⃣ Load dataset
# path = "IRIS.csv"   # change path if needed
# df = pd.read_csv(path)

# # 2️⃣ Check column names
# print("Column Names:", list(df.columns), "\n")

# # Fix naming issue (standardize column name)
# if 'species' in df.columns:
#     df.rename(columns={'species': 'Species'}, inplace=True)

# # 3️⃣ Display first few rows
# print("First 5 records:\n")
# print(df.head(), "\n")

# # 4️⃣ Display basic dataset info
# print("Dataset Info:\n")
# print(df.info(), "\n")

# # 5️⃣ Overall summary statistics
# print("Overall Statistical Summary:\n")
# print(df.describe(), "\n")

# # 6️⃣ Identify unique species
# species_list = df["Species"].unique()
# print("Species present in dataset:", species_list, "\n")

# # 7️⃣ Summary statistics for each species
# for species in species_list:
#     subset = df[df["Species"] == species]
#     print(f"📊 Statistical Details for {species}:\n")
#     print(subset.describe(percentiles=[.25, .5, .75]))
#     print("\n" + "-"*90 + "\n")
# 20. Write a program to cluster a set of points using K-means for IRIS dataset. Consider, K=3, clusters. Consider Euclidean distance as the distance measure. Randomly initialize a cluster mean as one of the data points. Iterate at least for 10 iterations. After iterations are over, print the final cluster means for each of the clusters. 
# # -----------------------------------------------------------------
# # Q20: K-means clustering on IRIS dataset (K = 3, 10 iterations)
# # Distance measure: Euclidean
# # Cluster means initialized randomly from data points
# # -----------------------------------------------------------------

# import pandas as pd
# import numpy as np

# # 1️⃣ Load the dataset
# path = "IRIS.csv"   # change path if needed
# df = pd.read_csv(path)

# # Use only numeric feature columns (ignore species for clustering)
# X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values

# # 2️⃣ Set parameters
# K = 3                      # number of clusters
# max_iters = 10             # number of iterations
# np.random.seed(42)         # for reproducibility (optional)

# # 3️⃣ Randomly initialize cluster means as K distinct data points
# initial_indices = np.random.choice(len(X), K, replace=False)
# means = X[initial_indices, :].astype(float)   # shape: (K, 4)

# print("Initial cluster means (chosen from data points):")
# for i, m in enumerate(means):
#     print(f"Cluster {i+1} initial mean:", m)
# print()

# # 4️⃣ Helper function: compute Euclidean distance between each point and each mean
# def compute_distances(data, means):
#     # data: (n_samples, n_features)
#     # means: (K, n_features)
#     # returns: (n_samples, K) distance matrix
#     dists = np.zeros((data.shape[0], means.shape[0]))
#     for i in range(means.shape[0]):
#         dists[:, i] = np.linalg.norm(data - means[i], axis=1)
#     return dists

# # 5️⃣ K-means main loop
# for it in range(max_iters):
#     # Step 1: Assign each point to nearest mean
#     distances = compute_distances(X, means)
#     cluster_labels = np.argmin(distances, axis=1)   # 0, 1, 2

#     # Step 2: Recompute cluster means as average of assigned points
#     new_means = np.zeros_like(means)
#     for k in range(K):
#         cluster_points = X[cluster_labels == k]
#         if len(cluster_points) > 0:
#             new_means[k] = cluster_points.mean(axis=0)
#         else:
#             # If a cluster gets no points, reinitialize its mean randomly
#             new_means[k] = X[np.random.choice(len(X))]

#     means = new_means

#     print(f"After iteration {it+1}, cluster means are:")
#     for i, m in enumerate(means):
#         print(f"  Cluster {i+1} mean: {m}")
#     print()

# # 6️⃣ Final output
# print("===========================================")
# print("Final cluster means after", max_iters, "iterations:")
# for i, m in enumerate(means):
#     print(f"Cluster {i+1} final mean: {m}")
# print("===========================================")
# 21. Write a program to cluster a set of points using K-means for IRIS dataset. Consider, K=4, clusters. Consider Euclidean distance as the distance measure. Randomly initialize a cluster mean as one of the data points. Iterate at least for 10 iterations. After iterations are over, print the final cluster means for each of the clusters. 
# # -----------------------------------------------------------------
# # Q21: K-means clustering on IRIS dataset (K = 4, 10 iterations)
# # Distance measure: Euclidean
# # Cluster means initialized randomly from actual data points
# # -----------------------------------------------------------------

# import pandas as pd
# import numpy as np

# # 1️⃣ Load the dataset
# path = "IRIS.csv"  # change path if required
# df = pd.read_csv(path)

# # Use only numerical columns
# X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values

# # 2️⃣ Parameters
# K = 4                      # number of clusters
# max_iters = 10             # number of iterations
# np.random.seed(42)         # for reproducibility

# # 3️⃣ Randomly initialize cluster centroids from existing points
# initial_indices = np.random.choice(len(X), K, replace=False)
# means = X[initial_indices, :].astype(float)

# print("Initial cluster means (randomly chosen data points):")
# for i, m in enumerate(means):
#     print(f"Cluster {i+1} initial mean: {m}")
# print()

# # 4️⃣ Define Euclidean distance function
# def compute_distances(data, means):
#     dists = np.zeros((data.shape[0], means.shape[0]))
#     for i in range(means.shape[0]):
#         dists[:, i] = np.linalg.norm(data - means[i], axis=1)
#     return dists

# # 5️⃣ Run K-means clustering for 10 iterations
# for it in range(max_iters):
#     # Step 1: Assign each point to the nearest cluster
#     distances = compute_distances(X, means)
#     cluster_labels = np.argmin(distances, axis=1)

#     # Step 2: Recalculate cluster means
#     new_means = np.zeros_like(means)
#     for k in range(K):
#         cluster_points = X[cluster_labels == k]
#         if len(cluster_points) > 0:
#             new_means[k] = cluster_points.mean(axis=0)
#         else:
#             # If cluster is empty, randomly reinitialize
#             new_means[k] = X[np.random.choice(len(X))]
    
#     means = new_means

#     print(f"After iteration {it+1}, cluster means are:")
#     for i, m in enumerate(means):
#         print(f"  Cluster {i+1} mean: {m}")
#     print()

# # 6️⃣ Print final cluster means
# print("=====================================================")
# print(f"Final cluster means after {max_iters} iterations:")
# for i, m in enumerate(means):
#     print(f"Cluster {i+1} final mean: {m}")
# print("=====================================================")
# 22. Compute Accuracy, Error rate, Precision, Recall for the following confusion matrix. Actual Class\Predicted class cancer = yes cancer = no Total cancer = yes 90 140 230 cancer = no 210 9560 9770 Total 300 9700 10000 
# Metric	Formula	Value	Interpretation
# Accuracy	(TP + TN) / Total	96.5%	Overall model correctness
# Error Rate	(FP + FN) / Total	3.5%	Fraction of incorrect predictions
# Precision	TP / (TP + FP)	39.13%	Reliability of positive predictions
# Recall (Sensitivity)	TP / (TP + FN)	30%	Ability to detect actual cancer cases

# 23. With reference to Table , obtain the Frequency table for the attribute age. From the frequency table you have obtained, calculate the information gain of the frequency table while splitting on Age. (Use step by step Python/Pandas commands) 
# # ---------------------------------------------------------------
# # Q23: Compute Information Gain for attribute 'Age'
# # ---------------------------------------------------------------

# import pandas as pd
# import numpy as np
# import math

# # 1️⃣ Create the dataset manually from the given table
# data = {
#     'Age': ['Young', 'Young', 'Middle', 'Old', 'Old', 'Old', 'Middle', 'Young', 'Young', 'Old', 
#             'Young', 'Middle', 'Middle', 'Old'],
#     'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 
#                'Medium', 'Low', 'High', 'Medium'],
#     'Married': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 
#                 'No', 'No', 'Yes', 'No'],
#     'Health': ['Fair', 'Good', 'Fair', 'Fair', 'Fair', 'Good', 'Good', 'Fair', 'Fair', 'Fair', 
#                'Good', 'Good', 'Fair', 'Good'],
#     'Class': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 
#               'Yes', 'Yes', 'Yes', 'No']
# }

# df = pd.DataFrame(data)
# print("Original Dataset:\n")
# print(df)
# print("\n")

# # 2️⃣ Frequency table for the attribute 'Age'
# freq_table = pd.crosstab(df['Age'], df['Class'])
# print("Frequency Table for Age:\n")
# print(freq_table)
# print("\n")

# # 3️⃣ Helper function to calculate entropy
# def entropy(p_list):
#     total = sum(p_list)
#     if total == 0:
#         return 0
#     ent = 0
#     for p in p_list:
#         if p != 0:
#             prob = p / total
#             ent -= prob * math.log2(prob)
#     return ent

# # 4️⃣ Compute overall dataset entropy (before splitting)
# overall_counts = df['Class'].value_counts()
# entropy_total = entropy(overall_counts)
# print(f"Overall Entropy of the Dataset = {entropy_total:.4f}\n")

# # 5️⃣ Compute entropy for each Age group
# age_groups = df.groupby('Age')['Class']
# entropy_age = {}
# for age, group in age_groups:
#     counts = group.value_counts()
#     e = entropy(counts)
#     entropy_age[age] = e
#     print(f"Entropy({age}) = {e:.4f}")

# # 6️⃣ Compute weighted average entropy after splitting on Age
# total_len = len(df)
# weighted_entropy = 0
# for age, group in age_groups:
#     weight = len(group) / total_len
#     weighted_entropy += weight * entropy_age[age]

# print(f"\nWeighted Entropy (after splitting on Age) = {weighted_entropy:.4f}")

# # 7️⃣ Information Gain
# info_gain = entropy_total - weighted_entropy
# print(f"Information Gain for splitting on 'Age' = {info_gain:.4f}")
# 24. Perform the following operations using Python on a suitable data set, counting unique values of data, format of each column, converting variable data type (e.g. from long to short, vice versa), identifying missing values and filling in the missing values. 
# # ---------------------------------------------------------------
# # Q24: Basic data wrangling operations using Pandas
# #   - counting unique values
# #   - checking format / data type of each column
# #   - converting data types (long → short, short → long)
# #   - identifying missing values
# #   - filling missing values
# # ---------------------------------------------------------------

# import pandas as pd
# import numpy as np

# # 1️⃣ Load a suitable dataset (Titanic here as example)
# # Replace "Titanic.csv" with your own dataset file if needed
# path = "Titanic.csv"
# df = pd.read_csv(path)

# print("First 5 rows of dataset:\n")
# print(df.head(), "\n")

# # 2️⃣ Format / data type of each column
# print("Data types (format) of each column:\n")
# print(df.dtypes, "\n")        # or df.info()

# # 3️⃣ Counting unique values of data

# print("Number of unique values in each column:\n")
# print(df.nunique(), "\n")

# # Example: unique values of specific columns
# print("Unique values in 'Sex' column:", df["Sex"].unique())
# print("Unique values in 'Pclass' column:", df["Pclass"].unique(), "\n")

# # 4️⃣ Converting variable data type
# #    Example 1: Convert 'Pclass' from int to category (saving memory)
# print("Before type conversion, Pclass dtype:", df["Pclass"].dtype)
# df["Pclass"] = df["Pclass"].astype("category")
# print("After type conversion, Pclass dtype:", df["Pclass"].dtype, "\n")

# #    Example 2: Convert 'PassengerId' from int64 (long) to int32 (short)
# print("Before conversion, PassengerId dtype:", df["PassengerId"].dtype)
# df["PassengerId"] = df["PassengerId"].astype("int32")
# print("After conversion, PassengerId dtype:", df["PassengerId"].dtype, "\n")

# #    Example 3: If needed, convert back to a larger type (short → long)
# df["PassengerId"] = df["PassengerId"].astype("int64")   # long again
# print("Converted back, PassengerId dtype:", df["PassengerId"].dtype, "\n")

# # 5️⃣ Identifying missing values

# print("Count of missing (NaN) values in each column:\n")
# print(df.isnull().sum(), "\n")

# # 6️⃣ Filling missing values

# # Example strategy:
# # - For numeric columns like 'Age' → fill with mean
# # - For categorical columns like 'Embarked' → fill with mode
# # - For text columns like 'Cabin' → fill with 'Unknown'

# # 6a) Fill numeric missing values (Age) with mean
# if "Age" in df.columns:
#     age_mean = df["Age"].mean()
#     print(f"Mean Age = {age_mean:.2f}")
#     df["Age"].fillna(age_mean, inplace=True)

# # 6b) Fill categorical missing values (Embarked) with mode
# if "Embarked" in df.columns:
#     embarked_mode = df["Embarked"].mode()[0]
#     print(f"Mode of Embarked = {embarked_mode}")
#     df["Embarked"].fillna(embarked_mode, inplace=True)

# # 6c) Fill text / string column (Cabin) with a label
# if "Cabin" in df.columns:
#     df["Cabin"].fillna("Unknown", inplace=True)

# print("\nMissing values after filling:\n")
# print(df.isnull().sum())

# # (Optional) Save cleaned dataset
# # df.to_csv("Titanic_cleaned.csv", index=False)
# 25. Perform Data Cleaning, Data transformation using Python on any data set
# # ---------------------------------------------------------------
# # Q25: Data Cleaning & Data Transformation using Python (Pandas)
# # Dataset used: Titanic.csv  (can be replaced with any dataset)
# # ---------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# # 1️⃣ Load dataset
# path = "Titanic.csv"      # change file name/path as needed
# df = pd.read_csv(path)

# print("=== 1. RAW DATA (FIRST 5 ROWS) ===")
# print(df.head(), "\n")

# print("=== 2. BASIC INFO BEFORE CLEANING ===")
# print(df.info(), "\n")

# # -------------------------------------------------------
# # DATA CLEANING
# # -------------------------------------------------------

# # 3️⃣ Remove duplicate rows (if any)
# before_dups = df.shape[0]
# df = df.drop_duplicates()
# after_dups = df.shape[0]
# print(f"Removed {before_dups - after_dups} duplicate rows.\n")

# # 4️⃣ Check missing values
# print("=== 3. MISSING VALUES (BEFORE FILLING) ===")
# print(df.isnull().sum(), "\n")

# # 5️⃣ Handle missing values

# # Example: fill missing Age with mean
# if "Age" in df.columns:
#     age_mean = df["Age"].mean()
#     df["Age"].fillna(age_mean, inplace=True)
#     print(f"Filled missing Age values with mean: {age_mean:.2f}")

# # Example: fill missing Embarked with mode
# if "Embarked" in df.columns:
#     embarked_mode = df["Embarked"].mode()[0]
#     df["Embarked"].fillna(embarked_mode, inplace=True)
#     print(f"Filled missing Embarked values with mode: {embarked_mode}")

# # Example: fill missing Cabin with 'Unknown'
# if "Cabin" in df.columns:
#     df["Cabin"].fillna("Unknown", inplace=True)
#     print("Filled missing Cabin values with 'Unknown'.")

# print("\n=== 4. MISSING VALUES (AFTER FILLING) ===")
# print(df.isnull().sum(), "\n")

# # 6️⃣ Fix data types (data-type conversion)

# # Pclass is categorical, not numeric
# if "Pclass" in df.columns:
#     print("Pclass before:", df["Pclass"].dtype)
#     df["Pclass"] = df["Pclass"].astype("category")
#     print("Pclass after:", df["Pclass"].dtype, "\n")

# # Ensure PassengerId is integer (shorter type)
# if "PassengerId" in df.columns:
#     print("PassengerId before:", df["PassengerId"].dtype)
#     df["PassengerId"] = df["PassengerId"].astype("int32")
#     print("PassengerId after:", df["PassengerId"].dtype, "\n")

# # -------------------------------------------------------
# # DATA TRANSFORMATION
# # -------------------------------------------------------

# # 7️⃣ Feature engineering – create new features

# # Example: FamilySize = SibSp + Parch + 1
# if set(["SibSp", "Parch"]).issubset(df.columns):
#     df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
#     print("Created new feature 'FamilySize' = SibSp + Parch + 1.\n")

# # Example: IsAlone flag (1 if alone, else 0)
# if "FamilySize" in df.columns:
#     df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
#     print("Created new feature 'IsAlone' (1 if alone, else 0).\n")

# # 8️⃣ One-Hot Encoding for categorical variables

# cat_cols = []
# for col in ["Sex", "Embarked", "Pclass"]:
#     if col in df.columns:
#         cat_cols.append(col)

# print("Categorical columns to encode:", cat_cols, "\n")

# # Use pandas get_dummies for simplicity
# df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# print("=== 5. DATA AFTER ONE-HOT ENCODING (FIRST 5 ROWS) ===")
# print(df_encoded.head(), "\n")

# # 9️⃣ Scaling (Standardization) of numeric features

# numeric_cols = df_encoded.select_dtypes(include=["int64", "int32", "float64"]).columns.tolist()

# # Remove target variable if present (e.g., 'Survived')
# for target_col in ["Survived"]:
#     if target_col in numeric_cols:
#         numeric_cols.remove(target_col)

# print("Numeric columns to scale:", numeric_cols, "\n")

# scaler = StandardScaler()
# df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

# print("=== 6. DATA AFTER SCALING NUMERIC FEATURES (FIRST 5 ROWS) ===")
# print(df_encoded.head(), "\n")

# # 🔟 Final info
# print("=== 7. FINAL CLEANED & TRANSFORMED DATA INFO ===")
# print(df_encoded.info(), "\n")

# # (Optional) Save cleaned + transformed data
# # df_encoded.to_csv("Titanic_clean_transformed.csv", index=False)


