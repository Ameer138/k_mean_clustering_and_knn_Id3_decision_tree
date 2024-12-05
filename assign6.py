import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load the dataset (same as before)
file_path = r'C:/Users/sayed/PycharmProjects/Assig5/aps_failure_training_set.csv'
df = pd.read_csv(file_path)

# Preprocess the data
# Replace 'na' with NaN and drop rows with missing values
df.replace('na', pd.NA, inplace=True)
df.dropna(inplace=True)

# Convert target 'class' from 'neg' and 'pos' to 0 and 1
df['class'] = df['class'].map({'neg': 0, 'pos': 1})

# Separate features (X) and target (y)
X = df.drop('class', axis=1)  # Features (all columns except 'class')
y = df['class']  # Target variable

### K-MEANS CLUSTERING ###
print("Performing K-means clustering...")

# Perform K-means clustering (choose 2 clusters since we have binary classification)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Add the clustering results to the dataset
df['cluster'] = kmeans.labels_

# Check the distribution of clusters
print("Cluster distribution:")
print(df['cluster'].value_counts())


### KNN CLASSIFICATION ###
def evaluate_knn(X, y, test_size):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])
    except:
        auc = "N/A"  # AUC is only applicable for binary classification

    # Print metrics
    print(f"Results for test size {test_size * 100}%:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
    disp.plot()
    plt.show()


# Evaluate KNN with 70-30 split
print("\nKNN Classification with 70-30 split:")
evaluate_knn(X, y, test_size=0.3)

# Evaluate KNN with 80-20 split
print("\nKNN Classification with 80-20 split:")
evaluate_knn(X, y, test_size=0.2)
