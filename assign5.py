import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Update the path to your dataset
file_path = r'C:/Users/sayed/PycharmProjects/assig5/aps_failure_training_set.csv'

# Load the dataset
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

# Function to train, test, and evaluate the model and visualize the decision tree
def evaluate_and_visualize_tree(X, y, test_size):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print metrics
    print(f"Results for test size {test_size * 100}%:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()

    # Visualize the decision tree
    plt.figure(figsize=(20,10))  # Set the size of the plot
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=['neg', 'pos'], rounded=True)
    plt.show()

# Evaluate for 70-30 split and visualize the tree
evaluate_and_visualize_tree(X, y, test_size=0.3)

# Evaluate for 80-20 split and visualize the tree
evaluate_and_visualize_tree(X, y, test_size=0.2)
