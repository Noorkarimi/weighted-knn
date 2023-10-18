import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class WeightedKNN(KNeighborsClassifier):
    def predict(self, X):
        # Find the k nearest neighbors
        distances, indices = self.kneighbors(X)
        
        # Calculate the weights (inverse of the distance)
        weights = 1 / np.maximum(distances, 1e-10)  # Avoid division by zero
        
        weighted_votes = []
        for i in range(len(X)):
            # Get the class labels of the neighbors
            neighbor_labels = self._y[indices[i]]
            
            # Get the weights of the neighbors
            neighbor_weights = weights[i]
            
            # Calculate the weighted vote
            class_weights = {}
            for label, weight in zip(neighbor_labels, neighbor_weights):
                class_weights[label] = class_weights.get(label, 0) + weight
            
            # Get the class with the highest weighted vote
            sorted_votes = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
            weighted_votes.append(sorted_votes[0][0])
        
        return np.array(weighted_votes)

# Create an instance of the weighted k-NN classifier
weighted_knn = WeightedKNN(n_neighbors=3)
weighted_knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = weighted_knn.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
