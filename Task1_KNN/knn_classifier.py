import numpy as np

data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
]

# Label encoding
labels = {
    'Apple': 0,
    'Banana': 1,
    'Orange': 2
}

# One Hot encoding not implemented


X = np.array([row[:3] for row in data], dtype=float)
y = np.array([labels[row[3]] for row in data])

# Normalising data so that Weights dont overpower size and color
def min_max_normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def z_score_normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = min_max_normalize(X)


# making separate functions to calculate different types of distances
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p=3):
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)


# the main KNN class (uses euclidian distance by default)
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            predictions.append(self.predict_one(x))
        return np.array(predictions)

    def distance(self, x1, x2):
        return euclidean_distance(x1,x2)

    def predict_one(self, x):
        distances = []
        for i in range(len(self.X_train)):
            #dist = euclidean_distance(x, self.X_train[i])
            dist = self.distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
         #print(distances)

        distances.sort(key=lambda d: d[0])

        k_nearest = [label for (dist, label) in distances[:self.k]]

        counts = np.bincount(k_nearest)
        return np.argmax(counts)
    

# Implementing Manhattan and Minkowski KNNs as subclasses, only overriding the distance() function
class KNN_Manhattan(KNN):
    def distance(self, x1, x2):
        return manhattan_distance(x1, x2)

class KNN_Minkowski(KNN):
    def __init__(self, k=3, p=3):
        super().__init__(k)
        self.p = p

    def distance(self, x1, x2):
        return minkowski_distance(x1, x2, self.p)
    

# Weighted KNN as subclass
class KNN_Weighted(KNN):
    def predict_one(self, x):
        distances = []
        for i in range(len(self.X_train)):
            dist = self.distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda d: d[0])
        k_nearest = distances[:self.k]

        # Weighted voting
        votes = {}

        for dist, label in k_nearest:
            weight = 1 / (dist + 1e-5)  # Higher dist => lower w and 1e-5 added to avoid division by zero

            if label in votes:
                votes[label] += weight
            else:
                votes[label] = weight

        return max(votes, key=votes.get)
    
# TESTING SECTION

test_data = np.array([
    [118, 6.2, 0],  # Expected: Banana
    [160, 7.3, 1],  # Expected: Apple
    [185, 7.7, 2]   # Expected: Orange
])
test_ytrue = np.array([1,0,2])
#test_data = min_max_normalize(test_data)
# Applying the same normalisation as training data on test data
test_data = (test_data - X_min) / (X_max - X_min)

# Function for Accuracy (on a scale of [0,1])
def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

knn3 = KNN(k=3)
knn5 = KNN(k=5)
knn_mh = KNN_Manhattan(k=3)
knn_mk = KNN_Minkowski(k=3, p=5)
knn_wt = KNN_Weighted(k=3)

knn3.fit(X, y)
knn5.fit(X, y)
knn_mh.fit(X, y)
knn_mk.fit(X, y)
knn_wt.fit(X, y)

# Making a function to print model results for each type
def evaluate_model(name, model, X_test, y_true):
    preds = model.predict(X_test)
    print(f"\n{name}")
    print("Predictions:", preds)
    print("Accuracy:", accuracy(y_true, preds))

evaluate_model("Euclidian (k=3)", knn3, test_data, test_ytrue)
evaluate_model("Euclidian (k=5)", knn5, test_data, test_ytrue)
evaluate_model("Manhattan (k=3)", knn_mh, test_data, test_ytrue)
evaluate_model("Minkowski (p=5,k=3)", knn_mk, test_data, test_ytrue)
evaluate_model("Weighted Euclidian (k=3)", knn_wt, test_data, test_ytrue)
