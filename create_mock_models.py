import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
# Generate some dummy data (100 samples, 13 features matching your app)
np.random.seed(42)
X_dummy = np.random.rand(100, 13)
# For specific columns like age, cholesterol, etc. we can scale them to be more realistic, 
# but for a dummy model, random data works fine.
X_dummy[:, 0] = np.random.randint(20, 100, 100)  # Age
y_dummy = np.random.randint(0, 2, 100)
# Create and fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummy)
# Create and fit decision tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_scaled, y_dummy)
# Save the models
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("decision_tree_model.pkl", "wb") as f:
    pickle.dump(dt, f)
print("Created dummy scaler.pkl and decision_tree_model.pkl")
