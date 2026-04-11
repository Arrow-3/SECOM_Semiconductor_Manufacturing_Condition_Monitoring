from src.preprocessing import load_and_preprocess
from src.feature_selection import select_features
from src.training import train_model
from src.evaluation import evaluate_model

print("=== SECOM Manufacturing Pipeline ===")

train_path = "C:\Drive W\Machine Learning\Semiconductor Manufacturing\secom\secom.data"
test_path = "C:\Drive W\Machine Learning\Semiconductor Manufacturing\secom\secom_labels.data"

# Step 1: Load & preprocess
X, y = load_and_preprocess(train_path, test_path)

# Step 2: Feature Selection
X_selected = select_features(X, y, method="mutual_info", k=40)

# Step 3: Train model
results = train_model(X_selected, y, model_type="random_forest")

# Step 4: Evaluate
evaluate_model(results)
