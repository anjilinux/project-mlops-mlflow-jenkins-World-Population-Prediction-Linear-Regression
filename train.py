import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Optional: set experiment name
mlflow.set_experiment("world_population_linear_regression")
mlflow.set_tracking("./mlruns")
# Load data
df = pd.read_csv("world_population.csv")

X = df[["year"]]
y = df["population"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run():

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2 Score:", r2)

    # Log parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Save model locally
    joblib.dump(model, "population_model.pkl")

    # Log model as artifact
    mlflow.log_artifact("population_model.pkl")

    # (Optional) log model using MLflow flavor
    mlflow.sklearn.log_model(model, artifact_path="model")









# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import joblib

# # Load data
# df = pd.read_csv("world_population.csv")

# X = df[["year"]]
# y = df["population"]

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("MSE:", mse)
# print("R2 Score:", r2)

# # Save model
# joblib.dump(model, "population_model.pkl")
