# %% Preparation
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import autokeras as ak
import pandas as pd
import numpy as np

house_dataset = fetch_california_housing()
df = pd.DataFrame(
    np.concatenate((house_dataset.data, house_dataset.target.reshape(-1, 1)), axis=1),
    columns=house_dataset.feature_names + ["Price"],
)
train_size = int(df.shape[0] * 0.9)

# %% Initialize the structured data regressor.

x_train = df[:train_size]
y_train = x_train.pop("Price")
y_train = pd.DataFrame(y_train)

# Preparing testing data.
x_test = df[train_size:]
y_test = x_test.pop("Price")

# It tries 10 different models.
reg = ak.StructuredDataRegressor(max_trials=3, overwrite=True)
# Feed the structured data regressor with training data.
reg.fit(x_train, y_train, epochs=10)

# Predict with the best model.
predicted_y = reg.predict(x_test)

# Evaluate the best model with testing data.
print(reg.evaluate(x_test, y_test))
