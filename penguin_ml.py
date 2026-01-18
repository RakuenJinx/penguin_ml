# %%
import os
import pickle

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# %%
os.chdir("f:/VSCode/streamlit/penguin_ml")
penguin_df = pd.read_csv("penguins.csv").dropna()
output = penguin_df.species
features_list = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]
features = pd.get_dummies(penguin_df[features_list])
output, uniques = pd.factorize(penguin_df.species)
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.8)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(f"score is {score}")
rf_pickle = open("random_forest_penguin.pickle", "wb")
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
output_pickle = open("output_penguin.pickle", "wb")
pickle.dump(uniques, output_pickle)
output_pickle.close()
# Create a DataFrame for sorting and plotting
importance_df = pd.DataFrame(
    {"feature": features.columns, "importance": rfc.feature_importances_}
).sort_values(by="importance", ascending=False)
fig, ax = plt.subplots()
# Plot using the sorted DataFrame with distinct colors
ax = sns.barplot(
    data=importance_df, x="importance", y="feature", hue="feature", legend=False
)
plt.title("Which features are the most important for species prediction?")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
fig.savefig("feature_importance.png")
