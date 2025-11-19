Agriculture Yield Analysis & Prediction

This project provides a complete workflow to analyse an agricultural dataset,
train machine‑learning models to predict crop yield, and offer an
interactive Streamlit application for forecasting and recommendations.

📦 Dataset Overview

The Agriculture and Farming dataset (sourced from Kaggle) contains
farm‑level information such as crop type, irrigation method, soil type and
season, together with numerical metrics like farm area, fertilizer and
pesticide usage, yield and water usage
kaggle.com
. Each row
represents one farm’s performance over a growing season. The target
variable is Yield(tons), expressed in metric tons per farm.

Columns
Column	Description
Farm_ID	Unique identifier for each farm.
Crop_Type	Type of crop grown (e.g., Cotton, Carrot, Sugarcane, Tomato, etc.).
Farm_Area(acres)	Size of the farm in acres.
Irrigation_Type	Irrigation method used (Manual, Drip, Flood, Rainfed, Sprinkler).
Fertilizer_Used(tons)	Amount of fertilizer applied in tons.
Pesticide_Used(kg)	Quantity of pesticide applied in kilograms.
Soil_Type	Dominant soil type (e.g., Silty, Peaty, Clayey, Sandy, Loamy).
Season	Cropping season: Kharif, Rabi or Zaid. Kharif crops are sown
	in July–October and harvested during the monsoon; Rabi crops are sown
	October–November and harvested in late winter; Zaid crops are short‑season
	crops grown in March–June, requiring warm, dry weather
testbook.com
.
Water_Usage(cubic meters)	Total water consumed during the season.
Yield(tons)	Total yield harvested in metric tons (target variable).

The dataset was used to explore how crop type, irrigation method, season
and other factors influence yield. Average yields by crop, irrigation
and season are visualised in the Exploratory Data Analysis section.

🧪 Exploratory Data Analysis

Run the script agriculture_analysis.py to generate summary statistics
and charts from the dataset. The script loads the CSV file, cleans and
describes the data, and produces visualisations such as:

Average yield by crop type: identifies high‑yield crops like carrots
and tomatoes.

Yield distribution by irrigation method: compares manual, drip,
flood, rainfed and sprinkler irrigation.

Average yield by season: compares Kharif, Rabi and Zaid seasons.

Correlation matrix: shows relationships between numeric variables
(farm area, fertilizer, pesticide, water and yield).

The figures are saved into the agri_plots directory for inclusion in
reports. You can customise colours, labels or add new plots by editing
the script.

🤖 Model Training

The training workflow is implemented in model_training.py. It performs
the following steps:

Data loading & preprocessing: Categorical variables (crop type,
irrigation type, soil type, season) are one‑hot encoded; numerical
variables are passed through unchanged. This is done via a
ColumnTransformer pipeline.

Splitting: The dataset is split into training and test sets.

Model training: Three tree‑based regressors are trained –
RandomForest, GradientBoosting and XGBRegressor. Each model is
evaluated with cross‑validation using root mean squared error (RMSE).
The best model (based on cross‑validation performance) is selected.

Saving: The preprocessing pipeline and best model are serialised
using joblib to preprocessor.pkl and best_model.pkl.

Reporting: Results (RMSE and R²) are saved to results.csv.

A random forest is a meta estimator that fits a number of decision tree
regressors on bootstrapped samples of the data and averages their
predictions to improve accuracy and reduce over‑fitting
scikit-learn.org
. Gradient
boosting and XGBoost build ensembles of weak learners iteratively to
minimise prediction errors. A simple LSTM training function is also
provided for demonstration (requires TensorFlow).

🌱 Streamlit Application

The interactive dashboard is implemented in agri_app_graphs.py. After
training a model, launch the app with:

streamlit run agri_app_graphs.py

Features

Exploratory Analysis: View average yields by crop, yield
distribution by irrigation type and average yield by season. Preview
the top rows of the dataset to familiarise yourself with the data.

Yield Prediction: Enter farm characteristics (crop, irrigation,
soil type, season, area, fertilizer, pesticide and water usage) in the
sidebar. When you click Predict, the model outputs an estimated
yield and displays a bar chart comparing your prediction to the
dataset’s average yield for the chosen crop, irrigation method and
season.

Recommendations: After each prediction, the app suggests
improvements. It highlights the top‑yielding crop, irrigation method
and season in the dataset and provides generic agronomic best
practices such as balanced fertilisation, efficient water management,
crop rotation and integrated pest management. These guidelines draw
from agricultural research which demonstrates that crop rotation
improves soil fertility and reduces pest pressure
produceleaders.com
, selecting
high‑yield varieties can increase productivity when managed properly
produceleaders.com
,
balanced fertilisation avoids nutrient deficiencies or toxicities
produceleaders.com
,
efficient irrigation (e.g., drip/sprinkler) delivers water
precisely
produceleaders.com
 and integrated pest management relies on
eco‑friendly pest control measures
produceleaders.com
.

Analysis charts (optional): Tick the “Show analysis charts below”
checkbox in the prediction section to display the same analysis charts
used in the EDA page.

Running Tips

The app expects the files agriculture_dataset.csv, preprocessor.pkl
and best_model.pkl to reside in the same directory. If you retrain
the model or modify the dataset, regenerate these files via
model_training.py before launching the app.

🛠️ Setup

The project requires Python 3.8 or newer. To install dependencies:

pip install pandas numpy matplotlib scikit-learn joblib xgboost streamlit


TensorFlow is only needed if you wish to experiment with the optional
LSTM training function. You can install it with:

pip install tensorflow

📁 Repository Structure
├── agriculture_dataset.csv      # Dataset used for training and analysis
├── agriculture_analysis.py      # Script for exploratory data analysis (EDA)
├── model_training.py            # Script to train models and save the best one
├── agri_app_graphs.py           # Streamlit app with EDA, prediction, charts & recommendations
├── preprocessor.pkl             # Saved preprocessing pipeline (generated by training script)
├── best_model.pkl               # Saved best model (generated by training script)
├── results.csv                  # Metrics (RMSE, R²) for each trained model
├── agri_plots/                  # Folder of charts generated by EDA script
└── README.md                    # This file

📝 Notes and Acknowledgements

The dataset originates from Kaggle (Agriculture and Farming dataset).
We thank the original dataset creator for making the data publicly
available for analysis.

Cropping season definitions (Kharif, Rabi, Zaid) were referenced from
agricultural extension sources
testbook.com
.

Irrigation methods and agronomic recommendations were drawn from
agricultural extension articles and academic sources, emphasising
balanced fertilisation, efficient irrigation, crop rotation and
integrated pest management
produceleaders.com
produceleaders.com
.

The Random Forest algorithm description is cited from the
scikit‑learn documentation
scikit-learn.org
.

🙋‍♂️ Contributing

This repository was developed as part of an internship project. Feel
free to fork and adapt it. 
hyperparameter tuning, advanced models, more interactive charts) are
welcome. Please open an issue or submit a pull request with your
improvements.
