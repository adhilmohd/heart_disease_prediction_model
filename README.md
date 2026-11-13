
Build a Heart Disease Prediction Model  
Author: Adhil Muhammed
---
![](https://nextwork.ai/adhil-muhammed/uploads/86fdf6d0-ebb6-4049-a54b-ba10db950a5f_vqfi3stj)
Project Overview
Goals and motivation
Heart disease is the leading cause of death worldwide. Doctors already use clinical measurements like blood pressure, cholesterol, and heart rate to assess risk, but what if a computer could learn from hundreds of real patient records to predict heart disease automatically?
In this project, i will build a machine learning model that predicts whether a patient has heart disease based on real clinical data from 303 patients.
I am going to build , A complete ML pipeline that loads real patient data, visualizes it, preprocesses it, trains two classifiers, evaluates them with cross-validation, and exports the best model as a file ready for API integration.
Setting Up the Environment
Step goals
In this step, I'm setting up a new notebook in google collab as it comes with popular data science libraries pre-installed in it. I verified all required libraries are there in it and understood the dataset that i would be working with it
![Setting Up the Environment](https://nextwork.ai/adhil-muhammed/uploads/86fdf6d0-ebb6-4049-a54b-ba10db950a5f_wzkezqdv)
Libraries and their roles
I imported pandas,numpy, matplotlib, seaborn, scikit-learn and joblib.
pandas loads and manipulates tabular data (your patient dataset).
numpy handles numerical operations under the hood.
matplotlib and seaborn create visualizations like distribution plots and heatmaps.
sklearn (scikit-learn) provides the machine learning tools: splitting data, scaling features, training models, and evaluating results.
joblib saves your trained model to a file so it can be loaded later.
Loading and Exploring Real Clinical Data
Step goals
In this step, I am loading the UCI Cleveland Heart Disease dataset, check it for missing values, and create visualizations that reveal which features relate most to heart disease.
Binarizing the target variable
Binarizing converted the target from 0 to 4, 0 to 1 , because we need to predict a yes/no answer, does the patient have disease or not.
![Loading and Exploring Real Clinical Data](https://nextwork.ai/adhil-muhammed/uploads/86fdf6d0-ebb6-4049-a54b-ba10db950a5f_dret9ctv)
Key insight from the heatmap
1. The Strongest Positive Correlation: thal or ca
thal (Thalassemia status): If your notebook successfully converted this column to numeric, it will show a positive correlation of about 0.52 with the target column github.com. This represents the strongest overall positive relationship in the dataset.
ca (Number of major vessels colored by fluoroscopy): If thal was not computed as a numeric type in your correlation matrix, ca is the next strongest positive correlate with a value of about 0.46 github.com.
2. The Strongest Negative Correlation: thalach
thalach (Maximum heart rate achieved): This feature has a negative correlation of about -0.42 with the target column github.com link.springer.com. This means that as a patient's maximum heart rate goes down, the likelihood of having heart disease goes up.
Preprocessing and Splitting the Dataset
Step goals
The visualizations revealed which features correlate most with heart disease. Now it's time to clean the data and prepare it for model training.
Machine learning models cannot handle missing values or unscaled features. In this step, I'll fill in the gaps, separate features from the target, and split the data into training and test sets.
![Preprocessing and Splitting the Dataset](https://nextwork.ai/adhil-muhammed/uploads/86fdf6d0-ebb6-4049-a54b-ba10db950a5f_nwctz88x)
Why stratified splitting matters
I used stratify because stratify=y preserves the same ratio of disease/no-disease patients in both the training and test sets. Without this, you could accidentally get a test set with mostly one class.
Training and Comparing Two Classifiers
Step goals
In this step, I'm training the data... so that I can... compare and check which one gives the accurate result. I will train a logistic regression pipeline first and then train a random forest pipeline and compare them both.
![Training and Comparing Two Classifiers](https://nextwork.ai/adhil-muhammed/uploads/86fdf6d0-ebb6-4049-a54b-ba10db950a5f_k50pp8x3)
Cross-validation findings
The Logistic Regression model generalizes better because it achieves a higher mean cross-validation accuracy (e.g., ~84%) and a lower standard deviation (e.g., ~0.03) compared to the Random Forest model. This is likely because the dataset is relatively small (303 rows), causing the complex Random Forest model to slightly overfit the training folds, whereas the simpler Logistic Regression model remains robust and consistent.
Evaluating and Exporting the Prediction Model
Step goals
I've trained two models and compared their cross-validation scores. Now it's time to see exactly where each model succeeds and fails.
In this step, I'll visualize both models with confusion matrices, pick the best performer, and export it as a file I can load into any Python application.
Sample patient prediction result
For the first sample patient in the test set, the saved model predicted No Disease. The predicted probability of heart disease for this patient is 0.2177 (or 21.77%), which indicates the model is 78.23% confident that the patient does not have heart disease.
Reflections and Takeaways
Key tools and concepts learned
1. The Tools We Used (The "Software Stack")
In basic Python, you use lists and dictionaries. In data science, we use specialized tools designed to handle millions of rows of data instantly:
a) pandas (The Data Organizer):   
Think of pandas as "Excel inside Python." Instead of using slow nested loops to read text files, pandas loads data into a beautiful, spreadsheet-like table called a DataFrame. We used it to load the data, label columns, find missing values, and calculate statistics.
b) seaborn and matplotlib (The Artists):   
These are Python's graphing libraries. They take the raw numbers from pandas and turn them into visual charts. We used them to create a correlation heatmap, which uses colors (like red and blue) to show us which physical traits are linked to heart disease.
c) scikit-learn (The Machine Learning Brain):   
This is the gold-standard Python library for machine learning. Instead of writing complex mathematical formulas from scratch, scikit-learn provides pre-built "algorithms" (like Logistic Regression and Random Forest) that can look at data and learn from it automatically.
d) joblib (The Model Saver):   
Normally, when your Python script finishes running, all variables and models in memory are deleted. Joblib is a tool that lets you "freeze" your fully trained model and save it as a physical file on your hard drive, so you can load it back up instantly later without retraining.
2. The Core Concepts We Learnt (The "Process")
Building a machine learning model isn't just about throwing data at a computer. It follows a very strict, logical pipeline:
A. Data Cleaning (Handling Missing Values)
Real-world data is messy. In your project, some patient records had question marks (?) instead of actual numbers. You learned that machine learning models will crash if they see empty spaces. You resolved this by calculating the median (middle) value of those columns and filling in the gaps so the data was 100% complete.
B. The Train-Test Split (The Practice Exam vs. The Final Exam)
If you give a student the exact same questions from their study guide on the final exam, they might get a 100% just by memorizing the answers, without actually understanding the subject.   
To prevent this in machine learning, we split our data into two parts:
Training Set (80% of the data): The "study guide" the model uses to learn patterns.
Testing Set (20% of the data): The "final exam" containing patients the model has never seen before, used to prove the model actually knows how to diagnose heart disease.
C. Overfitting vs. Generalization
You learned that a more complex model isn't always better. The complex Random Forest model tried so hard to be perfect that it "overfit" (memorized too much noise in the small training set). The simpler, linear Logistic Regression model generalized much better, meaning it was more consistent and reliable when tested on completely new patients.
D. Cross-Validation (Double-Checking the Results)
Instead of testing our model just once, we used cross-validation to split our data into 5 different chunks and test the model 5 separate times. This ensures our model's high accuracy wasn't just a "lucky guess" on one specific test set, giving us a reliable average score.
E. Predictions vs. Probability
You learned that models don't just output a simple "yes" or "no". Under the hood, they calculate a mathematical probability (like 0.21 or 21.77% chance of sickness). We can set a threshold (usually 50%) to convert that probability into a final clinical diagnosis.
Time and challenges
This project took me approximately 2 days. The challenge was i was new with the machine learning part, so it took me time learn and understand that part.
Final thoughts
I did this project today to learn how the machine learning model works and understand its concepts. My plan is to build more projects on data science and ml
