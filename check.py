import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import AgGrid
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
# Styles CSS
st.markdown("""
    <style>
    /* Style for header */
    .header {
        background-color: #4CAF50;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 24px;
        font-weight: bold;
    }
    
    /* Style for footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
    }
    
    /* Style for navigation bar */
    .navbar {
        overflow: hidden;
        background-color: #333;
    }
    .navbar a {
        float: left;
        display: block;
        color: white;
        text-align: center;
        padding: 14px 16px;
        text-decoration: none;
    }
    .navbar a:hover {
        background-color: #ddd;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# Ajouter le header
st.markdown('<div class="header">Application Machine Learning</div>', unsafe_allow_html=True)

# Ajouter la barre de navigation
st.markdown("""
<div class="navbar">
    <a href="#home">Home</a>
    <a href="#upload">Upload Data</a>
    <a href="#visualization">Visualization</a>
    <a href="#model">Model Training</a>
</div>
""", unsafe_allow_html=True)

# Corps principal de l'application
st.title("Bienvenue dans l'application")
st.write("Ceci est une démonstration d'une application Streamlit avec un header, une barre de navigation et un footer.")

# Ajouter le footer
st.markdown('<div class="footer">© 2025 Machine Learning App | Powered by Streamlit</div>', unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    st.write("Use the links below to navigate:")
    page = st.radio("Go to:", ["Home", "Data Upload", "Data Preprocessing", "Visualization", "Model Training"])

# Home Section
if page == "Home":
    st.subheader("Welcome to the Machine Learning Application")
    st.write("Explore your dataset, preprocess it, and build machine learning models seamlessly.")

# Data Upload Section
if page == "Data Upload":
    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        AgGrid(df)

# Data Preprocessing Section
if page == "Data Preprocessing":
    st.subheader("Data Preprocessing")
    if 'df' in locals():
        st.write("Dataset Loaded Successfully.")
        # Add preprocessing steps here (e.g., handling missing values, outliers)
    else:
        st.error("Please upload a dataset first.")

# Visualization Section
if page == "Visualization":
    st.subheader("Data Visualization")
    if 'df' in locals():
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        x_axis = st.selectbox("Select X-axis", numeric_columns)
        y_axis = st.selectbox("Select Y-axis", numeric_columns)
        chart_type = st.radio("Select Chart Type", ["Scatter", "Bar", "Line"])
        if x_axis and y_axis:
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis)
            elif chart_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis)
            st.plotly_chart(fig)
    else:
        st.error("Please upload a dataset first.")

# Model Training Section
if page == "Model Training":
    st.subheader("Train Your Model")
    if 'df' in locals():
        target = st.selectbox("Select Target Variable", df.columns)
        features = st.multiselect("Select Features", df.columns.drop(target))
        if target and features:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
            if st.button("Save Model"):
                joblib.dump(model, "model.pkl")
                st.success("Model saved successfully!")
    else:
        st.error("Please upload and preprocess a dataset first.")

# Footer
st.markdown('<div class="footer">© 2025 Machine Learning App | Created with ❤️ by Streamlit</div>', unsafe_allow_html=True)

import streamlit as st

# Set up a theme toggle
st.sidebar.title("Theme Toggle")
theme = st.sidebar.radio("Choose Theme:", ("Light", "Dark"))

# Define CSS for themes
light_theme = """
<style>
body {
    background-color: white;
    color: black;
}
</style>
"""

dark_theme = """
<style>
body {
    background-color: #2e2e2e;
    color: white;
}
</style>
"""

# Apply the selected theme
if theme == "Light":
    st.markdown(light_theme, unsafe_allow_html=True)
else:
    st.markdown(dark_theme, unsafe_allow_html=True)

# Example content
st.title("Streamlit Theme Toggle Example")
st.write("This is an example of how to toggle between light and dark themes.")


# Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    grid_return = AgGrid(df)

    # Capture the output of df.info()
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()

    # Display the captured info in Streamlit
    st.text("Dataset Information:")
    st.text(info)

    # Handle missing values
    st.write("### Handling Missing Values")
    if df.isnull().sum().any():
        df.fillna(df.median(numeric_only=True), inplace=True)
        st.write("Missing values filled with column medians.")
    else:
        st.write("No missing values detected.")

    # Remove duplicates
    st.write("### Removing Duplicates")
    initial_shape = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_shape:
        st.write(f"Removed {initial_shape - df.shape[0]} duplicate rows.")
    else:
        st.write("No duplicate rows detected.")

    # Handle outliers (Z-score method)
    st.write("### Handling Outliers")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    column_outlier = st.selectbox("Sélectionner une colonne pour gérer les outliers", numeric_columns)
    
    if column_outlier:
        # Méthode 1: Détection avec l'écart interquartile (IQR)
        Q1 = df[column_outlier].quantile(0.25)
        Q3 = df[column_outlier].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column_outlier] < lower_bound) | (df[column_outlier] > upper_bound)]
        
        st.write(f"Nombre d'outliers détectés : {outliers.shape[0]}")

    # Encode categorical features
    st.write("### Encoding Categorical Features")
    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    st.write("Categorical features encoded.")

    
    st.write("### Value Count")  
    selected_columns1 = st.multiselect("Sélectionnez les colonnes ", df.columns)
    for columns in selected_columns1:    
        df[columns].value_counts() * 100 / len(df)

    st.write("### Data Visualisation")     
    columns = df.columns.tolist()
    x_axis = st.selectbox("Select X-axis Column", columns)
    y_axis = st.selectbox("Select Y-axis Column", columns)
    chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot"])

    # Visualization using Plotly
    if x_axis and y_axis:
        if chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, title="Scatter Plot")
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis, title="Line Chart")
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, title="Bar Chart")
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_axis, y=y_axis, title="Box Plot")
        st.plotly_chart(fig)
    if st.checkbox("Show Pair Plot"):
       sns.pairplot(df)
       st.pyplot()
        
    if st.checkbox("Show Correlation Matrix"):
        st.write("Correlation Matrix:")
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        # Display correlation matrix as a heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
    # Split the data into train and test sets
    st.write("### Machine Learning Model")
    
    # Select learning type
    learning_type = st.selectbox("Select Learning Type", ["Supervised", "Unsupervised"])

    if learning_type == "Supervised":    
        
        learning_type1 = st.selectbox("Select Learning Type", ["Classification", "Regression"])
        target_column = st.selectbox("Select Target Column", df.columns)
        feature_columns = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_column])

        if target_column and feature_columns:
            # Prepare data
            X = df[feature_columns]
            y = df[target_column]

            # Train/Test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if learning_type1 == "Classification":
    
               num_features = st.slider("Select Number of Features", 1, len(feature_columns))
               selector = SelectKBest(score_func=f_classif, k=num_features)
               X_selected = selector.fit_transform(X, y)
               selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
               st.write("Selected Features:", selected_features) 
               # Select supervised algorithm
               Classification_method = st.selectbox(
                  "Select Classification Learning Method",
                  ["Logistic Regression", "KNN", "Decision Tree","Random Forest","Support Vector Machine"]
                )

               # Train the selected Classifcation model
               if Classification_method == "Logistic Regression":
                   model = LogisticRegression(max_iter=1000)
               elif Classification_method == "KNN":
                  model = KNeighborsClassifier(n_neighbors=28)
               elif Classification_method == "Decision Tree":
                  model = tree.DecisionTreeClassifier()    
               elif Classification_method == "Random Forest":
                   model = RandomForestClassifier(random_state=42)
               elif Classification_method == "Support Vector Machine":
                    model = SVC()
               # Train the model
               model.fit(X_train, y_train)

               # Make predictions
               y_pred = model.predict(X_test)
               y_pred_train = model.predict(X_train) # predict train
               cm = confusion_matrix(y_test, y_pred)
               st.write("Confusion Matrix:")
               fig, ax = plt.subplots()
               sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
               st.pyplot(fig) 

               # Display accuracy
               st.write(f"Classification Model: {Classification_method}")
               st.write("Accuracy of train={:.2f}".format(metrics.accuracy_score(y_train, y_pred_train)))
               st.write("Accuracy of test={:.2f}".format(metrics.accuracy_score(y_test, y_pred)))
               if st.button("Download Predictions"):
                     output = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                     buffer = io.BytesIO()
                     output.to_csv(buffer, index=False)
                     buffer.seek(0)
                     st.download_button("Download CSV", buffer, "predictions.csv") 
            elif learning_type1 == "Regression":
    
                regressor = st.selectbox(
                      "Select Regression Algorithm",
                      ["Linear Regression", "Polynomial Regression", "Multipolynomial Regression"]
                    )
               
                if regressor == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write("Regression Model: Linear Regression")
                    st.write("Mean Squared Error (MSE):", mse)
                    st.write("R² Score:", r2)

                elif regressor == "Polynomial Regression":
                    degree = st.slider("Select Polynomial Degree", 2, 5, 2)
                    poly = PolynomialFeatures(degree=degree)
                    X_poly_train = poly.fit_transform(X_train)
                    X_poly_test = poly.transform(X_test)

                    model = LinearRegression()
                    model.fit(X_poly_train, y_train)
                    y_pred = model.predict(X_poly_test)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write(f"Regression Model: Polynomial Regression (Degree {degree})")
                    st.write("Mean Squared Error (MSE):", mse)
                    st.write("R² Score:", r2)

                elif regressor == "Multipolynomial Regression":
                    degree_x = st.slider("Select Polynomial Degree for Feature 1", 2, 5, 2)
                    degree_y = st.slider("Select Polynomial Degree for Feature 2", 2, 5, 2)

                    if len(feature_columns) >= 2:
                        X_poly_train = np.hstack([
                            PolynomialFeatures(degree=degree_x).fit_transform(X_train.iloc[:, [0]]),
                            PolynomialFeatures(degree=degree_y).fit_transform(X_train.iloc[:, [1]])
                        ])
                        X_poly_test = np.hstack([
                            PolynomialFeatures(degree=degree_x).transform(X_test.iloc[:, [0]]),
                            PolynomialFeatures(degree=degree_y).transform(X_test.iloc[:, [1]])
                        ])

                        model = LinearRegression()
                        model.fit(X_poly_train, y_train)
                        y_pred = model.predict(X_poly_test)

                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        st.write(f"Regression Model: Multipolynomial Regression (Degrees {degree_x} & {degree_y})")
                        st.write("Mean Squared Error (MSE):", mse)
                        st.write("R² Score:", r2)
                    else:
                        st.error("Multipolynomial Regression requires at least two features.")
   
            
                if st.button("Download Predictions"):
                     output = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                     buffer = io.BytesIO()
                     output.to_csv(buffer, index=False)
                     buffer.seek(0)
                     st.download_button("Download CSV", buffer, "predictions.csv")
       
    elif learning_type == "Unsupervised":
        # Unsupervised Learning
        feature_columns = st.multiselect("Select Feature Columns", df.columns)

        if feature_columns:
            # Prepare data
            X = df[feature_columns]

            # Select unsupervised algorithm
            unsupervised_method = st.selectbox(
                "Select Unsupervised Learning Method",
                ["K-Means Clustering", "Principal Component Analysis (PCA)"]
            )

            if unsupervised_method == "K-Means Clustering":
                n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
                model = KMeans(n_clusters=n_clusters, random_state=42)
                model.fit(X)

                # Add cluster labels to the dataset
                df['Cluster'] = model.labels_
                st.write("Cluster Labels:")
                st.dataframe(df)
                cluster_fig = px.scatter(
                df, x=feature_columns[0], y=feature_columns[1], color=df['Cluster'].astype(str),
                title="K-Means Clustering Visualization"
                   )
                st.plotly_chart(cluster_fig)

            elif unsupervised_method == "Principal Component Analysis (PCA)":
                n_components = st.slider("Select Number of Components", 2, min(len(feature_columns), 10))
                model = PCA(n_components=n_components)
                principal_components = model.fit_transform(X)

                # Create a DataFrame with PCA results
                pca_df = pd.DataFrame(
                    principal_components,
                    columns=[f"Principal Component {i+1}" for i in range(n_components)]
                )
                st.write("PCA Results:")
                st.dataframe(pca_df)

if st.button("Save Model"):
    joblib.dump(model, "trained_model.pkl")
    st.success("Model saved as 'trained_model.pkl'")

if st.button("Load Model"):
    model = joblib.load("trained_model.pkl")
    st.success("Model loaded successfully")  
import pdfkit
from docx import Document

# Generate Report
scaler_option = "StandardScaler"  # Example assignment
if st.button("Generate Report"):
    report_content = f"""
    Dataset Overview:
    {df.describe().to_string()}

    Preprocessing Steps:
    - Scaling: {scaler_option}
    - Outlier Detection: {outlier_method}

    Model Metrics:
    """
    for model_name, metrics in results.items():
        report_content += f"\n{model_name}:\nAccuracy: {metrics['Accuracy']:.2f}\n"

    # Save as PDF
    pdfkit.from_string(report_content, "report.pdf")
    st.write("Report saved as 'report.pdf'")
    st.download_button("Download PDF", data=open("report.pdf", "rb"), file_name="report.pdf")

feedback = st.text_area("Please provide your feedback:")
if feedback:
    st.success("Thank you for your feedback!")
