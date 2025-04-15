import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,mean_squared_error
import pandas as pd

# Load dataset

st.set_page_config(page_title="Machine Learning", page_icon="📈")

st.markdown("# Machine Learning")
st.sidebar.header("Machine Learning")

upload_file=st.file_uploader("Upload your CSV file", type=["csv"])

if upload_file is not None:
    uploaded_filename = upload_file.name
    if 'uploaded_filename' not in st.session_state or st.session_state.uploaded_filename != uploaded_filename:
        st.success("File uploaded successfully!")
    data = pd.read_csv(upload_file)

# Sidebar for user input
    st.sidebar.header("Model Configuration")
    test_size = st.sidebar.slider("Test Size (Fraction of Data)", 0.1, 0.5, 0.2, 0.05)
    target = st.selectbox("Select your target variable", data.columns.tolist())
    colmn = st.multiselect("Select columns to visualize", data.columns.tolist())

    # Model selection
    st.sidebar.header("Select Algorithms")
    use_random_forest = st.sidebar.checkbox("Random Forest", False)
    use_decision_tree = st.sidebar.checkbox("Decision Tree", False)
    use_svm = st.sidebar.checkbox("Support Vector Machine (SVM)", False)
    use_lr=st.sidebar.checkbox("Linear Regression", False)
    use_lor=st.sidebar.checkbox("Logistic Regression", False)

    # Split data
    X = pd.DataFrame(data[colmn], columns=colmn)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=11)

    # Results dictionary
    results = {}

    if use_random_forest:
        rf_model = RandomForestClassifier(random_state=11)
        rf_model.fit(X_train, y_train)
        rf_y_pred = rf_model.predict(X_test)
        results['Random Forest'] = {
            'Accuracy': accuracy_score(y_test, rf_y_pred),
            'Report': classification_report(y_test, rf_y_pred, target_names=data[target].unique(), output_dict=True)
        }

    if use_decision_tree:
        dt_model = DecisionTreeClassifier(random_state=11)
        dt_model.fit(X_train, y_train)
        dt_y_pred = dt_model.predict(X_test)
        results['Decision Tree'] = {
            'Accuracy': accuracy_score(y_test, dt_y_pred),
            'Report': classification_report(y_test, dt_y_pred, target_names=data[target].unique(), output_dict=True)
        }

    if use_svm:
        svm_model = SVC(random_state=11)
        svm_model.fit(X_train, y_train)
        svm_y_pred = svm_model.predict(X_test)
        results['SVM'] = {
            'Accuracy': accuracy_score(y_test, svm_y_pred),
            'Report': classification_report(y_test, svm_y_pred, target_names=data[target].unique(), output_dict=True)
        }

    if use_lr:
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_y_pred = lr_model.predict(X_test)

        results['Lr'] = {
            
            'Accuracy': mean_squared_error(y_test, lr_y_pred)

        }

    if use_lor:
        lor_model = LogisticRegression(random_state=11)
        lor_model.fit(X_train, y_train)
        lor_y_pred = lor_model.predict(X_test)
        results['Lor'] = {
            'Accuracy': accuracy_score(y_test, lor_y_pred),
            'Report': classification_report(y_test, lor_y_pred, target_names=data[target].unique(), output_dict=True)
        }

    st.write("### Model Results")
    for model_name, result in results.items():
        st.write(f"#### {model_name}")
        st.write(f"Accuracy: {result['Accuracy']:.2f}")
        if model_name=="Lr":
            print()
        else:
            st.write("Classification Report:")
            st.dataframe(pd.DataFrame(result['Report']).transpose())    

    # User input for prediction
    st.write("### Make a Prediction")
    input_data = []
    st.write("#### Select Feature and values for prediction")
    for feature in colmn:
        min_value = float(X[feature].min())
        max_value = float(X[feature].max())
        value = st.slider(f"{feature}", min_value, max_value, float(X[feature].mean()))
        input_data.append(value)
    input_df = pd.DataFrame([input_data], columns=colmn)
    st.write(input_df)

    if st.button("Predict"):
        if use_random_forest:
            rf_prediction = rf_model.predict(input_df)
            st.write(f"Random Forest Prediction: {data[target][rf_prediction[0]]}")
        if use_decision_tree:
            dt_prediction = dt_model.predict(input_df)
            st.write(f"Decision Tree Prediction: {data[target][dt_prediction[0]]}")
        if use_svm:
            svm_prediction = svm_model.predict(input_df)
            st.write(f"SVM Prediction: {data[target][svm_prediction[0]]}")
        if use_lr:
            lr_prediction = lr_model.predict(input_df)
            #st.write(f"Lr Prediction: {data[target][lr_prediction]}")
            st.write(f"Lr Prediction: {lr_prediction[0]:.2f}")
        if use_lor:
            lor_prediction = lor_model.predict(input_df)
            st.write(f"Lor Prediction: {data[target][lor_prediction[0]]}")

