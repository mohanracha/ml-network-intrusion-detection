import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import warnings
import io

warnings.filterwarnings("ignore")

# -----------------------------
# STREAMLIT APP TITLE
# -----------------------------
st.title("🚀 Network Intrusion Detection using Machine Learning")
st.write("This app uses a **Random Forest model** with oversampling to detect intrusions from the NSL-KDD dataset.")

# -----------------------------
# FILE UPLOAD
# -----------------------------
st.sidebar.title("Upload Datasets")
uploaded_train = st.sidebar.file_uploader("Upload KDDTrain.txt", type=["txt", "csv"])
uploaded_test = st.sidebar.file_uploader("Upload KDDTest.txt", type=["txt", "csv"])

# -----------------------------
# CACHE: PREPROCESSING FUNCTION
# -----------------------------
@st.cache_data
def preprocess_data(train_file, test_file):
    datacols = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"
    ]

    # Load datasets
    train_file.seek(0)
    test_file.seek(0)
    df_train = pd.read_csv(train_file, sep=",", names=datacols, engine="python")
    df_test = pd.read_csv(test_file, sep=",", names=datacols, engine="python")

    df_train = df_train.iloc[:, :-1]
    df_test = df_test.iloc[:, :-1]

    # Attack mapping
    attack_mapping = {
        'ipsweep': 'Probe', 'satan': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'saint': 'Probe', 'mscan': 'Probe',
        'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS', 'back': 'DoS', 'neptune': 'DoS', 'smurf': 'DoS', 'mailbomb': 'DoS',
        'udpstorm': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
        'perl': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'buffer_overflow': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
        'sqlattack': 'U2R', 'httptunnel': 'U2R',
        'ftp_write': 'R2L', 'phf': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L',
        'imap': 'R2L', 'spy': 'R2L', 'multihop': 'R2L', 'named': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L',
        'snmpgetattack': 'R2L', 'xsnoop': 'R2L', 'xlock': 'R2L', 'sendmail': 'R2L',
        'normal': 'Normal'
    }

    df_train['attack_class'] = df_train['attack'].map(attack_mapping)
    df_test['attack_class'] = df_test['attack'].map(attack_mapping)

    df_train.drop('attack', axis=1, inplace=True)
    df_test.drop('attack', axis=1, inplace=True)

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    # Scale numeric features
    num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols] = scaler.transform(df_test[num_cols])

    # Encode categorical features
    cat_cols = df_train.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in cat_cols:
        df_train[col] = encoder.fit_transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col])

    X_train = df_train.drop('attack_class', axis=1)
    y_train = df_train['attack_class']
    X_test = df_test.drop('attack_class', axis=1)
    y_test = df_test['attack_class']

    # Oversampling
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    return X_res, y_res, X_train, y_train, X_test, y_test

# -----------------------------
# CACHE: MODEL TRAINING
# -----------------------------
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# -----------------------------
# MAIN APP LOGIC
# -----------------------------
if uploaded_train and uploaded_test:
    X_res, y_res, X_train, y_train, X_test, y_test = preprocess_data(uploaded_train, uploaded_test)
    model = train_model(X_res, y_res)

    st.sidebar.success("✅ Data successfully processed!")

    # Sidebar options
    option = st.sidebar.radio("Choose an action:", 
                              ["Show Dataset Info", "Show Feature Importances", "Evaluate Model", "Test Model"])

    if option == "Show Dataset Info":
        st.subheader("Class Distribution")
        st.write(f"Original Dataset: {Counter(y_train)}")
        st.write(f"Resampled Dataset: {Counter(y_res)}")

    elif option == "Show Feature Importances":
        st.subheader("Feature Importances")
        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax)
        st.pyplot(fig)

    elif option == "Evaluate Model":
        st.subheader("Training Evaluation")
        y_pred_train = model.predict(X_train)
        st.write("✅ **Training Accuracy:**", accuracy_score(y_train, y_pred_train))

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, ax=ax, cmap="Blues")
        st.pyplot(fig)

        st.text(classification_report(y_train, y_pred_train))

    elif option == "Test Model":
        st.subheader("Testing Evaluation")
        y_pred_test = model.predict(X_test)
        st.write("✅ **Testing Accuracy:**", accuracy_score(y_test, y_pred_test))

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=ax, cmap="Greens")
        st.pyplot(fig)

        st.text(classification_report(y_test, y_pred_test))

        # Download predictions
        csv = X_test.copy()
        csv["Actual"] = y_test
        csv["Predicted"] = y_pred_test
        csv_file = io.BytesIO()
        csv.to_csv(csv_file, index=False)
        st.download_button("📥 Download Predictions", data=csv_file.getvalue(),
                           file_name="predictions.csv", mime="text/csv")
else:
    st.info("👆 Please upload both train and test files to continue.")
