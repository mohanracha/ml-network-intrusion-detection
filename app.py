import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Streamlit app title
st.title("Network Intrusion Detection using Machine Learning")

# File upload
st.subheader("Upload NSL-KDD Dataset")
uploaded_train = st.file_uploader("Upload KDDTrain.txt", type=["txt"])
uploaded_test = st.file_uploader("Upload KDDTest.txt", type=["txt"])

if uploaded_train and uploaded_test:
    # Dataset field names
    datacols = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"]

    # Load datasets
    dfkdd_train = pd.read_csv(uploaded_train, sep=",", names=datacols)
    dfkdd_train = dfkdd_train.iloc[:, :-1]

    dfkdd_test = pd.read_csv(uploaded_test, sep=",", names=datacols)
    dfkdd_test = dfkdd_test.iloc[:, :-1]

    # Map attack types to classes
    attack_mapping = {
        'ipsweep': 'Probe', 'satan': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'saint': 'Probe', 'mscan': 'Probe',
        'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS', 'back': 'DoS', 'neptune': 'DoS', 'smurf': 'DoS', 'mailbomb': 'DoS',
        'udpstorm': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
        'perl': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'buffer_overflow': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
        'sqlattack': 'U2R', 'httptunnel': 'U2R',
        'ftp_write': 'R2L', 'phf': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'imap': 'R2L',
        'spy': 'R2L', 'multihop': 'R2L', 'named': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L', 'snmpgetattack': 'R2L',
        'xsnoop': 'R2L', 'xlock': 'R2L', 'sendmail': 'R2L',
        'normal': 'Normal'
    }

    # Apply attack class mapping
    dfkdd_train['attack_class'] = dfkdd_train['attack'].map(attack_mapping)
    dfkdd_test['attack_class'] = dfkdd_test['attack'].map(attack_mapping)

    # Drop original attack column
    dfkdd_train.drop('attack', axis=1, inplace=True)
    dfkdd_test.drop('attack', axis=1, inplace=True)

    # Handle missing values if any
    dfkdd_train.dropna(inplace=True)
    dfkdd_test.dropna(inplace=True)

    # Scaling numerical columns
    num_cols = dfkdd_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    dfkdd_train[num_cols] = scaler.fit_transform(dfkdd_train[num_cols])
    dfkdd_test[num_cols] = scaler.transform(dfkdd_test[num_cols])

    # Encoding categorical columns
    cat_cols = dfkdd_train.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in cat_cols:
        dfkdd_train[col] = encoder.fit_transform(dfkdd_train[col])
        dfkdd_test[col] = encoder.transform(dfkdd_test[col])

    # Prepare data for modeling
    X_train = dfkdd_train.drop('attack_class', axis=1)
    y_train = dfkdd_train['attack_class']
    X_test = dfkdd_test.drop('attack_class', axis=1)
    y_test = dfkdd_test['attack_class']

    # Apply oversampling to handle class imbalance
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    st.write(f'Original dataset shape: {Counter(y_train)}')
    st.write(f'Resampled dataset shape: {Counter(y_res)}')

    # Train Random Forest Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)

    # Display feature importances
    if st.button("Show Feature Importances"):
        feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax)
        st.pyplot(fig)

    # Evaluate model
    if st.button("Evaluate Model"):
        y_pred_train = model.predict(X_train)
        st.write("Training Accuracy:", accuracy_score(y_train, y_pred_train))
        st.write("Confusion Matrix:", confusion_matrix(y_train, y_pred_train))
        st.text(classification_report(y_train, y_pred_train))

    # Test model
    if st.button("Test Model"):
        y_pred_test = model.predict(X_test)
        st.write("Testing Accuracy:", accuracy_score(y_test, y_pred_test))
        st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred_test))
        st.text(classification_report(y_test, y_pred_test))
