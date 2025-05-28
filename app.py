import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

st.title('Model Klasifikasi Wine Quality')
st.write('Aplikasi untuk memprediksi kualitas wine berdasarkan fitur yang ada.')

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    X = data.drop('quality', axis=1)
    y = data['quality']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=5)
    st.write(f'Mean Cross-validation score: {cv_scores.mean()}')

    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    st.write("Feature Importance:")
    st.dataframe(feature_importance_df)

    if st.button("Prediksi"):
        predictions = model.predict(X)
        st.write("Hasil Prediksi:")
        st.dataframe(predictions)