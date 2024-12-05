import streamlit as st
import pandas as pd
import joblib
import numpy as np
from utils import preprocess_data, engineer_features

# Load model and components
@st.cache_resource
def load_model():
    model = joblib.load('model/flight_delay_model.pkl')
    encoders = joblib.load('model/feature_encoders.pkl')
    threshold = joblib.load('model/optimal_threshold.pkl')
    return model, encoders, threshold

def main():
    st.title('Flight Delay Predictor')
    st.write('Predict if your flight will be delayed by 15+ minutes')
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        month = st.selectbox('Month', range(1, 13))
        day = st.selectbox('Day of Month', range(1, 32))
        day_of_week = st.selectbox('Day of Week', range(1, 8))
        dep_time = st.number_input('Departure Time (HHMM)', 0, 2359)
        
    with col2:
        carrier = st.selectbox('Airline Carrier', 
            ['AA', 'UA', 'DL', 'WN', 'AS', 'B6', 'NK', 'F9'])
        origin = st.text_input('Origin Airport (3-letter code)', max_chars=3).upper()
        dest = st.text_input('Destination Airport (3-letter code)', max_chars=3).upper()
        distance = st.number_input('Flight Distance (miles)', 0)

    if st.button('Predict Delay Probability'):
        try:
            # Load model components
            model, encoders, threshold = load_model()
            
            # Create input dataframe
            input_data = {
                'Month': [month],
                'DayofMonth': [day],
                'DayOfWeek': [day_of_week],
                'DepTime': [dep_time],
                'UniqueCarrier': [carrier],
                'Origin': [origin],
                'Dest': [dest],
                'Distance': [distance]
            }
            input_df = pd.DataFrame(input_data)
            
            # Preprocess and engineer features
            processed_df = preprocess_data(input_df)
            featured_df = engineer_features(processed_df)
            
            # Make prediction
            prediction = model.predict(featured_df)[0]
            
            # Display result
            st.subheader('Prediction Result')
            probability = prediction * 100
            
            if prediction > threshold:
                st.error(f'⚠️ High chance of delay: {probability:.1f}%')
            else:
                st.success(f'✈️ Low chance of delay: {probability:.1f}%')
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure all inputs are valid.")

if __name__ == "__main__":
    main()
