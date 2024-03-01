import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load your dataset and define the mappings here...
df=pd.read_csv('./new_Data.csv')
encoder = OneHotEncoder(sparse_output=False)

# Fit and transform the 'brand' column
brand_encoded = encoder.fit_transform(df[['Brand']])

# Create a new DataFrame with one-hot encoded columns
brand_encoded_df = pd.DataFrame(encoder.fit_transform(df[['Brand']]), columns=encoder.get_feature_names_out())

# Concatenate the one-hot encoded DataFrame with the original DataFrame
df = pd.concat([df, brand_encoded_df], axis=1)

df = df.drop(['PhoneLink','Color','Size','Processor','Camera','FrontCam','Network','Battery','Brand'], axis=1)



st.title("Mobile Price Prediction")

st.sidebar.header("Input Features")

# Create input fields for user data


# Load your mappings for preprocessing...
processor_mapping = {'MediaTek': 0, 'Dimensity': 1, 'Unisoc': 2, 'Snapdragon': 3,  'Exynos': 4}

net_mapping = {'4G' : 0, '5G': 1 }

storage_mapping = {64: 0, 128: 1, 256: 2}

ram_mapping = {4: 0, 8: 1, 12: 2}

backCam_mapping = {8: 0, 12: 0, 13: 0, 16: 0, 48: 1, 50: 1, 64: 1, 100: 2, 108: 2, 200: 2}

frontCam_mapping = {5: 0, 8: 0, 10: 0, 13: 1, 16: 1, 20: 1, 32: 2, 50: 2}

df['Power'] = df['Power'] / 1000

df['DcPrice'] = df['DcPrice'] / 1000

df['ProcessorNew'] = df['ProcessorNew'].replace(processor_mapping)

df['Net'] = df['Net'].replace(net_mapping)

df['Storage'] = df['Storage'].replace(storage_mapping)

df['RAM'] = df['RAM'].replace(ram_mapping)

df['BackCamera'] = df['BackCamera'].replace(backCam_mapping)

df['FrontCamera'] = df['FrontCamera'].replace(frontCam_mapping)

ram = st.sidebar.selectbox("Select RAM (GB)", [4, 8, 12])
storage = st.sidebar.selectbox("Storage (GB)", [64, 128, 256])
processor = st.sidebar.selectbox("Processor", ('MediaTek', 'Dimensity', 'Unisoc', 'Snapdragon', 'Exynos'))
new_size = st.sidebar.slider("New Size (inches)", 4.0, 7.0, 6.5)
back_camera = st.sidebar.selectbox("Back Camera (MP)", (8, 12, 13, 16, 48, 50, 64, 100, 108, 200))
front_camera = st.sidebar.selectbox("Front Camera (MP)", (5, 8, 10, 13, 16, 20, 32, 50))
net = st.sidebar.selectbox("Network", ('4G', '5G'))
power = st.sidebar.slider("Battery Capacity (mAh)", 1000, 10000, 5000)
brand_oppo = st.sidebar.checkbox("OPPO")
brand_poco = st.sidebar.checkbox("POCO")
brand_realme = st.sidebar.checkbox("Realme")
brand_redmi = st.sidebar.checkbox("Redmi")
brand_samsung = st.sidebar.checkbox("Samsung")
brand_vivo = st.sidebar.checkbox("Vivo")
quick_charge = st.sidebar.checkbox("Quick Charge")

# Preprocess the user input
user_input = {
    'RAM': [ram],
    'Storage': [storage],
    'ProcessorNew': [processor_mapping[processor]],
    'NewSize': [new_size],
    'BackCamera': [backCam_mapping[back_camera]],
    'FrontCamera': [frontCam_mapping[front_camera]],
    'Net': [net_mapping[net]],
    'Power': [power / 1000],
    'Brand_OPPO': [int(brand_oppo)],
    'Brand_POCO': [int(brand_poco)],
    'Brand_REALME': [int(brand_realme)],
    'Brand_REDMI': [int(brand_redmi)],
    'Brand_SAMSUNG': [int(brand_samsung)],
    'Brand_VIVO': [int(brand_vivo)],
    'QuickCharge': [int(quick_charge)]
}

user_input_df = pd.DataFrame(user_input)

# Load your preprocessed dataset (df) and train your RandomForestRegressor (rf_regressor)...
X = df[['RAM', 'Storage', 'ProcessorNew', 'NewSize', 'BackCamera', 'FrontCamera', 'Net', 'Power', 'Brand_OPPO', 'Brand_POCO', 'Brand_REALME', 'Brand_REDMI', 'Brand_SAMSUNG', 'Brand_VIVO', 'QuickCharge' ]]

y = df['DcPrice']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=80, random_state=42, max_depth=11)
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)

if st.button("Predict"):
    answer = rf_regressor.predict(user_input_df)
    ans=int(answer[0]*1000)
    st.write(f"Predicted Price: ₹ {ans}")

st.sidebar.subheader("Model Evaluation")
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
score=rf_regressor.score(X_test,y_test)

st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R-squared (R2): {r2}")
st.write(score)

