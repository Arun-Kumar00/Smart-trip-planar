import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import itertools

# ------------------------
# 1. Generate or load data
# ------------------------
def generate_travel_data(file_name='travel_price_data.csv', num_samples=100000):
    if os.path.exists(file_name):
        print(f"Loading data from {file_name}")
        return pd.read_csv(file_name)

    np.random.seed(42)
    destinations = ['Paris','Tokyo','New York','Sydney','Goa','London','Rome','Dubai','Kyoto','Amsterdam']
    months = list(range(1,13))
    airlines = ['Delta','United','Air France','Qantas','JAL']
    hotel_types = ['Budget','Mid-range','Luxury']
    days_of_week = list(range(7))
    data = {
        'origin_city': np.random.choice(destinations, num_samples),
        'destination_city': np.random.choice(destinations, num_samples),
        'travel_month': np.random.choice(months, num_samples),
        'travel_day_of_week': np.random.choice(days_of_week, num_samples),
        'airline': np.random.choice(airlines, num_samples),
        'hotel_type': np.random.choice(hotel_types, num_samples),
        'days_in_advance_booked': np.random.randint(1,365, num_samples),
        'num_stops': np.random.choice([0,1,2], num_samples),
        'base_price_usd': np.random.randint(150,1500,num_samples)
    }
    df = pd.DataFrame(data)

    # Create realistic price
    df['is_peak_season'] = df['travel_month'].apply(lambda m: 1 if m in [5,6,7,8] else 0)
    df['price'] = (df['base_price_usd'] *
                   (1 + 0.1*df['is_peak_season']) *
                   (1 + 0.05*df['num_stops']) +
                   np.random.normal(0,50,num_samples)).round(2)
    df.to_csv(file_name,index=False)
    return df

# ------------------------
# 2. Preprocess
# ------------------------
def preprocess_data(df):
    cat_features = ['origin_city','destination_city','airline','hotel_type']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(df[cat_features])
    X_num = df[['travel_month','travel_day_of_week','days_in_advance_booked','num_stops','is_peak_season']].values
    X = np.hstack([X_cat,X_num])
    y = df['price'].values
    return X, y, encoder

# ------------------------
# 3. Train model
# ------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train,y_train)
    print("Model trained successfully.")
    return model

# ------------------------
# 4. Recommendation function
# ------------------------
def recommend_trip(model, encoder, origin_city=None, destination_city=None, budget=1000):
    """
    If destination_city is provided, recommend month, airline, hotel_type
    If destination_city is None, recommend possible destination cities + airline + hotel
    """
    months = list(range(1,13))
    airlines = ['Delta','United','Air France','Qantas','JAL']
    hotel_types = ['Budget','Mid-range','Luxury']
    stops = [0,1,2]
    day_of_week = 3  # default

    candidate_list = []

    destinations = [destination_city] if destination_city else ['Paris','Tokyo','New York','Sydney','Goa','London','Rome','Dubai','Kyoto','Amsterdam']

    for dest, month, airline, hotel, stop in itertools.product(destinations, months, airlines, hotel_types, stops):
        df_input = pd.DataFrame([{
            'origin_city': origin_city if origin_city else 'Paris',
            'destination_city': dest,
            'airline': airline,
            'hotel_type': hotel
        }])
        X_cat = encoder.transform(df_input)
        X_num = np.array([[month, day_of_week, 60, stop, 1 if month in [5,6,7,8] else 0]])
        X_input = np.hstack([X_cat, X_num])
        price = model.predict(X_input)[0]
        if price <= budget:
            candidate_list.append({
                'destination': dest,
                'month': month,
                'airline': airline,
                'hotel_type': hotel,
                'num_stops': stop,
                'predicted_price': round(price,2)
            })

    if not candidate_list:
        return "No options found within budget"
    candidate_list_sorted = sorted(candidate_list, key=lambda x: -x['predicted_price'])
    return candidate_list_sorted[:5]

# ------------------------
# 5. Main
# ------------------------
if __name__ == "__main__":
    df = generate_travel_data()
    X, y, encoder = preprocess_data(df)
    model = train_model(X,y)
    joblib.dump(model,'trip_price_model.pkl')
    joblib.dump(encoder,'encoder.pkl')
    print("Model and encoder saved.")

    # Example 1: Specific destination
    print("\n🔹 Trip Recommendation: Tokyo, budget 1000")
    rec1 = recommend_trip(model, encoder, origin_city='Dubai', destination_city='Tokyo', budget=1000)
    for r in rec1:
        print(r)

    # Example 2: Find destinations within budget
    print("\n🔹 Destination Suggestions from Goa, budget 1000")
    rec2 = recommend_trip(model, encoder, origin_city='Goa', destination_city=None, budget=1000)
    for r in rec2:
        print(r)




