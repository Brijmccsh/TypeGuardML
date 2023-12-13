
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import keyboard
import time

# Function to record user interactions for a specific duration
def record_user_interaction(duration):
    tapping_rhythm = []
    tapping_pressure = []
    swiping_speed = []
    swiping_length = []
    typing_speed = []
    navigation_patterns = []
    clicking_behavior = []
    scrolling_behavior = []
    time_spent_on_tasks = []

    start_time = time.time()

    def on_press(key):
        nonlocal tapping_rhythm, tapping_pressure, typing_speed
        # Record tapping features
        tapping_rhythm.append(time.time() - start_time)
        tapping_pressure.append(np.random.uniform(0.5, 1.0))

    def on_scroll(x, y, dx, dy):
        nonlocal scrolling_behavior
        # Record scrolling behavior
        scrolling_behavior.append((dx, dy))

    def on_move(x, y):
        nonlocal swiping_speed, swiping_length
        # Record swiping features
        swiping_speed.append(np.random.uniform(0.1, 0.5))
        swiping_length.append(np.random.uniform(1.0, 3.0))

    def on_release(key):
        nonlocal typing_speed, clicking_behavior, navigation_patterns, time_spent_on_tasks
        # Record typing features
        typing_speed.append(np.random.uniform(0.3, 0.8))
        # Record clicking behavior
        clicking_behavior.append((np.random.uniform(0, 1920), np.random.uniform(0, 1080)))
        # Record navigation patterns
        navigation_patterns.append(np.random.uniform(0.1, 1.0))
        # Record time spent on tasks
        time_spent_on_tasks.append(time.time() - start_time)

    # Set up listeners
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener, \
         keyboard.Listener(on_move=on_move, on_scroll=on_scroll) as mouse_listener:

        # Run the listeners for the specified duration
        while time.time() - start_time < duration:
            pass

    # Return recorded features
    return {
        'Tapping_Rhythm': tapping_rhythm,
        'Tapping_Pressure': tapping_pressure,
        'Swiping_Speed': swiping_speed,
        'Swiping_Length': swiping_length,
        'Typing_Speed': typing_speed,
        'Navigation_Patterns': navigation_patterns,
        'Clicking_Behavior': clicking_behavior,
        'Scrolling_Behavior': scrolling_behavior,
        'Time_Spent_On_Tasks': time_spent_on_tasks
    }

# Function to generate synthetic dataset for testing
def generate_synthetic_dataset(num_samples=1000):
    data = {'Tapping_Rhythm': [], 'Tapping_Pressure': [],
            'Swiping_Speed': [], 'Swiping_Length': [],
            'Typing_Speed': [], 'Navigation_Patterns': [],
            'Clicking_Behavior': [], 'Scrolling_Behavior': [],
            'Time_Spent_On_Tasks': [], 'User_Label': []}

    # Generate synthetic data for genuine users
    for _ in range(num_samples):
        user_data = record_user_interaction(duration=10)
        for feature, values in user_data.items():
            data[feature].extend(values)

        # Label the data as genuine user (1)
        data['User_Label'].extend([1])

    # Generate synthetic data for impostors
    for _ in range(num_samples):
        impostor_data = record_user_interaction(duration=10)
        for feature, values in impostor_data.items():
            data[feature].extend(values)

        # Label the data as impostor (0)
        data['User_Label'].extend([0])

    return pd.DataFrame(data)

# Function to calculate features
def calculate_features(recorded_data):
    # For simplicity, just return the recorded data for now
    return recorded_data

# Function to train the model
def train_model(X_train, y_train):
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)

    # Train the model
    model.fit(X_train_scaled, y_train)

    return model, scaler

# Function to authenticate the user
def authenticate_user(model, scaler, recorded_data):
    # Calculate features from recorded data
    features = calculate_features(recorded_data)

    # Standardize the features
    features_scaled = scaler.transform(features)

    # Make predictions using the trained model
    prediction = model.predict(features_scaled)

    return prediction[0]

# Function to visualize label distribution
def visualize_label_distribution(y):
    sns.countplot(x=y)
    plt.title('Label Distribution')
    plt.show()

# Main function
def main():
    # Generate a synthetic dataset for testing
    synthetic_df = generate_synthetic_dataset(num_samples=500)

    # Extract features (X) and labels (y)
    X = synthetic_df.drop('User_Label', axis=1)
    y = synthetic_df['User_Label']

    # Split the dataset into training and testing sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model, scaler = train_model(X_train, y_train)

    # Visualize label distribution
    visualize_label_distribution(y)

    # Record data for a genuine user
    genuine_user_data = record_user_interaction(duration=10)

    # Authenticate the genuine user
    prediction_genuine_user = authenticate_user(model, scaler, genuine_user_data)
    print(f'Prediction for genuine user: {prediction_genuine_user}')

if __name__ == "__main__":
    main()
