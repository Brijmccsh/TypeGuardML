import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keyboard
import matplotlib.pyplot as plt

# Constants for keystroke features
HOLD_DURATION = 'hold_duration'
RELEASE_DURATION = 'release_duration'
TYPING_SPEED = 'typing_speed'

# Function to capture keystroke data for a specified duration
def record_keystroke_data(duration=10):
    print("Recording keystroke data for the user...")
    keystrokes = []
    start_time = time.time()

    while time.time() - start_time < duration:
        event = keyboard.read_event(suppress=True)
        timestamp = time.time()
        keystrokes.append((event, timestamp))

    print(f"Number of keystrokes recorded: {len(keystrokes)}")
    return keystrokes

# Function to extract features from keystroke data
def calculate_features(keystrokes):
    features = []
    for i in range(1, len(keystrokes)):
        prev_event, prev_timestamp = keystrokes[i - 1]
        event, timestamp = keystrokes[i]

        if event.event_type == keyboard.KEY_DOWN and prev_event.event_type == keyboard.KEY_DOWN:
            hold_duration = timestamp - prev_timestamp
            release_duration = 0
            typing_speed = 0
        elif event.event_type == keyboard.KEY_UP and prev_event.event_type == keyboard.KEY_DOWN:
            hold_duration = 0
            release_duration = timestamp - prev_timestamp
            typing_speed = 1 / (timestamp - prev_timestamp)
        else:
            continue

        features.append({
            HOLD_DURATION: hold_duration,
            RELEASE_DURATION: release_duration,
            TYPING_SPEED: typing_speed
        })

    return features

# Function to train a machine learning model
def train_model(features, labels):
    X = np.array([[f[HOLD_DURATION], f[RELEASE_DURATION], f[TYPING_SPEED]] for f in features])

    label_to_numeric = {label: i for i, label in enumerate(set(labels))}
    y = np.array([label_to_numeric[label] for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy:", accuracy)

    # Display distribution of user labels in training data
    visualize_label_distribution(labels)

    return model

# Function to authenticate a user using the trained model
def authenticate_user(model, keystroke_features):
    features = np.array([[kf[HOLD_DURATION], kf[RELEASE_DURATION], kf[TYPING_SPEED]] for kf in keystroke_features])
    prediction = model.predict(features)
    return prediction[0]

# Function to plot the distribution of user labels
def visualize_label_distribution(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.bar(unique_labels, counts, color=['blue', 'green'])
    plt.xlabel('User Labels')
    plt.ylabel('Count')
    plt.title('Distribution of User Labels in Training Data')
    plt.show()

# Main function
def main():
    keystrokes = record_keystroke_data(duration=10)
    features = calculate_features(keystrokes)

    labels = ['user1', 'user2']
    user_labels = np.random.choice(labels, len(features))
    
    model = train_model(features, user_labels)

    user_keystroke_features = [{HOLD_DURATION: 0, RELEASE_DURATION: 0.05, TYPING_SPEED: 9.13}]
    user = authenticate_user(model, user_keystroke_features)
    print("Authenticated User:", labels[user])

if __name__ == "__main__":
    main()
