# TypeGuardML - Enhanced Behavioral Biometrics

Behavioral biometrics is an advanced domain focusing on scrutinizing user behavior to bolster security and authentication protocols. This repository acts as a central hub for all things related to the groundbreaking technology of TypeGuardML.

## Keystroke Biometrics Authentication

TypeGuardML introduces a Python script designed to capture keystroke dynamics, extract pertinent features, and train a machine-learning model for user authentication based on keystroke behavior. It presents a streamlined demonstration of behavioral biometrics.

### Features

- **Data Collection:** Records keystroke events and their timestamps for a specified duration (default: 10 seconds).

- **Feature Extraction:** Computes key hold time, key release time, and typing speed from the gathered keystrokes.

- **Model Training:** Deploys a Random Forest classifier to train on keystroke features and associated user labels.

- **Authentication:** Verifies a user's identity based on their keystroke features using the trained model.

### Prerequisites

- Python 3.x
- Required libraries: keyboard, numpy, scikit-learn
- Install the necessary libraries using `pip install keyboard numpy scikit-learn`.

### Usage

1. Run the script by executing `python TypeGuardML_Authentication.py`.

2. Follow the prompts and type to capture keystroke data.

3. The script simulates two users (user1 and user2) with random labels for demonstration. Substitute this simulated data with actual user data for authenticating real users.

4. The machine learning model is trained on the simulated data.

5. Replace the simulated user keystroke features with actual user data for authentication. The predicted user label is printed.

### Notes

- This code serves as a simplified demonstration and should be extended for real-world applications with authentic user data.

- Modify the simulated user data and labels to facilitate the authentication of real users.

- Ensure the user data aligns with the feature extraction process.

### License

This project is licensed under the MIT License. Refer to the [LICENSE](LICENSE) for details.

