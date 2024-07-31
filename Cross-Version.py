import tkinter as tk
import tensorflow as tf
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# TensorFlow and Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
import joblib

class DataAnalysisApp:
    sampling_methods = [
        "No sampling",
        "Over sampling (SMOTE)",
        "Under sampling",
        "SMOTE",
        "ADASYN",
        "SMOTE + ADASYN",
       # "Balanced Bagging Classifier",
        "Random Under-Sampling",
        "SMOTE + Tomek Links",
        "SMOTE + ENN",
        "Cluster Centroids"
    ]

    model_options = [
        "CNN + LSTM",
        "Random Forest + Gradient Boosting",
        "K-Means Clustering + Principal Component Analysis (PCA)",
        "Autoencoder + Support Vector Machine (SVM)",
        "Decision Tree + Logistic Regression",
        "Word2Vec + Recurrent Neural Network (RNN)",
        "Naive Bayes + k-Nearest Neighbors (k-NN)",
        #"Reinforcement Learning + Deep Q-Learning",
        "Gradient Boosting + Neural Network",
       # "Genetic Algorithm + Neural Network"
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis App")

        # Variables
        self.training_files = []
        self.common_features = []
        self.sampling_needed = tk.BooleanVar()
        self.sampling_option = tk.StringVar()
        self.sampling_option.set(self.sampling_methods[0])  # Set default value
        self.selected_model = tk.StringVar()
        self.selected_model.set(self.model_options[0])  # Set default value

        # Styling
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 10))
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TFrame", padding="10")
        style.configure("TEntry", font=("Helvetica", 10))
        style.configure("TCheckbutton", font=("Helvetica", 10))
        style.configure("TOptionMenu", font=("Helvetica", 10))

        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # UI Elements
        label_train_files = ttk.Label(main_frame, text="Upload CSV Files for Training:")
        button_browse_train = ttk.Button(main_frame, text="Browse", command=self.browse_train_files)

        check_sampling = ttk.Checkbutton(main_frame, text="Perform Sampling", variable=self.sampling_needed)
        option_sampling = ttk.OptionMenu(main_frame, self.sampling_option, *self.sampling_methods)

        label_model = ttk.Label(main_frame, text="Select Model:")
        option_model = ttk.OptionMenu(main_frame, self.selected_model, *self.model_options)

        button_train = ttk.Button(main_frame, text="Train Model", command=self.train_model)

        # Layout
        label_train_files.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        button_browse_train.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        check_sampling.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        option_sampling.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
        label_model.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
        option_model.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
        button_train.grid(row=3, column=0, columnspan=2, padx=10, pady=20)
        button_predict = ttk.Button(main_frame, text="Predict on New Data", command=self.predict_faults)
        button_predict.grid(row=4, column=0, columnspan=2, padx=10, pady=20)

    def browse_train_files(self):
        self.training_files = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])
        if self.training_files:
            messagebox.showinfo("Files Selected", f"Selected training files: {self.training_files}")
            self.process_data()

    def process_data(self):
        if not self.training_files:
            messagebox.showerror("Error", "Please select CSV files for training.")
            return

        try:
            # Load training data
            dfs_train = [pd.read_csv(file) for file in self.training_files]

            # Ensure all dataframes have the 'faults' column
            if not all('faults' in df.columns for df in dfs_train):
                messagebox.showerror("Error", "All datasets must contain a 'faults' column.")
                return

            # Merge training data
            merged_train_df = pd.concat(dfs_train, ignore_index=True)

            # Handle missing and null values
            merged_train_df = self.handle_missing_values(merged_train_df)

            # Separate features and target
            self.X_train = merged_train_df.drop('faults', axis=1)
            self.y_train = merged_train_df['faults']

            self.feature_names = list(self.X_train.columns)

            messagebox.showinfo("Data Processed", f"Data has been processed.\n"
                                                  f"Number of samples: {len(self.X_train)}\n"
                                                  f"Number of features: {len(self.feature_names)}")

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred while processing data: {str(e)}")

    def handle_missing_values(self, df):
        # Check for missing values
        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()

        if total_missing > 0:
            messagebox.showinfo("Missing Values", f"Total missing values: {total_missing}\n\nMissing values by column:\n{missing_values}")

            # Handle numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            imputer_numeric = SimpleImputer(strategy='mean')
            df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])

            # Handle categorical columns
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            df[categorical_columns] = imputer_categorical.fit_transform(df[categorical_columns])

            messagebox.showinfo("Missing Values Handled", "Missing values have been imputed.")
        else:
            messagebox.showinfo("Data Quality", "No missing values found in the dataset.")

        return df

    def apply_sampling(self, X, y):
        sampling_option = self.sampling_option.get()
        if sampling_option == "Over sampling (SMOTE)":
            sampler = SMOTE(random_state=42)
        elif sampling_option == "Under sampling":
            sampler = RandomUnderSampler(random_state=42)
        elif sampling_option == "SMOTE":
            sampler = SMOTE(random_state=42)
        elif sampling_option == "ADASYN":
            sampler = ADASYN(random_state=42)
        elif sampling_option == "SMOTE + ADASYN":
            sampler = SMOTEENN(random_state=42)
        #elif sampling_option == "Balanced Bagging Classifier":
            # This is handled differently as it's a classifier, not just a sampler
           # model = BalancedRandomForestClassifier(random_state=42)
           # model.fit(X, y)
           # return X, y  # Return original data as this is a complete model
        elif sampling_option == "Random Under-Sampling":
            sampler = RandomUnderSampler(random_state=42)
        elif sampling_option == "SMOTE + Tomek Links":
            sampler = SMOTETomek(random_state=42)
        elif sampling_option == "SMOTE + ENN":
            sampler = SMOTEENN(random_state=42)
        elif sampling_option == "Cluster Centroids":
            sampler = ClusterCentroids(random_state=42)
        else:  # "No sampling"
            return X, y

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled

    def train_model(self):
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            messagebox.showerror("Error", "Training data is not available. Please process the data first.")
            return

        sampling_option = self.sampling_option.get()
        if self.sampling_needed.get() and sampling_option != "No sampling":
            X_resampled, y_resampled = self.apply_sampling(self.X_train, self.y_train)
        else:
            X_resampled, y_resampled = self.X_train, self.y_train

        model_option = self.selected_model.get()
        model = None  # Initialize model with a default value

        try:
            if model_option == "CNN + LSTM":
                model = Sequential([
                    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_resampled.shape[1], 1)),
                    LSTM(50, activation='relu'),
                    Flatten(),
                    Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_resampled, y_resampled, epochs=10, batch_size=32, verbose=1)
            elif model_option == "Random Forest + Gradient Boosting":
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_resampled, y_resampled)
                gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
                gb.fit(X_resampled, y_resampled)
                model = (rf, gb)
            elif model_option == "K-Means Clustering + Principal Component Analysis (PCA)":
                kmeans = KMeans(n_clusters=3, random_state=42)
                kmeans.fit(X_resampled)
                pca = PCA(n_components=2)
                pca.fit(X_resampled)
                model = (kmeans, pca)
            elif model_option == "Autoencoder + Support Vector Machine (SVM)":
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_resampled.shape[1],)),
                    Dense(32, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(X_resampled.shape[1], activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_resampled, X_resampled, epochs=10, batch_size=32, verbose=1)
                svm = SVC(kernel='linear', random_state=42)
                svm.fit(X_resampled, y_resampled)
                model = (model, svm)
            elif model_option == "Decision Tree + Logistic Regression":
                dt = DecisionTreeClassifier(random_state=42)
                dt.fit(X_resampled, y_resampled)
                lr = LogisticRegression(random_state=42)
                lr.fit(X_resampled, y_resampled)
                model = (dt, lr)
            elif model_option == "Word2Vec + Recurrent Neural Network (RNN)":
                rnn_model = Sequential([
                    LSTM(50, activation='relu', input_shape=(X_resampled.shape[1], 1)),
                    Dense(1, activation='sigmoid')
                ])
                rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                rnn_model.fit(X_resampled, y_resampled, epochs=10, batch_size=32, verbose=1)
                model = rnn_model
            elif model_option == "Naive Bayes + k-Nearest Neighbors (k-NN)":
                nb = GaussianNB()
                nb.fit(X_resampled, y_resampled)
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_resampled, y_resampled)
                model = (nb, knn)
           # elif model_option == "Reinforcement Learning + Deep Q-Learning":
                # Placeholder for RL and DQN implementation
              #  model = "Reinforcement Learning + Deep Q-Learning model"
            elif model_option == "Gradient Boosting + Neural Network":
                gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
                gb.fit(X_resampled, y_resampled)
                nn_model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_resampled.shape[1],)),
                    Dense(1, activation='sigmoid')
                ])
                nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                nn_model.fit(X_resampled, y_resampled, epochs=10, batch_size=32, verbose=1)
                model = (gb, nn_model)
           # elif model_option == "Genetic Algorithm + Neural Network":
                # Placeholder for Genetic Algorithm and Neural Network implementation
              #  model = "Genetic Algorithm + Neural Network model"
            else:
                messagebox.showerror("Error", f"Unknown model option: {model_option}")
                return

            # Save the model
            if model_option == "CNN + LSTM" or model_option == "Word2Vec + Recurrent Neural Network (RNN)":
                model.save('model.h5')
            else:
                joblib.dump(model, 'model.pkl')

            messagebox.showinfo("Model Training", "Model has been trained and saved successfully.")

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred during model training: {str(e)}")

    def predict_faults(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
            if not file_path:
                return

            new_data = pd.read_csv(file_path)
            new_data = self.handle_missing_values(new_data)
            if hasattr(self, 'feature_names') and self.feature_names:
                new_data = new_data[self.feature_names]

            if self.selected_model.get() == "CNN + LSTM" or self.selected_model.get() == "Word2Vec + Recurrent Neural Network (RNN)":
                model = tf.keras.models.load_model('model.h5')
                new_data = np.expand_dims(new_data, axis=-1)  # Add dimension for LSTM/CNN
                predictions = model.predict(new_data)
            else:
                model = joblib.load('model.pkl')
                if self.selected_model.get() == "Random Forest + Gradient Boosting":
                    rf, gb = model
                    predictions_rf = rf.predict(new_data)
                    predictions_gb = gb.predict(new_data)
                    predictions = (predictions_rf + predictions_gb) / 2
                elif self.selected_model.get() == "K-Means Clustering + Principal Component Analysis (PCA)":
                    kmeans, pca = model
                    new_data_pca = pca.transform(new_data)
                    predictions = kmeans.predict(new_data_pca)
                elif self.selected_model.get() == "Autoencoder + Support Vector Machine (SVM)":
                    ae, svm = model
                    encoded_data = ae.predict(new_data)
                    predictions = svm.predict(encoded_data)
                elif self.selected_model.get() == "Decision Tree + Logistic Regression":
                    dt, lr = model
                    predictions_dt = dt.predict(new_data)
                    predictions_lr = lr.predict(new_data)
                    predictions = (predictions_dt + predictions_lr) / 2
                elif self.selected_model.get() == "Naive Bayes + k-Nearest Neighbors (k-NN)":
                    nb, knn = model
                    predictions_nb = nb.predict(new_data)
                    predictions_knn = knn.predict(new_data)
                    predictions = (predictions_nb + predictions_knn) / 2
                elif self.selected_model.get() == "Gradient Boosting + Neural Network":
                    gb, nn = model
                    predictions_gb = gb.predict(new_data)
                    predictions_nn = nn.predict(new_data)
                    predictions = (predictions_gb + predictions_nn) / 2
                else:
                    predictions = model.predict(new_data)

            # Show predictions
            results = pd.DataFrame({'Predictions': predictions.flatten()})
            results_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if results_file:
                results.to_csv(results_file, index=False)
                messagebox.showinfo("Prediction Results", f"Predictions saved to: {results_file}")

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()
