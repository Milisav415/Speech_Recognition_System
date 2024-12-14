import os
import librosa
import librosa.display
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from scipy.fft import fft, ifft
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer


def plot_audio_analysis(file_path):
    # Load the audio file
    path = 'C:/Users/jm190/Desktop/Diplomski/Baza 3.1 extended'
    audio, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')

    # Calculate the duration of the audio in seconds
    duration = float(len(audio)) / sr
    print(f'Duration for {file_path.replace(path, '')} is {duration}')

    # Time array for plotting the waveform
    time = np.linspace(0., duration, len(audio))

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.title(f"Waveform for {file_path.replace(path, '')}")
    plt.plot(time, audio)
    plt.ylabel(f"Amplitude")
    plt.xlabel("Time [s]")
    plt.show()

    # Plot the spectrum
    plt.figure(figsize=(10, 4))
    fft_spectrum = fft(audio)
    frequency = np.linspace(0, sr, len(fft_spectrum))
    plt.title(f"Spectrumfor {file_path.replace(path, '')}")
    plt.plot(frequency[:len(frequency) // 2], np.abs(fft_spectrum)[:len(frequency) // 2])  # Plot only the first half
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.show()

    # Plot the log of the spectrum magnitude
    plt.figure(figsize=(10, 4))
    plt.title(f"Log of Spectrum Magnitude for {file_path.replace(path, '')}")
    plt.plot(frequency[:len(frequency) // 2], 20 * np.log(np.abs(fft_spectrum)[:len(frequency) // 2]))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Log Magnitude")
    plt.show()

    # Plot the power cepstrum
    plt.figure(figsize=(10, 4))
    power_spectrum = np.abs(fft_spectrum) ** 2
    log_power_spectrum = np.log(power_spectrum)
    power_cepstrum = np.log(np.abs(ifft(log_power_spectrum)) ** 2)
    plt.title(f"Log Power Cepstrum for {file_path.replace(path, '')}")
    plt.plot(time, power_cepstrum)
    plt.xlabel("Quefrency [s]")
    plt.ylabel("Amplitude")
    plt.show()

    # Plot the Spectrogram
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

    # Calculate and plot the Log Mel-Spectrogram
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    # Convert to dB
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log Mel-Spectrogram')
    plt.show()

    # Plot the Mel-frequency cepstrum coefficients (MFCCs)
    plt.figure(figsize=(10, 4))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title(f"MFCC for {file_path.replace(path, '')}")
    plt.tight_layout()
    plt.show()


def plot_3d_points(features, num_classes=4):
    """
    This function plots points in 3D space with different colors for each of the 4 classes.
    The input is a matrix of dimensions 4*N by 3, where N is the number of samples per class.

    Parameters:
    - features: A numpy array of shape (4*N, 3), where each row represents a point in 3D space.

    Returns:
    - This function does not return a value. It displays a 3D scatter plot of the points with colors and legend.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y']  # Colors for each class
    class_names = ['kreni', 'stani', 'levo', 'desno']  # Names for each class

    N = features.shape[0] // num_classes  # Number of samples per class
    for i, color in enumerate(colors):
        class_points = features[i * N:(i + 1) * N, :]  # Extract points for each class
        ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2], c=color, label=class_names[i])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


# Example of augmenting MFCCs with deltas
def augment_features_with_deltas(features):
    delta_features = librosa.feature.delta(features)
    delta_delta_features = librosa.feature.delta(features, order=2)
    augmented_features = np.concatenate((features, delta_features, delta_delta_features), axis=0)
    return augmented_features


def plot_mean_spectogram(audio_path, N=10, M=10):
    """
    Plots the regular spectrogram of an audio file.

    Parameters:
        - file_path: Path to the audio file.
    """
    # Load the audio file
    signal, fs = librosa.load(audio_path, sr=None)  # sr=None to preserve the original sampling rate

    # Compute the Short-Time Fourier Transform (STFT)
    D = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Compute the spectrogram
    S = np.abs(librosa.stft(y))

    # Convert to dB
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    # Resize the spectrogram to the desired dimensions
    S_resized = librosa.resample(S_dB, orig_sr=S_dB.shape[1], target_sr=M, axis=1)
    S_resized_final = librosa.resample(S_resized, orig_sr=S_resized.shape[0], target_sr=N, axis=0)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_resized_final, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Usrednjeni spektogram')
    plt.ylabel('Frekventne komponente')
    plt.xlabel('Vremenske komponente')
    plt.tight_layout()
    plt.show()


def extract_features_resized_spectrogram_vec(audio_path, N=10, M=10):
    """
    Generates a spectrogram of dimensions N x M from an audio file.
    Parameters:
    - audio_path: Path to the input audio file.
    - N: Desired number of time resolution cells.
    - M: Desired number of frequency resolution cells.
    - plot_path: Path to save the spectrogram plot. If None, the plot is shown.

    Returns:
    - Tha vector representation of the spectrogram as a numpy array.
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, res_type='kaiser_fast')

    # Compute the spectrogram
    S = np.abs(librosa.stft(y))

    # Convert to dB
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    # Resize the spectrogram to the desired dimensions
    S_resized = librosa.resample(S_dB, orig_sr=S_dB.shape[1], target_sr=M, axis=1)
    S_resized_final = librosa.resample(S_resized, orig_sr=S_resized.shape[0], target_sr=N, axis=0).flatten()
    return S_resized_final


# Function to extract features
def extract_features_mfcc_vec(audio_path, num_of_mfcc=26, total_frames=18, mean=False):
    """
    This function takes the audio file path and return a feature vector in the form of a MFCC spectrogram
    :param audio_path: absolute path to the audio file.
    :param num_of_mfcc: Number of MFCC coefficients.
    :param mean: weather or not we want to find the mean by rows of the MFCC spectrogram.
    :param total_frames: Total number of time frames we split the audio file.
    :return: The feature vec of the given audio sample
    """
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    hop_length = len(audio) // (total_frames * 2)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_of_mfcc, hop_length=hop_length)
    mfccs_log = np.log(np.abs(mfccs) + 1e-6)
    if mean:
        mfccs_processed = np.mean(mfccs.T, axis=0)
    else:
        # Sort each column of mfccs_cut
        mfccs_sorted_columns = np.sort(mfccs_log, axis=0)

        # Flatten the sorted matrix into a vector
        mfccs_vector = mfccs_sorted_columns.flatten()

        # assign the data to return it
        mfccs_processed = np.array(mfccs_vector)
    return mfccs_processed


def train_classifier_no_dim_red(audio_folder_path, num=200):
    """
    :param audio_folder_path: path to audio folder, using BD 3.1
    :param num: number od samples per class
    :return: void
    """

    # Prepare the dataset
    features = []
    features_reduced = []
    labels = []
    audio_files = [f for f in os.listdir(audio_folder_path) if f.endswith('.wav')]
    audio_files.sort()  # Make sure the list is sorted to maintain the order

    # Assign class labels based on file position
    for i, file_name in enumerate(audio_files):
        file_path = f'{audio_folder_path}/{file_name}'
        feature_vec = extract_features_mfcc_vec(file_path, mean=False)
        if feature_vec is not None:
            features.append(feature_vec)
            # Determine class label based on file position
            labels.append(i // num)  # Integer division to assign class labels

    # Convert to numpy arrays for ML processing
    features = np.array(features)
    labels = np.array(labels)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model (using SVM in this example)
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Save the model and scaler for later use (optional)
    import joblib
    joblib.dump(model, 'spoken_word_classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')


def create_features(audio_folder_path, number_of_samples_per_class, freq_frames=26, time_frames=20, num_classes=4):
    lda = LDA(n_components=3)

    features_final = None
    words = ['kreni', 'stani', 'levo', 'desno']
    for word in words:
        features = []
        for i in range(number_of_samples_per_class):
            path = f'{audio_folder_path}/{word}_{i + 1}.wav'
            feature_vec = extract_features_mfcc_vec(path)
            features.append(feature_vec)
        features = np.array(features)
        if features_final is None:
            features_final = features
        else:
            features_final = np.concatenate((features_final, features))
    labels = np.array([i // number_of_samples_per_class for i in range(num_classes * number_of_samples_per_class)]).T

    features_final = lda.fit_transform(features_final, labels)

    plot_3d_points(features_final)

    return features_final, labels


def train_classifier_3d(audio_folder_path, number_of_samples_per_class, freq_frames=26, time_frames=18, num_classes=4):
    """
    :param audio_folder_path: Path to the audio DB (in this case use 3.1)
    :param number_of_samples_per_class: Number of samples per class
    :param num_classes: Number of classes default is 4
    :return: The model we trained on the DB and its accuracy score
    """
    features_final, labels = create_features(audio_folder_path, number_of_samples_per_class,
                                             freq_frames=freq_frames, time_frames=time_frames)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features_final, labels, test_size=0.3, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model (using SVM in this example)
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # predict the whole DB (3.1)
    features_final = scaler.fit_transform(features_final)
    y_pred_full = model.predict(features_final)

    # Compute confusion matrix
    cm = confusion_matrix(labels, y_pred_full)

    # Plot the heatmap
    words = ['kreni', 'stani', 'levo', 'desno']
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=words, yticklabels=words)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


    # return the accuracy and model itself
    return model, accuracy


def final_test(model, audio_folder_path):
    """
    Final test for the trained model
    :param model: The model we trained on the train DB
    :param audio_folder_path: The path to the test DB
    :return: The accuracy score of the model
    """
    features_final, labels = create_features(audio_folder_path, number_of_samples_per_class=60)

    scaler = StandardScaler()
    features_final = scaler.fit_transform(features_final)
    y_pred_full = model.predict(features_final)

    accuracy = accuracy_score(labels, y_pred_full)

    # Compute confusion matrix
    cm = confusion_matrix(labels, y_pred_full)

    # Plot the heatmap
    words = ['kreni', 'stani', 'levo', 'desno']
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=words, yticklabels=words)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy


def misc():
    # Define parameters
    num_filters = 20
    fs = 16000  # Sampling frequency (Hz)
    low_freq = 0  # Lower frequency limit of the filterbank (Hz)
    high_freq = fs / 2  # Upper frequency limit of the filterbank (Hz)

    # Convert frequency limits to Mel scale
    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)

    # Equally spaced Mel frequencies
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)

    # Convert Mel frequencies back to Hz
    freq_points = 700 * (10 ** (mel_points / 2595) - 1)

    # Calculate bandwidth of each filter
    bandwidths = freq_points[1:] - freq_points[:-1]

    # Visualize the Mel-filterbank (triangular filters)
    plt.figure(figsize=(10, 4))
    plt.title('Mel-filterbank')
    for i in range(num_filters):
        # Create triangular filter shape
        freqs = np.linspace(freq_points[i], freq_points[i + 2], 1000)
        filter_shape = np.maximum(0, 1 - np.abs((freqs - freq_points[i + 1]) / (bandwidths[i] / 2)))
        plt.plot(freqs, filter_shape)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Path to the folder containing all audio files
    number_of_samples_per_class = 200
    num_classes = 4
    audio_folder_path = 'C:/Users/jm190/Desktop/Diplomski/Baza 3.1 extended'
    test_db_path = 'C:/Users/jm190/Desktop/Diplomski/Baza test/tmp'

    # plot_audio_analysis(f'{audio_folder_path}/kreni_34.wav')

    # plot_mean_spectrogram(f'{audio_folder_path}/kreni_70.wav')

    model, acc = train_classifier_3d(audio_folder_path, number_of_samples_per_class)
    print(f'Score on train DB: {acc}')
    print(f'Score on final test DB: {final_test(model, test_db_path)}')

    # Set the ranges for x and y
    x_range = range(13, 41)  # Integers between 13 and 40
    y_range = range(5, 21)  # Integers between 5 and 20

    # Initialize lists to store the results
    x_vals = []
    y_vals = []
    z_vals = []

    prog = 1
    total = len(x_range) * len(y_range)
    # Calculate the function values for each combination of x and y
    for x in x_range:
        for y in y_range:
            model, acc = train_classifier_3d(audio_folder_path, number_of_samples_per_class,
                                             freq_frames=x, time_frames=y)
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(acc)
            print(f'Progress: {prog}/{total}')
            prog += 1

    # Convert lists to numpy arrays for easier manipulation
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    # Find the maximum value and its location
    max_index = np.argmax(z_vals)
    max_x = x_vals[max_index]
    max_y = y_vals[max_index]
    max_z = z_vals[max_index]

    # Plot the surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D scatter plot
    scat = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', alpha=0.8)
    fig.colorbar(scat, ax=ax, label='Accuracy-value')

    # Highlight the maximum value
    ax.scatter(max_x, max_y, max_z, color='r', s=100, label=f'Maximum ({max_x}, {max_y}, {max_z})')
    ax.legend()

    # Set axis labels
    ax.set_xlabel('Number of MFCCs')
    ax.set_ylabel('Number of time frames')
    ax.set_zlabel('Accuracy')

    # Show the plot
    plt.show()
