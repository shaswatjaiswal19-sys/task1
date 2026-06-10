# =============================================================================
# CodeAlpha Internship — Task 2: Emotion Recognition from Speech
# Objective: Recognize human emotions (happy, angry, sad) from speech audio
# Approach: Deep Learning + Speech Signal Processing (MFCCs)
# Models: CNN, LSTM
# Dataset: RAVDESS (Real) / Synthetic MFCC (Demo Mode)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import librosa
import soundfile as sf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, Flatten,
    LSTM, BatchNormalization, Reshape
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =============================================================================
# CONFIGURATION
# =============================================================================
# Set DEMO_MODE = False if you have downloaded the RAVDESS dataset locally
# Set DEMO_MODE = True to run with synthetic MFCC data instantly

DEMO_MODE = True
RAVDESS_PATH = "./ravdess_data"  # Change this to your RAVDESS folder path
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 150  # Max time steps for MFCC frames

# RAVDESS Emotion Mapping
EMOTION_MAP = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}

# Emotion abbreviations for plots
EMOTION_ABBREV = {
    'neutral': 'NEU', 'calm': 'CAL', 'happy': 'HAP', 'sad': 'SAD',
    'angry': 'ANG', 'fearful': 'FEA', 'disgust': 'DIS', 'surprised': 'SUR'
}


# =============================================================================
# STEP 1: DATA LOADING & FEATURE EXTRACTION
# =============================================================================

def extract_mfcc(file_path, max_pad_len=MAX_PAD_LEN):
    """
    Extract MFCC features from an audio file.
    Pads or truncates to a fixed length for uniform input shape.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

        # Pad or truncate
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None


def load_ravdess_data(data_path):
    """
    Load RAVDESS dataset and extract MFCC features.
    RAVDESS filename format: 03-01-01-01-01-01-01.wav
    Third segment (01-08) represents the emotion.
    """
    features = []
    labels = []

    print(f"📂 Loading RAVDESS data from: {data_path}")

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                try:
                    emotion_code = int(file.split('-')[2])
                    emotion = EMOTION_MAP.get(emotion_code)
                    if emotion is None:
                        continue

                    file_path = os.path.join(root, file)
                    mfccs = extract_mfcc(file_path)

                    if mfccs is not None:
                        features.append(mfccs)
                        labels.append(emotion)
                except:
                    continue

    print(f"   ✅ Loaded {len(features)} audio files")
    return np.array(features), np.array(labels)


def generate_synthetic_mfcc_data(n_samples=1200, n_classes=8, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    """
    Generate synthetic MFCC data that mimics real audio features.
    Enables the code to run out-of-the-box for demonstration.
    """
    print("🎭 DEMO MODE: Generating synthetic MFCC data...")

    emotions = list(EMOTION_MAP.values())
    features = []
    labels = []

    samples_per_class = n_samples // n_classes

    for emotion in emotions:
        # Each emotion gets a slightly different MFCC distribution
        if emotion == 'angry':
            base_freq, energy = 8.0, 2.5
        elif emotion == 'sad':
            base_freq, energy = 3.0, 0.8
        elif emotion == 'happy':
            base_freq, energy = 7.0, 2.0
        elif emotion == 'calm':
            base_freq, energy = 4.0, 1.0
        elif emotion == 'fearful':
            base_freq, energy = 9.0, 1.5
        elif emotion == 'disgust':
            base_freq, energy = 6.0, 1.8
        elif emotion == 'surprised':
            base_freq, energy = 10.0, 2.2
        else:  # neutral
            base_freq, energy = 5.0, 1.2

        for _ in range(samples_per_class):
            # Simulate MFCC: (n_mfcc, time_steps)
            noise = np.random.randn(n_mfcc, max_pad_len) * energy
            pattern = np.sin(np.linspace(0, base_freq * np.pi, max_pad_len))

            # Add frequency patterns to lower MFCC coefficients
            mfcc = noise.copy()
            for i in range(min(10, n_mfcc)):
                mfcc[i] += pattern * (10 - i) * 0.3

            # Add temporal dynamics
            mfcc += np.random.randn(n_mfcc, max_pad_len) * 0.5

            features.append(mfcc)
            labels.append(emotion)

    features = np.array(features)
    labels = np.array(labels)

    # Shuffle
    idx = np.random.permutation(len(features))
    features = features[idx]
    labels = labels[idx]

    print(f"   ✅ Generated {len(features)} synthetic samples across {n_classes} emotions")
    return features, labels


def load_data():
    """Load data based on DEMO_MODE configuration."""
    if DEMO_MODE:
        return generate_synthetic_mfcc_data(n_samples=1200)
    else:
        if not os.path.exists(RAVDESS_PATH):
            print(f"❌ RAVDESS path not found: {RAVDESS_PATH}")
            print("   Switching to DEMO MODE...")
            return generate_synthetic_mfcc_data(n_samples=1200)
        return load_ravdess_data(RAVDESS_PATH)


# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# =============================================================================

def perform_eda(features, labels):
    """Visualize MFCC features and emotion distribution."""

    print("\n" + "=" * 70)
    print("📊 EXPLORATORY DATA ANALYSIS — MFCC Features")
    print("=" * 70)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n🔹 Total Samples: {len(labels)}")
    print(f"🔹 Feature Shape: {features.shape}")
    print(f"🔹 Number of Classes: {len(unique)}")
    print(f"\n🔹 Class Distribution:")
    for emo, count in zip(unique, counts):
        print(f"   {emo:12s}: {count:4d} samples ({count/len(labels)*100:.1f}%)")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Emotion Recognition from Speech — EDA', fontsize=18, fontweight='bold')

    # 1. Emotion Distribution
    ax = axes[0, 0]
    colors = sns.color_palette("husl", len(unique))
    bars = ax.bar(unique, counts, color=colors)
    ax.set_title('Emotion Distribution', fontweight='bold')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # 2-4. MFCC Spectrograms for different emotions
    emotions_to_show = ['happy', 'sad', 'angry'] if all(e in unique for e in ['happy', 'sad', 'angry']) else unique[:3]

    for idx, emotion in enumerate(emotions_to_show):
        ax = axes[0 + idx // 3, 0 + idx % 3]
        if idx >= 3:
            break
        emo_idx = np.where(labels == emotion)[0][0]
        mfcc = features[emo_idx]

        img = ax.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'MFCC Spectrogram — {emotion.upper()}', fontweight='bold')
        ax.set_xlabel('Time Frames')
        ax.set_ylabel('MFCC Coefficients')
        plt.colorbar(img, ax=ax, shrink=0.8)

    # 5. Average MFCC per Emotion (first coefficient)
    ax = axes[1, 1]
    for emotion in unique:
        emo_features = features[labels == emotion]
        avg_mfcc_0 = emo_features[:, 0, :].mean(axis=0)
        ax.plot(avg_mfcc_0, label=EMOTION_ABBREV.get(emotion, emotion), alpha=0.8)

    ax.set_title('Average MFCC-0 Over Time by Emotion', fontweight='bold')
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('MFCC Value')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 6. MFCC Energy Distribution
    ax = axes[1, 2]
    for emotion in unique:
        emo_features = features[labels == emotion]
        energy = emo_features.var(axis=(1, 2))
        ax.hist(energy, bins=20, alpha=0.4, label=EMOTION_ABBREV.get(emotion, emotion))

    ax.set_title('MFCC Variance Distribution', fontweight='bold')
    ax.set_xlabel('Variance')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig('eda_speech.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ EDA plots saved as 'eda_speech.png'")


# =============================================================================
# STEP 3: DATA PREPROCESSING
# =============================================================================

def preprocess_data(features, labels):
    """Encode labels, normalize features, split data."""

    print("\n" + "=" * 70)
    print("🔧 DATA PREPROCESSING")
    print("=" * 70)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y_categorical = to_categorical(y_encoded)
    num_classes = len(le.classes_)

    print(f"   ✅ Label Encoding: {num_classes} classes → {list(le.classes_)}")

    # Normalize features
    mean = features.mean()
    std = features.std()
    features_normalized = (features - mean) / (std + 1e-8)

    print(f"   ✅ Feature Normalization: mean={mean:.4f}, std={std:.4f}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        features_normalized, y_categorical,
        test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1, random_state=42, stratify=y_train.argmax(axis=1)
    )

    print(f"   ✅ Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # Reshape for CNN: (samples, height, width, channels)
    X_train_cnn = X_train[..., np.newaxis]
    X_val_cnn = X_val[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]

    # For LSTM: (samples, timesteps, features) — transpose MFCC
    X_train_lstm = X_train.transpose(0, 2, 1)
    X_val_lstm = X_val.transpose(0, 2, 1)
    X_test_lstm = X_test.transpose(0, 2, 1)

    print(f"   ✅ CNN input shape: {X_train_cnn.shape}")
    print(f"   ✅ LSTM input shape: {X_train_lstm.shape}")

    return {
        'cnn': (X_train_cnn, X_val_cnn, X_test_cnn),
        'lstm': (X_train_lstm, X_val_lstm, X_test_lstm),
        'labels': (y_train, y_val, y_test),
        'encoder': le,
        'num_classes': num_classes,
        'norm_params': (mean, std)
    }


# =============================================================================
# STEP 4: MODEL BUILDING — CNN & LSTM
# =============================================================================

def build_cnn_model(input_shape, num_classes):
    """Build a 2D CNN model for MFCC spectrogram classification."""

    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Classifier
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_lstm_model(input_shape, num_classes):
    """Build an LSTM model for MFCC sequence classification."""

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =============================================================================
# STEP 5: MODEL TRAINING
# =============================================================================

def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=50):
    """Train the model with callbacks."""

    print(f"\n{'─' * 50}")
    print(f"🏋️ Training: {model_name}")
    print(f"{'─' * 50}")

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# =============================================================================
# STEP 6: EVALUATION & VISUALIZATION
# =============================================================================

def evaluate_model(model, X_test, y_test, le, model_name, history):
    """Evaluate model and generate comprehensive metrics & plots."""

    # Predict
    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    print(f"\n{'─' * 50}")
    print(f"📊 Evaluation: {model_name}")
    print(f"{'─' * 50}")
    print(f"   Test Accuracy: {acc:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    return y_true, y_pred, acc, history


def plot_training_history(histories, model_names):
    """Plot training curves for all models."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training History — Loss & Accuracy', fontsize=16, fontweight='bold')

    colors = ['#3498db', '#e74c3c']

    # Accuracy
    ax = axes[0]
    for i, (history, name) in enumerate(zip(histories, model_names)):
        ax.plot(history.history['accuracy'], label=f'{name} Train', color=colors[i], linewidth=2)
        ax.plot(history.history['val_accuracy'], label=f'{name} Val', color=colors[i],
                linewidth=2, linestyle='--')
    ax.set_title('Model Accuracy', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1]
    for i, (history, name) in enumerate(zip(histories, model_names)):
        ax.plot(history.history['loss'], label=f'{name} Train', color=colors[i], linewidth=2)
        ax.plot(history.history['val_loss'], label=f'{name} Val', color=colors[i],
                linewidth=2, linestyle='--')
    ax.set_title('Model Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Training history saved as 'training_history.png'")


def plot_evaluation_dashboard(results, le):
    """Create comprehensive evaluation dashboard."""

    model_names = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle('Emotion Recognition — Model Evaluation Dashboard', fontsize=18, fontweight='bold')

    # 1. Accuracy Comparison
    ax = axes[0, 0]
    accs = [results[name]['accuracy'] for name in model_names]
    bars = ax.bar(model_names, accs, color=['#3498db', '#e74c3c'], alpha=0.85)
    ax.set_ylim(0, 1.0)
    ax.set_title('Test Accuracy Comparison', fontweight='bold')
    ax.set_ylabel('Accuracy')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontweight='bold', fontsize=12)

    # 2. CNN Confusion Matrix
    ax = axes[0, 1]
    if 'CNN' in results:
        cm = confusion_matrix(results['CNN']['y_true'], results['CNN']['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_title('Confusion Matrix — CNN', fontweight='bold')
        ax.set_ylabel('True Emotion')
        ax.set_xlabel('Predicted Emotion')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    # 3. LSTM Confusion Matrix
    ax = axes[1, 0]
    if 'LSTM' in results:
        cm = confusion_matrix(results['LSTM']['y_true'], results['LSTM']['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_title('Confusion Matrix — LSTM', fontweight='bold')
        ax.set_ylabel('True Emotion')
        ax.set_xlabel('Predicted Emotion')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    # 4. Per-Class F1-Score Comparison
    ax = axes[1, 1]
    from sklearn.metrics import f1_score
    width = 0.35
    x = np.arange(len(le.classes_))

    for i, name in enumerate(model_names):
        f1 = f1_score(results[name]['y_true'], results[name]['y_pred'], average=None)
        ax.bar(x + i * width, f1, width, label=name,
               color=['#3498db', '#e74c3c'][i], alpha=0.85)

    ax.set_title('Per-Class F1-Score', fontweight='bold')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('F1-Score')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(le.classes_, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('evaluation_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Evaluation dashboard saved as 'evaluation_dashboard.png'")


# =============================================================================
# STEP 7: PREDICTION FUNCTION
# =============================================================================

def predict_emotion(model, file_path, le, norm_params, model_type='cnn'):
    """Predict emotion from an audio file."""

    mean, std = norm_params

    # Extract MFCC
    mfccs = extract_mfcc(file_path)
    if mfccs is None:
        return None

    # Normalize
    mfccs = (mfccs - mean) / (std + 1e-8)

    # Reshape
    if model_type == 'cnn':
        mfccs = mfccs[np.newaxis, ..., np.newaxis]
    else:  # lstm
        mfccs = mfccs.T[np.newaxis, ...]

    # Predict
    prediction = model.predict(mfccs, verbose=0)
    emotion_idx = prediction.argmax(axis=1)[0]
    emotion = le.classes_[emotion_idx]
    confidence = prediction[0][emotion_idx] * 100

    # Top 3 predictions
    top3_idx = prediction[0].argsort()[-3:][::-1]

    print("\n" + "=" * 50)
    print("🎙️ EMOTION PREDICTION FROM SPEECH")
    print("=" * 50)
    print(f"   🎯 Predicted Emotion: {emotion.upper()}")
    print(f"   📊 Confidence: {confidence:.1f}%")
    print(f"\n   Top 3 Predictions:")
    for idx in top3_idx:
        print(f"   → {le.classes_[idx]:12s}: {prediction[0][idx]*100:.1f}%")
    print("=" * 50)

    return emotion, confidence


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("🎙️" + " CODEALPHA — EMOTION RECOGNITION FROM SPEECH ".center(62, "=") + "🎙️")

    # Step 1: Load Data
    print("\n📁 STEP 1: Loading Data & Extracting MFCC Features...")
    features, labels = load_data()

    # Step 2: EDA
    print("\n📊 STEP 2: Performing EDA...")
    perform_eda(features, labels)

    # Step 3: Preprocessing
    print("\n🔧 STEP 3: Preprocessing...")
    data = preprocess_data(features, labels)

    X_train_cnn, X_val_cnn, X_test_cnn = data['cnn']
    X_train_lstm, X_val_lstm, X_test_lstm = data['lstm']
    y_train, y_val, y_test = data['labels']
    le = data['encoder']
    num_classes = data['num_classes']
    norm_params = data['norm_params']

    # Step 4: Build Models
    print("\n🏗️ STEP 4: Building Models...")

    cnn_model = build_cnn_model(
        input_shape=X_train_cnn.shape[1:],
        num_classes=num_classes
    )
    print("\n📋 CNN Architecture:")
    cnn_model.summary()

    lstm_model = build_lstm_model(
        input_shape=X_train_lstm.shape[1:],
        num_classes=num_classes
    )
    print("\n📋 LSTM Architecture:")
    lstm_model.summary()

    # Step 5: Train Models
    print("\n🏋️ STEP 5: Training Models...")

    # Adjust epochs for demo mode
    epochs = 30 if DEMO_MODE else 80

    cnn_model, cnn_history = train_model(
        cnn_model, X_train_cnn, y_train, X_val_cnn, y_val,
        'CNN', epochs=epochs
    )

    lstm_model, lstm_history = train_model(
        lstm_model, X_train_lstm, y_train, X_val_lstm, y_val,
        'LSTM', epochs=epochs
    )

    # Step 6: Evaluation
    print("\n📈 STEP 6: Model Evaluation...")

    results = {}

    y_true_cnn, y_pred_cnn, acc_cnn, _ = evaluate_model(
        cnn_model, X_test_cnn, y_test, le, 'CNN', cnn_history
    )
    results['CNN'] = {
        'y_true': y_true_cnn, 'y_pred': y_pred_cnn,
        'accuracy': acc_cnn, 'history': cnn_history
    }

    y_true_lstm, y_pred_lstm, acc_lstm, _ = evaluate_model(
        lstm_model, X_test_lstm, y_test, le, 'LSTM', lstm_history
    )
    results['LSTM'] = {
        'y_true': y_true_lstm, 'y_pred': y_pred_lstm,
        'accuracy': acc_lstm, 'history': lstm_history
    }

    # Step 7: Visualizations
    print("\n📊 STEP 7: Generating Visualizations...")
    plot_training_history([cnn_history, lstm_history], ['CNN', 'LSTM'])
    plot_evaluation_dashboard(results, le)

    # Step 8: Save Models
    print("\n💾 STEP 8: Saving Models...")
    cnn_model.save('emotion_cnn_model.h5')
    lstm_model.save('emotion_lstm_model.h5')
    import joblib
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(norm_params, 'norm_params.pkl')
    print("   ✅ Models saved: emotion_cnn_model.h5, emotion_lstm_model.h5")

    # Step 9: Summary
    print("\n" + "=" * 70)
    print("🎉 EMOTION RECOGNITION FROM SPEECH — TASK COMPLETE!")
    print("=" * 70)

    print(f"\n📊 FINAL RESULTS:")
    print(f"   🔹 CNN  Test Accuracy: {acc_cnn:.4f}")
    print(f"   🔹 LSTM Test Accuracy: {acc_lstm:.4f}")

    best = 'CNN' if acc_cnn >= acc_lstm else 'LSTM'
    best_acc = max(acc_cnn, acc_lstm)
    print(f"\n   🏆 Best Model: {best} ({best_acc:.4f})")

    print(f"\n📦 Deliverables:")
    print("   ✅ emotion_recognition_speech.py — Complete source code")
    print("   ✅ emotion_cnn_model.h5         — Saved CNN model")
    print("   ✅ emotion_lstm_model.h5        — Saved LSTM model")
    print("   ✅ eda_speech.png               — EDA visualizations")
    print("   ✅ training_history.png         — Training curves")
    print("   ✅ evaluation_dashboard.png     — Evaluation dashboard")

    if DEMO_MODE:
        print(f"\n💡 TIP: Set DEMO_MODE = False and provide RAVDESS dataset path")
        print(f"   for real-world emotion recognition!")


if __name__ == '__main__':
    main()
