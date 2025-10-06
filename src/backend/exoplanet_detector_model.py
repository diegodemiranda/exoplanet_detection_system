import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense,
                                    Dropout, Concatenate, BatchNormalization,
                                    Bidirectional, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

class ExoplanetDetector:
    """
    Hybrid CNN + LSTM model for exoplanet detection
    Based on Google's AstroNet and optimized for Kepler/TESS data
    """

    def __init__(self, sequence_length=2001, n_features=1, n_classes=3, classes=None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        # Fixed class order to maintain service compatibility
        self.classes = classes or ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
        self.model = None
        self.scaler = StandardScaler()

    def preprocess_light_curve(self, flux_data, length=None):
        """
        Preprocess light curves for the model

        Args:
            flux_data: Flux data array
            length: Desired time series length

        Returns:
            Normalized and processed light curve
        """
        if length is None:
            length = self.sequence_length

        flux_data = np.asarray(flux_data, dtype=np.float32)

        # Robust normalization (median + MAD) with protection for MAD=0
        median_flux = np.median(flux_data)
        mad_flux = np.median(np.abs(flux_data - median_flux))
        if mad_flux <= 0:
            flux_normalized = flux_data - median_flux
        else:
            flux_normalized = (flux_data - median_flux) / (1.4826 * mad_flux)

        # Outlier removal (> 5 sigma)
        flux_clipped = np.clip(flux_normalized, -5.0, 5.0)

        # Padding or truncation to fixed length
        if len(flux_clipped) > length:
            flux_processed = flux_clipped[:length]
        else:
            flux_processed = np.pad(flux_clipped, (0, length - len(flux_clipped)), 'constant')

        return flux_processed

    def create_model(self):
        """
        Create a hybrid CNN + LSTM model optimized for exoplanet detection
        """
        # Input for light curves (Global view)
        light_curve_global = Input(shape=(self.sequence_length, self.n_features), name='global_view')
        # Input for local view (transit window)
        light_curve_local = Input(shape=(201, self.n_features), name='local_view')

        # CNN branch for global view (more filters)
        global_branch = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(light_curve_global)
        global_branch = BatchNormalization()(global_branch)
        global_branch = MaxPooling1D(pool_size=5)(global_branch)

        global_branch = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(global_branch)
        global_branch = BatchNormalization()(global_branch)
        global_branch = MaxPooling1D(pool_size=5)(global_branch)

        global_branch = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(global_branch)
        global_branch = BatchNormalization()(global_branch)
        global_branch = MaxPooling1D(pool_size=5)(global_branch)

        # Bidirectional LSTM to capture temporal patterns
        global_branch = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(global_branch)
        global_branch = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(global_branch)

        # CNN branch for local view (more filters)
        local_branch = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(light_curve_local)
        local_branch = BatchNormalization()(local_branch)
        local_branch = MaxPooling1D(pool_size=2)(local_branch)

        local_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(local_branch)
        local_branch = BatchNormalization()(local_branch)
        local_branch = MaxPooling1D(pool_size=2)(local_branch)

        local_branch = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(local_branch)
        local_branch = BatchNormalization()(local_branch)
        local_branch = GlobalAveragePooling1D()(local_branch)

        # Input for auxiliary features (stellar parameters)
        auxiliary_input = Input(shape=(10,), name='auxiliary_features')
        aux = Dense(64, activation='relu')(auxiliary_input)
        aux = Dropout(0.3)(aux)
        aux = Dense(32, activation='relu')(aux)

        # Concatenate all features
        combined = Concatenate()([global_branch, local_branch, aux])

        # Final dense layers with L2
        combined = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(combined)
        combined = Dropout(0.4)(combined)
        combined = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(combined)
        combined = Dropout(0.3)(combined)

        # Output layer
        output = Dense(self.n_classes, activation='softmax', name='classification')(combined)

        # Create model
        model = Model(inputs=[light_curve_global, light_curve_local, auxiliary_input], outputs=output)

        # Compile with lower LR
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def augment_data(self, flux_data, labels, noise_level=0.01):
        """
        Apply data augmentation to light curves
        """
        augmented_flux = []
        augmented_labels = []

        for i, flux in enumerate(flux_data):
            label = labels[i]

            # Original data
            augmented_flux.append(flux)
            augmented_labels.append(label)

            # Add Gaussian noise
            noisy_flux = flux + np.random.normal(0, noise_level, flux.shape)
            augmented_flux.append(noisy_flux)
            augmented_labels.append(label)

            # Time reflection
            reflected_flux = np.flip(flux)
            augmented_flux.append(reflected_flux)
            augmented_labels.append(label)

            # Small temporal shift
            shift = np.random.randint(-50, 51)
            shifted_flux = np.roll(flux, shift)
            augmented_flux.append(shifted_flux)
            augmented_labels.append(label)

        return np.array(augmented_flux), np.array(augmented_labels)

    def _encode_labels(self, y):
        """Encode labels according to the fixed class order."""
        indices = np.array([self.classes.index(lbl) for lbl in y], dtype=np.int32)
        return to_categorical(indices, num_classes=self.n_classes)

    def train(self, X_global, X_local, X_aux, y, validation_data=None, validation_split=0.0, epochs=100, batch_size=32):
        """
        Train the model without EarlyStopping (ensures all epochs run)
        and with class weights to handle imbalance.
        If validation_data is provided, use (Xg_val, Xl_val, Xa_val, y_val).
        """
        # Callback for LR adjustment
        callbacks = [ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7, monitor='val_loss')]

        # Encode labels with fixed order
        y_indices = np.array([self.classes.index(lbl) for lbl in y], dtype=np.int32)
        y_encoded = to_categorical(y_indices, num_classes=self.n_classes)

        # Balanced class weights
        try:
            cls_weights_arr = compute_class_weight(class_weight='balanced', classes=np.arange(self.n_classes), y=y_indices)
            class_weight = {i: float(w) for i, w in enumerate(cls_weights_arr)}
        except Exception:
            class_weight = None

        fit_kwargs = {
            'x': [X_global, X_local, X_aux],
            'y': y_encoded,
            'epochs': epochs,
            'batch_size': batch_size,
            'callbacks': callbacks,
            'shuffle': True,
            'class_weight': class_weight,
            'verbose': 1
        }
        if validation_data is not None:
            Xg_val, Xl_val, Xa_val, y_val = validation_data
            y_val_idx = np.array([self.classes.index(lbl) for lbl in y_val], dtype=np.int32)
            y_val_enc = to_categorical(y_val_idx, num_classes=self.n_classes)
            fit_kwargs['validation_data'] = ([Xg_val, Xl_val, Xa_val], y_val_enc)
        else:
            fit_kwargs['validation_split'] = validation_split

        history = self.model.fit(**fit_kwargs)
        return history

    def predict(self, X_global, X_local, X_aux):
        """
        Make predictions
        """
        predictions = self.model.predict([X_global, X_local, X_aux])
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_classes = [self.classes[i] for i in predicted_indices]
        return np.array(predicted_classes), predictions

    def evaluate_model(self, X_global, X_local, X_aux, y_true):
        """
        Evaluate model performance
        """
        y_pred, probabilities = self.predict(X_global, X_local, X_aux)

        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)

        return report, cm, probabilities
