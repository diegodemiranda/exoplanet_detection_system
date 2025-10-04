import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense,
                                    Dropout, Concatenate, BatchNormalization,
                                    Bidirectional, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

class ExoplanetDetector:
    """
    Modelo híbrido CNN + LSTM para detecção de exoplanetas
    Baseado no AstroNet do Google AI e otimizado para dados Kepler/TESS
    """

    def __init__(self, sequence_length=2001, n_features=1, n_classes=3, classes=None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        # Ordem fixa de classes para manter compatibilidade com o serviço
        self.classes = classes or ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
        self.model = None
        self.scaler = StandardScaler()

    def preprocess_light_curve(self, flux_data, length=None):
        """
        Pré-processa curvas de luz para o modelo

        Args:
            flux_data: Array de dados de fluxo
            length: Comprimento desejado da série temporal

        Returns:
            Curva de luz normalizada e processada
        """
        if length is None:
            length = self.sequence_length

        flux_data = np.asarray(flux_data, dtype=np.float32)

        # Normalização robusta (median + MAD) com proteção para MAD=0
        median_flux = np.median(flux_data)
        mad_flux = np.median(np.abs(flux_data - median_flux))
        if mad_flux <= 0:
            flux_normalized = flux_data - median_flux
        else:
            flux_normalized = (flux_data - median_flux) / (1.4826 * mad_flux)

        # Remoção de outliers (> 5 sigma)
        flux_clipped = np.clip(flux_normalized, -5.0, 5.0)

        # Padding ou truncamento para tamanho fixo
        if len(flux_clipped) > length:
            flux_processed = flux_clipped[:length]
        else:
            flux_processed = np.pad(flux_clipped, (0, length - len(flux_clipped)), 'constant')

        return flux_processed

    def create_model(self):
        """
        Cria modelo híbrido CNN + LSTM otimizado para detecção de exoplanetas
        """

        # Entrada para curvas de luz (Global view)
        light_curve_global = Input(shape=(self.sequence_length, self.n_features), name='global_view')

        # Entrada para vista local (janela de trânsito)
        light_curve_local = Input(shape=(201, self.n_features), name='local_view')

        # Branch CNN para vista global
        global_branch = Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(light_curve_global)
        global_branch = BatchNormalization()(global_branch)
        global_branch = MaxPooling1D(pool_size=5)(global_branch)

        global_branch = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(global_branch)
        global_branch = BatchNormalization()(global_branch)
        global_branch = MaxPooling1D(pool_size=5)(global_branch)

        global_branch = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(global_branch)
        global_branch = BatchNormalization()(global_branch)
        global_branch = MaxPooling1D(pool_size=5)(global_branch)

        # LSTM bidirecional para capturar padrões temporais
        global_branch = Bidirectional(LSTM(50, return_sequences=True, dropout=0.2))(global_branch)
        global_branch = Bidirectional(LSTM(50, return_sequences=False, dropout=0.2))(global_branch)

        # Branch CNN para vista local
        local_branch = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(light_curve_local)
        local_branch = BatchNormalization()(local_branch)
        local_branch = MaxPooling1D(pool_size=2)(local_branch)

        local_branch = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(local_branch)
        local_branch = BatchNormalization()(local_branch)
        local_branch = MaxPooling1D(pool_size=2)(local_branch)

        local_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(local_branch)
        local_branch = BatchNormalization()(local_branch)
        local_branch = GlobalAveragePooling1D()(local_branch)

        # Entrada para características auxiliares (parâmetros estelares)
        auxiliary_input = Input(shape=(10,), name='auxiliary_features')
        aux = Dense(64, activation='relu')(auxiliary_input)
        aux = Dropout(0.3)(aux)
        aux = Dense(32, activation='relu')(aux)

        # Concatenação de todas as características
        combined = Concatenate()([global_branch, local_branch, aux])

        # Camadas densas finais
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.4)(combined)
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.3)(combined)

        # Camada de saída
        output = Dense(self.n_classes, activation='softmax', name='classification')(combined)

        # Criar o modelo
        model = Model(inputs=[light_curve_global, light_curve_local, auxiliary_input], outputs=output)

        # Compilar o modelo com métricas válidas para inferência/treino
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def augment_data(self, flux_data, labels, noise_level=0.01):
        """
        Aplica data augmentation em curvas de luz
        """
        augmented_flux = []
        augmented_labels = []

        for i, flux in enumerate(flux_data):
            label = labels[i]

            # Dados originais
            augmented_flux.append(flux)
            augmented_labels.append(label)

            # Adição de ruído gaussiano
            noisy_flux = flux + np.random.normal(0, noise_level, flux.shape)
            augmented_flux.append(noisy_flux)
            augmented_labels.append(label)

            # Reflexão temporal
            reflected_flux = np.flip(flux)
            augmented_flux.append(reflected_flux)
            augmented_labels.append(label)

            # Shift temporal pequeno
            shift = np.random.randint(-50, 51)
            shifted_flux = np.roll(flux, shift)
            augmented_flux.append(shifted_flux)
            augmented_labels.append(label)

        return np.array(augmented_flux), np.array(augmented_labels)

    def _encode_labels(self, y):
        """Codifica labels conforme a ordem fixa de classes."""
        indices = np.array([self.classes.index(lbl) for lbl in y], dtype=np.int32)
        return to_categorical(indices, num_classes=self.n_classes)

    def train(self, X_global, X_local, X_aux, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Treina o modelo
        """
        # Preparar callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7, monitor='val_loss')
        ]

        # Codificar labels com ordem fixa
        y_encoded = self._encode_labels(y)

        # Treinar o modelo
        history = self.model.fit(
            [X_global, X_local, X_aux],
            y_encoded,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, X_global, X_local, X_aux):
        """
        Faz predições
        """
        predictions = self.model.predict([X_global, X_local, X_aux])
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_classes = [self.classes[i] for i in predicted_indices]
        return np.array(predicted_classes), predictions

    def evaluate_model(self, X_global, X_local, X_aux, y_true):
        """
        Avalia o desempenho do modelo
        """
        y_pred, probabilities = self.predict(X_global, X_local, X_aux)

        # Relatório de classificação
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)

        return report, cm, probabilities
