import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import scipy.stats as stats
from sklearn.preprocessing import RobustScaler
import os
from tensorflow.keras.models import load_model

from models import OptimizedExoplanetCandidate, OptimizedPredictionResult
from cache import cached_prediction, prediction_cache
from exceptions import ModelNotLoadedError, ProcessingError, PredictionError
from config import settings
from exoplanet_detector_model import ExoplanetDetector

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Métricas de processamento"""
    preprocessing_time: float = 0.0
    feature_extraction_time: float = 0.0
    prediction_time: float = 0.0
    total_time: float = 0.0
    data_quality_score: float = 0.0


class DataPreprocessor:
    """Preprocessador de dados otimizado"""

    def __init__(self):
        self.scaler = RobustScaler()
        self._fitted = False

    def preprocess_light_curve(self, flux_data: List[float], length: int = None) -> np.ndarray:
        """Pré-processamento otimizado da curva de luz"""
        start_time = datetime.now()

        try:
            if length is None:
                length = 2001

            flux_array = np.array(flux_data, dtype=np.float32)

            # Normalização robusta usando percentis
            q25, q75 = np.percentile(flux_array, [25, 75])
            iqr = q75 - q25
            if iqr > 0:
                flux_normalized = (flux_array - np.median(flux_array)) / iqr
            else:
                flux_normalized = flux_array - np.median(flux_array)

            # Detecção e remoção de outliers usando z-score modificado
            mad = np.median(np.abs(flux_normalized - np.median(flux_normalized)))
            if mad > 0:
                modified_z_scores = 0.6745 * (flux_normalized - np.median(flux_normalized)) / mad
                flux_clean = np.where(np.abs(modified_z_scores) < 3.5, flux_normalized, np.median(flux_normalized))
            else:
                flux_clean = flux_normalized

            # Redimensionamento para tamanho fixo
            flux_processed = self._resize_sequence(flux_clean, length)

            return flux_processed

        except Exception as e:
            raise ProcessingError("preprocessamento de curva de luz", str(e))

    def _resize_sequence(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Redimensiona sequência mantendo características importantes"""
        if len(data) == target_length:
            return data
        elif len(data) > target_length:
            # Downsampling inteligente preservando características de trânsito
            indices = np.linspace(0, len(data) - 1, target_length, dtype=int)
            return data[indices]
        else:
            # Padding simples
            return np.pad(data, (0, target_length - len(data)), 'constant')


class FeatureExtractor:
    """Extrator de características otimizado"""

    @staticmethod
    def extract_statistical_features(flux_data: np.ndarray) -> Dict[str, float]:
        """Extrai características estatísticas da curva de luz"""
        try:
            mean = float(np.mean(flux_data))
            std = float(np.std(flux_data))
            min_v = float(np.min(flux_data))
            max_v = float(np.max(flux_data))
            median = float(np.median(flux_data))

            # Tratar casos degenerados (sinal constante)
            if std < 1e-12:
                skew = 0.0
                kurt = 0.0
                cv = 0.0
                autocorr_lag1 = 0.0
            else:
                skew = float(stats.skew(flux_data))
                kurt = float(stats.kurtosis(flux_data))
                cv = float(np.std(flux_data) / (abs(mean) + 1e-8))
                # Autocorrelação lag-1 com proteção contra NaN
                ac = np.corrcoef(flux_data[:-1], flux_data[1:])[0, 1] if len(flux_data) > 1 else 0.0
                autocorr_lag1 = float(np.nan_to_num(ac, nan=0.0))

            depth_estimate = float(abs(min_v))
            snr_estimate = float(abs(min_v) / (std + 1e-8))
            mad = float(np.median(np.abs(flux_data - median)))
            iqr = float(np.percentile(flux_data, 75) - np.percentile(flux_data, 25))
            rng = float(np.ptp(flux_data))

            features = {
                'mean': mean,
                'std': std,
                'min': min_v,
                'max': max_v,
                'median': median,
                'skewness': skew,
                'kurtosis': kurt,
                'range': rng,
                'mad': mad,
                'iqr': iqr,
                'cv': cv,
                'autocorr_lag1': autocorr_lag1,
                'depth_estimate': depth_estimate,
                'snr_estimate': snr_estimate
            }

            # Sanitizar valores para evitar NaN/Inf
            for k, v in list(features.items()):
                features[k] = float(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0))

            return features

        except Exception as e:
            raise ProcessingError("extração de características estatísticas", str(e))


class IExoplanetDetector(ABC):
    """Interface para detectores de exoplanetas"""

    @abstractmethod
    async def predict(self, candidate: OptimizedExoplanetCandidate) -> OptimizedPredictionResult:
        """Prediz classificação de candidato"""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do modelo"""
        pass


class OptimizedExoplanetDetectorService(IExoplanetDetector):
    """Serviço otimizado de detecção de exoplanetas"""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_extractor = FeatureExtractor()
        # Carregar modelo real de Deep Learning
        self.detector: Optional[ExoplanetDetector] = None
        self.model_loaded = False
        self.classes = ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
        self._prediction_count = 0
        self._total_processing_time = 0.0

        # Métricas do modelo (inicialização)
        self._metrics = {
            "accuracy": None,
            "precision": {},
            "recall": {},
            "f1_score": {},
            "total_predictions": 0,
            "last_updated": datetime.now()
        }

        # Inicializa e carrega modelo
        self._init_model()

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo carregado para troubleshooting"""
        info = {
            "loaded": bool(self.model_loaded and self.detector is not None and getattr(self.detector, 'model', None) is not None),
            "full_model_path": settings.model_full_path,
            "classes": self.classes,
            "input_shapes": {
                "global_view": (settings.sequence_length, settings.n_features),
                "local_view": (settings.local_view_length, settings.n_features),
                "auxiliary_features": (10,)
            }
        }
        return info

    def _init_model(self) -> None:
        """Inicializa arquitetura e carrega o modelo DL (.keras)"""
        try:
            # Inicializar wrapper do detector (por consistência no preprocess)
            self.detector = ExoplanetDetector(
                sequence_length=settings.sequence_length,
                n_features=settings.n_features,
                n_classes=settings.n_classes
            )

            # Carregar modelo completo .keras
            full_path = settings.model_full_path
            if os.path.exists(full_path):
                self.detector.model = load_model(full_path)
                self.model_loaded = True
                logger.info(f"Modelo completo (.keras) carregado com sucesso de: {full_path}")
            else:
                self.model_loaded = False
                logger.error(f"Modelo .keras não encontrado em: {full_path}")
        except Exception as e:
            logger.exception(f"Falha ao inicializar/carregar o modelo .keras: {e}")
            self.model_loaded = False

    @cached_prediction(prediction_cache, ttl=3600)
    async def predict(self, candidate: OptimizedExoplanetCandidate) -> OptimizedPredictionResult:
        """Predição com modelo real de Deep Learning e métricas"""
        if not self.model_loaded or self.detector is None:
            raise ModelNotLoadedError()

        start_time = datetime.now()
        metrics = ProcessingMetrics()

        try:
            # Pré-processamento consistente com o treinamento
            preprocessing_start = datetime.now()
            flux_array = np.array(candidate.light_curve.flux, dtype=np.float32)
            flux_processed = self.detector.preprocess_light_curve(flux_array, length=settings.sequence_length)
            metrics.preprocessing_time = (datetime.now() - preprocessing_start).total_seconds()

            # Extração de características auxiliares (para qualidade e features auxiliares)
            feature_start = datetime.now()
            features = self.feature_extractor.extract_statistical_features(flux_processed)
            metrics.feature_extraction_time = (datetime.now() - feature_start).total_seconds()

            # Avaliação de qualidade dos dados
            metrics.data_quality_score = self._calculate_data_quality(features)

            # Montagem das entradas do modelo
            X_global = flux_processed.reshape(1, settings.sequence_length, 1)

            # Vista local: janela centrada no mínimo (possível trânsito)
            local_len = settings.local_view_length
            idx_min = int(np.argmin(flux_processed)) if len(flux_processed) > 0 else 0
            half = local_len // 2
            start = max(0, idx_min - half)
            end = start + local_len
            if end > len(flux_processed):
                end = len(flux_processed)
                start = max(0, end - local_len)
            local_view = flux_processed[start:end]
            if len(local_view) < local_len:
                local_view = np.pad(local_view, (0, local_len - len(local_view)), 'constant')
            X_local = local_view.reshape(1, local_len, 1)

            # Features auxiliares (10): combinar parâmetros estelares + estatísticas do sinal
            aux_vec = self._build_auxiliary_features(candidate, features)
            X_aux = aux_vec.reshape(1, 10)

            # Predição do modelo (TensorFlow)
            prediction_start = datetime.now()
            probs = self.detector.model.predict([X_global, X_local, X_aux], verbose=0)[0]
            metrics.prediction_time = (datetime.now() - prediction_start).total_seconds()

            # Pós-processamento
            probs = np.clip(probs, 1e-6, 1.0)
            probs = probs / np.sum(probs)
            probabilities = dict(zip(self.classes, probs.astype(float).tolist()))
            predicted_class = self.classes[int(np.argmax(probs))]
            confidence = float(np.max(probs))

            metrics.total_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(metrics.total_time)

            result = OptimizedPredictionResult(
                target_name=candidate.target_name,
                prediction=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                processing_time=metrics.total_time,
                timestamp=datetime.now(),
                model_version="1.0.0",
                features_used=list(features.keys()),
                quality_score=metrics.data_quality_score
            )

            logger.info(f"Predição DL concluída para {candidate.target_name}: {predicted_class} ({confidence:.3f})")
            return result

        except Exception as e:
            error_msg = f"Erro na predição para {candidate.target_name}: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(candidate.target_name, str(e))

    def _build_auxiliary_features(self, candidate: OptimizedExoplanetCandidate, features: Dict[str, float]) -> np.ndarray:
        """Monta vetor de 10 features auxiliares (estelares + estatísticas)"""
        sp = candidate.stellar_params
        aux = [
            (getattr(sp, 'teff', None) or 0.0) if sp else 0.0,
            (getattr(sp, 'logg', None) or 0.0) if sp else 0.0,
            (getattr(sp, 'feh', None) or 0.0) if sp else 0.0,
            (getattr(sp, 'radius', None) or 0.0) if sp else 0.0,
            (getattr(sp, 'mass', None) or 0.0) if sp else 0.0,
            float(features.get('depth_estimate', 0.0)),
            float(features.get('snr_estimate', 0.0)),
            float(features.get('cv', 0.0)),
            float(features.get('skewness', 0.0)),
            float(features.get('kurtosis', 0.0)),
        ]
        return np.array(aux, dtype=np.float32)

    def _calculate_data_quality(self, features: Dict[str, float]) -> float:
        """Calcula score de qualidade dos dados"""
        try:
            snr = features.get('snr_estimate', 0.0)
            cv = features.get('cv', 1.0)
            skew = features.get('skewness', 0.0)
            kurt = features.get('kurtosis', 0.0)

            # Substituir quaisquer NaN/Inf por valores seguros
            snr = float(np.nan_to_num(snr, nan=0.0, posinf=1.0, neginf=0.0))
            cv = float(np.nan_to_num(cv, nan=1.0, posinf=1.0, neginf=0.0))
            skew = float(np.nan_to_num(skew, nan=0.0, posinf=1.0, neginf=-1.0))
            kurt = float(np.nan_to_num(kurt, nan=0.0, posinf=1.0, neginf=-1.0))

            quality_factors = [
                min(snr / 10.0, 1.0),
                1.0 - min(abs(cv), 1.0),
                1.0 - min(abs(skew) / 5.0, 1.0),
                1.0 - min(abs(kurt) / 10.0, 1.0)
            ]

            # Garantir que todos os fatores são válidos
            quality_factors = [float(np.nan_to_num(f, nan=0.5, posinf=1.0, neginf=0.0)) for f in quality_factors]

            score = float(np.clip(np.mean(quality_factors), 0.0, 1.0))

            # Verificação final: se ainda for NaN ou Inf, retornar valor padrão
            if not np.isfinite(score):
                return 0.5

            return score
        except Exception as e:
            logger.warning(f"Erro ao calcular qualidade dos dados: {e}")
            return 0.5

    def _update_stats(self, processing_time: float) -> None:
        """Atualiza estatísticas internas"""
        self._prediction_count += 1
        self._total_processing_time += processing_time
        self._metrics["total_predictions"] = self._prediction_count
        self._metrics["avg_processing_time"] = self._total_processing_time / self._prediction_count
        self._metrics["last_updated"] = datetime.now()

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas atualizadas do modelo"""
        return {
            **self._metrics,
            "cache_stats": prediction_cache.stats(),
            "processing_stats": {
                "total_predictions": self._prediction_count,
                "avg_processing_time": self._total_processing_time / max(self._prediction_count, 1),
                "total_processing_time": self._total_processing_time
            }
        }

    async def predict_batch(self, candidates: List[OptimizedExoplanetCandidate]) -> List[OptimizedPredictionResult]:
        """Predição em lote otimizada"""
        start_time = datetime.now()

        # Processamento paralelo limitado para evitar sobrecarga
        semaphore = asyncio.Semaphore(5)

        async def predict_with_semaphore(candidate):
            async with semaphore:
                return await self.predict(candidate)

        # Execução paralela
        tasks = [predict_with_semaphore(candidate) for candidate in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtrar exceções e logar erros
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Erro na predição do candidato {candidates[i].target_name}: {result}")
            else:
                valid_results.append(result)

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Predição em lote concluída: {len(valid_results)}/{len(candidates)} sucessos em {total_time:.2f}s")

        return valid_results
