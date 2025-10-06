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
import json
from tensorflow.keras.models import load_model

from backend.models import OptimizedExoplanetCandidate, OptimizedPredictionResult
from backend.cache import cached_prediction, prediction_cache
from backend.exceptions import ModelNotLoadedError, ProcessingError, PredictionError
from backend.config import settings
from backend.exoplanet_detector_model import ExoplanetDetector

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Processing metrics"""
    preprocessing_time: float = 0.0
    feature_extraction_time: float = 0.0
    prediction_time: float = 0.0
    total_time: float = 0.0
    data_quality_score: float = 0.0


class DataPreprocessor:
    """Optimized data preprocessor"""

    def __init__(self):
        self.scaler = RobustScaler()
        self._fitted = False

    def preprocess_light_curve(self, flux_data: List[float], length: int = None) -> np.ndarray:
        """Optimized light curve preprocessing"""
        start_time = datetime.now()

        try:
            if length is None:
                length = 2001

            flux_array = np.array(flux_data, dtype=np.float32)

            # Robust normalization using percentiles
            q25, q75 = np.percentile(flux_array, [25, 75])
            iqr = q75 - q25
            if iqr > 0:
                flux_normalized = (flux_array - np.median(flux_array)) / iqr
            else:
                flux_normalized = flux_array - np.median(flux_array)

            # Outlier detection and removal using modified z-score
            mad = np.median(np.abs(flux_normalized - np.median(flux_normalized)))
            if mad > 0:
                modified_z_scores = 0.6745 * (flux_normalized - np.median(flux_normalized)) / mad
                flux_clean = np.where(np.abs(modified_z_scores) < 3.5, flux_normalized, np.median(flux_normalized))
            else:
                flux_clean = flux_normalized

            # Resize to fixed length
            flux_processed = self._resize_sequence(flux_clean, length)

            return flux_processed

        except Exception as e:
            raise ProcessingError("preprocessamento de curva de luz", str(e))

    def _resize_sequence(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Resize sequence while preserving important characteristics"""
        if len(data) == target_length:
            return data
        elif len(data) > target_length:
            # Smart downsampling while preserving transit characteristics
            indices = np.linspace(0, len(data) - 1, target_length, dtype=int)
            return data[indices]
        else:
            # Simple padding
            return np.pad(data, (0, target_length - len(data)), 'constant')


class FeatureExtractor:
    """Optimized feature extractor"""

    @staticmethod
    def extract_statistical_features(flux_data: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from light curve"""
        try:
            mean = float(np.mean(flux_data))
            std = float(np.std(flux_data))
            min_v = float(np.min(flux_data))
            max_v = float(np.max(flux_data))
            median = float(np.median(flux_data))

            # Handle degenerate cases (constant signal)
            if std < 1e-12:
                skew = 0.0
                kurt = 0.0
                cv = 0.0
                autocorr_lag1 = 0.0
            else:
                skew = float(stats.skew(flux_data))
                kurt = float(stats.kurtosis(flux_data))
                cv = float(np.std(flux_data) / (abs(mean) + 1e-8))
                # Autocorrelation lag-1 with NaN protection
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

            # Sanitize values to avoid NaN/Inf
            for k, v in list(features.items()):
                features[k] = float(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0))

            return features

        except Exception as e:
            raise ProcessingError("extração de características estatísticas", str(e))


class IExoplanetDetector(ABC):
    """Interface for exoplanet detectors"""

    @abstractmethod
    async def predict(self, candidate: OptimizedExoplanetCandidate) -> OptimizedPredictionResult:
        """Predict candidate classification"""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Return model metrics"""
        pass


class OptimizedExoplanetDetectorService(IExoplanetDetector):
    """Optimized exoplanet detection service"""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_extractor = FeatureExtractor()
        # Load real Deep Learning model
        self.detector: Optional[ExoplanetDetector] = None
        self.model_loaded = False
        self.classes = ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
        self._prediction_count = 0
        self._total_processing_time = 0.0

        # Model metrics (initialization)
        self._metrics = {
            "accuracy": None,
            "precision": {},
            "recall": {},
            "f1_score": {},
            "total_predictions": 0,
            "last_updated": datetime.now()
        }

        # Initialize and load model
        self._init_model()

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model for troubleshooting"""
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
        """Initialize architecture and load DL model (.keras)"""
        try:
            # Initialize detector wrapper (for preprocessing consistency)
            self.detector = ExoplanetDetector(
                sequence_length=settings.sequence_length,
                n_features=settings.n_features,
                n_classes=settings.n_classes
            )

            # Load full .keras model
            full_path = settings.model_full_path
            if os.path.exists(full_path):
                self.detector.model = load_model(full_path)
                self.model_loaded = True
                logger.info(f"Full model (.keras) successfully loaded from: {full_path}")

                # Load metrics from model_metadata.json file
                metadata_path = "models/model_metadata.json"
                self._load_metrics_from_metadata(metadata_path)
            else:
                self.model_loaded = False
                logger.error(f".keras model not found at: {full_path}")
        except Exception as e:
            logger.exception(f"Failed to initialize/load .keras model: {e}")
            self.model_loaded = False

    def _load_metrics_from_metadata(self, metadata_path: str) -> None:
        """Load validation metrics from model_metadata.json file"""
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Extract validation metrics
                validation = metadata.get("validation", {})
                report = validation.get("report", {})

                # Update service metrics
                self._metrics["accuracy"] = validation.get("accuracy", 0.0)

                # Extract precision, recall and f1-score per class
                for class_name in self.classes:
                    if class_name in report:
                        self._metrics["precision"][class_name] = report[class_name].get("precision", 0.0)
                        self._metrics["recall"][class_name] = report[class_name].get("recall", 0.0)
                        self._metrics["f1_score"][class_name] = report[class_name].get("f1-score", 0.0)

                self._metrics["last_updated"] = datetime.now()
                logger.info(f"Validation metrics loaded: Accuracy={self._metrics['accuracy']:.4f}")
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
        except Exception as e:
            logger.error(f"Error loading metrics from metadata: {e}")

    @cached_prediction(prediction_cache, ttl=3600)
    async def predict(self, candidate: OptimizedExoplanetCandidate) -> OptimizedPredictionResult:
        """Prediction with real Deep Learning model and metrics"""
        if not self.model_loaded or self.detector is None:
            raise ModelNotLoadedError()

        start_time = datetime.now()
        metrics = ProcessingMetrics()

        try:
            # Preprocessing consistent with training
            preprocessing_start = datetime.now()
            flux_array = np.array(candidate.light_curve.flux, dtype=np.float32)
            flux_processed = self.detector.preprocess_light_curve(flux_array, length=settings.sequence_length)
            metrics.preprocessing_time = (datetime.now() - preprocessing_start).total_seconds()

            # Auxiliary feature extraction (for quality and auxiliary features)
            feature_start = datetime.now()
            features = self.feature_extractor.extract_statistical_features(flux_processed)
            metrics.feature_extraction_time = (datetime.now() - feature_start).total_seconds()

            # Data quality evaluation
            metrics.data_quality_score = self._calculate_data_quality(features)

            # Build model inputs
            X_global = flux_processed.reshape(1, settings.sequence_length, 1)

            # Local view: window centered at the minimum (possible transit)
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

            # Auxiliary features (10): combine stellar params + signal statistics
            aux_vec = self._build_auxiliary_features(candidate, features)
            X_aux = aux_vec.reshape(1, 10)

            # Model prediction (TensorFlow)
            prediction_start = datetime.now()
            probs = self.detector.model.predict([X_global, X_local, X_aux], verbose=0)[0]
            metrics.prediction_time = (datetime.now() - prediction_start).total_seconds()

            # Post-processing
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

            logger.info(f"DL prediction finished for {candidate.target_name}: {predicted_class} ({confidence:.3f})")
            return result

        except Exception as e:
            error_msg = f"Prediction error for {candidate.target_name}: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(candidate.target_name, str(e))

    def _build_auxiliary_features(self, candidate: OptimizedExoplanetCandidate, features: Dict[str, float]) -> np.ndarray:
        """Build 10-element auxiliary feature vector (stellar + statistics)"""
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
        """Calculate data quality score"""
        try:
            snr = features.get('snr_estimate', 0.0)
            cv = features.get('cv', 1.0)
            skew = features.get('skewness', 0.0)
            kurt = features.get('kurtosis', 0.0)

            # Replace any NaN/Inf with safe values
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

            # Ensure all factors are valid
            quality_factors = [float(np.nan_to_num(f, nan=0.5, posinf=1.0, neginf=0.0)) for f in quality_factors]

            score = float(np.clip(np.mean(quality_factors), 0.0, 1.0))

            # Final check: if still NaN or Inf, return default value
            if not np.isfinite(score):
                return 0.5

            return score
        except Exception as e:
            logger.warning(f"Error calculating data quality: {e}")
            return 0.5

    def _update_stats(self, processing_time: float) -> None:
        """Update internal statistics"""
        self._prediction_count += 1
        self._total_processing_time += processing_time
        self._metrics["total_predictions"] = self._prediction_count
        self._metrics["avg_processing_time"] = self._total_processing_time / self._prediction_count
        self._metrics["last_updated"] = datetime.now()

    def get_metrics(self) -> Dict[str, Any]:
        """Return updated model metrics"""
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
        """Optimized batch prediction"""
        start_time = datetime.now()

        # Limited parallel processing to avoid overload
        semaphore = asyncio.Semaphore(5)

        async def predict_with_semaphore(candidate):
            async with semaphore:
                return await self.predict(candidate)

        # Parallel execution
        tasks = [predict_with_semaphore(candidate) for candidate in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Erro na predição do candidato {candidates[i].target_name}: {result}")
            else:
                valid_results.append(result)

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Batch prediction finished: {len(valid_results)}/{len(candidates)} successes in {total_time:.2f}s")

        return valid_results

    def _load_metrics_from_file(self, file_path: str) -> None:
        """Load model metrics from a JSON file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    metrics = json.load(f)
                    self._metrics.update(metrics)
                    self._metrics["last_updated"] = datetime.now()
                    logger.info(f"Model metrics loaded successfully from: {file_path}")
            else:
                logger.warning(f"Metrics file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading metrics from file {file_path}: {e}")
