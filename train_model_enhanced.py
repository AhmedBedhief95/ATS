"""
train_model_enhanced.py
Production-Grade Training Pipeline
Fixed: HTML stripping, duplicate columns, safe DataFrame access
"""

import pandas as pd
import joblib
import os
import json
import re
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore', category=FutureWarning)

HTML_TAG_RE = re.compile(r'<[^>]+>')
HTML_ENTITY_RE = re.compile(r'&\w+;|&#\d+;')


def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = HTML_TAG_RE.sub(' ', text)
    text = HTML_ENTITY_RE.sub(' ', text)
    return text


@dataclass
class TrainingConfig:
    resume_csv: str = 'dataset/resume/resume.csv'
    salary_csv: str = 'dataset/salary/job_salary_prediction_dataset.csv'
    hiring_csv: str = 'dataset/hiring/historical_decisions.csv'
    model_dir: str = 'models'

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    max_tfidf_features: int = 8000
    tfidf_ngram_range: Tuple[int, int] = (1, 3)
    use_svd: bool = True
    svd_components: int = 300
    use_stacking: bool = False

    salary_poly_features: bool = True
    salary_poly_degree: int = 2

    use_optuna: bool = False
    optuna_trials: int = 50

    build_embedding_index: bool = False
    embedding_model: str = 'all-MiniLM-L6-v2'

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelMetrics:
    model_name: str
    timestamp: str
    training_duration_seconds: float
    config: Dict

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None

    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None

    top_features: List[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class TextPreprocessor:
    NOISE_PATTERNS = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        r'linkedin\.com/\S+',
        r'github\.com/\S+',
        r'https?://\S+',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
    ]

    SECTION_HEADERS = {
        'experience': ['experience', 'work experience', 'employment', 'work history', 'professional experience'],
        'education': ['education', 'academic', 'qualifications', 'degree'],
        'skills': ['skills', 'technical skills', 'competencies', 'technologies', 'expertise'],
        'projects': ['projects', 'portfolio', 'personal projects'],
        'certifications': ['certifications', 'certificates', 'licenses', 'accreditations'],
        'summary': ['summary', 'objective', 'profile', 'about']
    }

    def __init__(self):
        self.section_pattern = re.compile(
            r'\b(' + '|'.join(
                [h for headers in self.SECTION_HEADERS.values() for h in headers]
            ) + r')\b',
            re.IGNORECASE
        )

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = strip_html(text)

        for pattern in self.NOISE_PATTERNS:
            text = re.sub(pattern, ' ', text)

        text = re.sub(r'\s+', ' ', text)
        text = self._normalize_sections(text)
        text = re.sub(r'[•·⋅∙◦▪▫]', ', ', text)

        return text.strip().lower()

    def _normalize_sections(self, text: str) -> str:
        def replace_header(match):
            header = match.group(1).lower()
            for section, headers in self.SECTION_HEADERS.items():
                if header in headers:
                    return f" [{section.upper()}] "
            return match.group(0)

        return self.section_pattern.sub(replace_header, text)

    def extract_structured_features(self, text: str) -> Dict[str, Any]:
        clean_text = strip_html(text) if isinstance(text, str) else ""

        features = {
            'text_length': len(clean_text),
            'word_count': len(clean_text.split()),
            'sentence_count': len(re.split(r'[.!?]+', clean_text)),
            'section_count': len(re.findall(r'\[([A-Z_]+)\]', clean_text)),
            'has_experience_section': int('[EXPERIENCE]' in clean_text.upper()),
            'has_education_section': int('[EDUCATION]' in clean_text.upper()),
            'has_skills_section': int('[SKILLS]' in clean_text.upper()),
            'has_projects_section': int('[PROJECTS]' in clean_text.upper()),
            'has_certifications_section': int('[CERTIFICATIONS]' in clean_text.upper()),
            'bullet_count': len(re.findall(r'[•·⋅∙◦▪▫]', clean_text)),
            'number_count': len(re.findall(r'\d+', clean_text)),
            'email_present': int(bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', clean_text))),
            'url_present': int(bool(re.search(r'https?://\S+', clean_text))),
            'linkedin_present': int(bool(re.search(r'linkedin', clean_text, re.IGNORECASE))),
            'github_present': int(bool(re.search(r'github', clean_text, re.IGNORECASE))),
        }

        exp_indicators = ['year', 'years', 'yr', 'yrs', 'month', 'months', 'experience']
        features['experience_mentions'] = sum(1 for w in exp_indicators if w in clean_text.lower())

        edu_indicators = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'bs', 'ba', 'ms', 'mba']
        features['education_mentions'] = sum(1 for w in edu_indicators if w in clean_text.lower())

        return features


class CategoryClassifierTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.metrics = None
        self.model_artifact = None

    def load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        print(f"📂 Loading resume dataset from {self.config.resume_csv}...")

        if not os.path.exists(self.config.resume_csv):
            print(f"❌ Dataset not found: {self.config.resume_csv}")
            return None, None

        df = pd.read_csv(self.config.resume_csv)

        required_cols = ['Resume_str', 'Category']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"❌ Missing columns: {missing}")
            return None, None

        df['Resume_str'] = df['Resume_str'].fillna('').astype(str)
        df['processed_text'] = df['Resume_str'].apply(self.preprocessor.clean)

        initial = len(df)
        df = df[df['processed_text'].str.len() > 20]
        if initial - len(df) > 0:
            print(f"  Removed {initial - len(df)} empty/short resumes")

        print("  Extracting structured features...")
        struct_features = df['Resume_str'].apply(self.preprocessor.extract_structured_features)
        struct_df = pd.DataFrame(struct_features.tolist())

        df = pd.concat([
            df[['Resume_str', 'Category', 'processed_text']].reset_index(drop=True),
            struct_df.reset_index(drop=True)
        ], axis=1)

        print(f"✓ Loaded {len(df)} resumes across {df['Category'].nunique()} categories")
        print(f"  Category distribution:\n{df['Category'].value_counts().head(10)}")

        return df, df['Category']

    def train(self, df: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        import time
        start_time = time.time()

        X_text = df['processed_text']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['processed_text', 'Resume_str', 'Category']]
        X_struct = df[numeric_cols].copy()

        print(f"  Using {len(numeric_cols)} numeric structured features: {numeric_cols}")

        X_text_train, X_text_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
            X_text, X_struct, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        print("\n🔤 Fitting TF-IDF Vectorizer...")
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=self.config.max_tfidf_features,
            ngram_range=self.config.tfidf_ngram_range,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )

        X_train_tfidf = tfidf.fit_transform(X_text_train)
        X_test_tfidf = tfidf.transform(X_text_test)
        print(f"  TF-IDF shape: {X_train_tfidf.shape}")

        X_struct_train_arr = X_struct_train.values.astype(np.float64)
        X_struct_test_arr = X_struct_test.values.astype(np.float64)

        if self.config.use_svd:
            print(f"  Applying SVD (n_components={self.config.svd_components})...")
            svd = TruncatedSVD(n_components=self.config.svd_components, random_state=self.config.random_state)
            X_train_dense = svd.fit_transform(X_train_tfidf)
            X_test_dense = svd.transform(X_test_tfidf)

            X_train_combined = np.hstack([X_train_dense, X_struct_train_arr])
            X_test_combined = np.hstack([X_test_dense, X_struct_test_arr])
        else:
            X_train_combined = X_train_tfidf
            X_test_combined = X_test_tfidf

        print(f"  Final feature shape: {X_train_combined.shape}")
        print(f"  Feature dtype: {X_train_combined.dtype}")

        if isinstance(X_train_combined, np.ndarray):
            assert not np.isnan(X_train_combined).any(), "NaN values found in training data!"
            assert not np.isinf(X_train_combined).any(), "Inf values found in training data!"

        print(f"\n📊 Running {self.config.cv_folds}-fold cross-validation...")
        cv_clf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=self.config.random_state,
            n_jobs=-1
        )

        cv_scores = cross_val_score(
            cv_clf, X_train_combined, y_train,
            cv=StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state),
            scoring='f1_weighted',
            n_jobs=-1
        )
        print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

        print("\n🌲 Training final classifier...")
        final_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            min_samples_split=5,
            class_weight='balanced',
            random_state=self.config.random_state,
            n_jobs=-1
        )
        final_clf.fit(X_train_combined, y_train)

        print("  Calibrating probabilities...")
        calibrated_clf = CalibratedClassifierCV(final_clf, method='isotonic', cv='prefit')
        calibrated_clf.fit(X_test_combined, y_test)

        print("\n📈 Evaluating model...")
        y_pred = calibrated_clf.predict(X_test_combined)
        y_pred_proba = calibrated_clf.predict_proba(X_test_combined)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"\n  Test Set Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        top_features = []
        if hasattr(final_clf, 'feature_importances_'):
            if self.config.use_svd:
                feature_names = [f"svd_{i}" for i in range(self.config.svd_components)]
            else:
                feature_names = list(tfidf.get_feature_names_out())
            feature_names += list(X_struct.columns)

            importances = final_clf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:30]
            top_features = [
                {'name': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
                 'importance': float(importances[idx])}
                for idx in top_idx
            ]

            print(f"\n  Top 10 important features:")
            for feat in top_features[:10]:
                print(f"    {feat['name']}: {feat['importance']:.4f}")

        duration = time.time() - start_time
        self.metrics = ModelMetrics(
            model_name='category_classifier',
            timestamp=datetime.now().isoformat(),
            training_duration_seconds=duration,
            config=self.config.to_dict(),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            top_features=top_features
        )

        self.model_artifact = {
            'tfidf': tfidf,
            'svd': svd if self.config.use_svd else None,
            'classifier': calibrated_clf,
            'base_classifier': final_clf,
            'feature_names': list(X_struct.columns),
            'classes': list(calibrated_clf.classes_)
        }

        return self.model_artifact

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.model_artifact['tfidf'], os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        if self.model_artifact['svd'] is not None:
            joblib.dump(self.model_artifact['svd'], os.path.join(model_dir, 'svd_transformer.pkl'))
        joblib.dump(self.model_artifact['classifier'], os.path.join(model_dir, 'category_classifier.pkl'))

        metadata = {
            'classes': self.model_artifact['classes'],
            'feature_names': self.model_artifact['feature_names'],
            'use_svd': self.config.use_svd,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics.to_dict() if self.metrics else {}
        }
        with open(os.path.join(model_dir, 'classifier_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n💾 Saved category classifier to {model_dir}/")


class SalaryPredictorTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics = None
        self.model_artifact = None

    def load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        print(f"\n📂 Loading salary dataset from {self.config.salary_csv}...")

        if not os.path.exists(self.config.salary_csv):
            print(f"⚠️ Salary dataset not found: {self.config.salary_csv}")
            return None, None

        df = pd.read_csv(self.config.salary_csv)

        if 'salary' not in df.columns:
            print("❌ Salary column not found")
            return None, None

        print(f"✓ Loaded {len(df)} salary records")
        print(f"  Salary range: ${df['salary'].min():,.0f} - ${df['salary'].max():,.0f}")

        return df.drop(columns=['salary']), df['salary']

    def _deduplicate_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(X.columns) != len(set(X.columns)):
            print(f"  ⚠️ Duplicate columns detected: {X.columns[X.columns.duplicated()].tolist()}")
            cols = pd.Series(X.columns)
            for dup in cols[cols.duplicated()].unique():
                mask = cols == dup
                cols.loc[mask] = [f"{dup}_{i}" if i > 0 else dup for i in range(mask.sum())]
            X.columns = cols
        return X

    def _safe_get_column(self, X: pd.DataFrame, col_name: str) -> Optional[pd.Series]:
        if col_name not in X.columns:
            return None
        result = X[col_name]
        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0]
        return result

    def engineer_features(self, X: pd.DataFrame, y: pd.Series, fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
        X = X.copy()
        X = self._deduplicate_columns(X)

        encoders = {} if fit else None

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        print(f"  Numeric features: {numeric_cols}")
        print(f"  Categorical features: {categorical_cols}")

        for col in categorical_cols:
            unique_count = X[col].nunique()

            le = LabelEncoder()
            if fit:
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = {'type': 'label', 'encoder': le}
            else:
                le = encoders[col]['encoder']
                X[col] = X[col].astype(str).apply(lambda x: x if x in le.classes_ else 'UNKNOWN')
                if 'UNKNOWN' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'UNKNOWN')
                X[col] = le.transform(X[col])

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if self.config.salary_poly_features and len(numeric_cols) >= 2:
            print("  Generating polynomial features...")
            poly = PolynomialFeatures(degree=self.config.salary_poly_degree, include_bias=False, interaction_only=True)
            numeric_data = X[numeric_cols].fillna(0)
            poly_features = poly.fit_transform(numeric_data)
            poly_names = poly.get_feature_names_out(numeric_cols)

            poly_df = pd.DataFrame(poly_features, columns=poly_names, index=X.index)
            correlations = poly_df.corrwith(y).abs().sort_values(ascending=False)
            top_poly = correlations.head(20).index.tolist()

            X = pd.concat([X, poly_df[top_poly]], axis=1)
            if fit:
                encoders['poly'] = {'features': top_poly, 'transformer': poly}

        exp_col = self._safe_get_column(X, 'experience_years')
        if exp_col is not None:
            X.loc[:, 'experience_squared'] = exp_col.values ** 2
            X.loc[:, 'experience_log'] = np.log1p(np.clip(exp_col.values, 0, None))

        skills_col = self._safe_get_column(X, 'skills_count')
        exp_col = self._safe_get_column(X, 'experience_years')
        if skills_col is not None and exp_col is not None:
            X.loc[:, 'skills_per_year'] = skills_col.values / np.clip(exp_col.values, 1, None)

        X = self._deduplicate_columns(X)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        return X, encoders if fit else None

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        import time
        start_time = time.time()

        print("\n🔧 Engineering features...")
        X_engineered, encoders = self.engineer_features(X, y, fit=True)

        non_numeric = X_engineered.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"  ⚠️ Dropping non-numeric columns: {non_numeric}")
            X_engineered = X_engineered.drop(columns=non_numeric)

        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        print(f"  Training: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Features: {X_engineered.shape[1]}")
        print(f"  Feature names: {list(X_engineered.columns)[:10]}...")

        print("\n🌲 Training salary models...")

        models = {
            'rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        }

        if XGBOOST_AVAILABLE:
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.random_state,
                n_jobs=-1
            )

        predictions = {}
        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            predictions[name] = preds

            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

            print(f"    {name.upper()} - MAE: ${mae:,.0f}, RMSE: ${rmse:,.0f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")

        if len(predictions) > 1:
            ensemble_preds = np.mean(list(predictions.values()), axis=0)
            final_model = models
            final_preds = ensemble_preds
        else:
            final_model = models['rf']
            final_preds = predictions['rf']

        top_features = []
        if hasattr(models.get('rf'), 'feature_importances_'):
            importances = models['rf'].feature_importances_
            feature_names = X_engineered.columns
            top_idx = np.argsort(importances)[::-1][:20]
            top_features = [
                {'feature': feature_names[idx], 'importance': float(importances[idx])}
                for idx in top_idx
            ]

        duration = time.time() - start_time
        self.metrics = ModelMetrics(
            model_name='salary_predictor',
            timestamp=datetime.now().isoformat(),
            training_duration_seconds=duration,
            config=self.config.to_dict(),
            mae=float(mean_absolute_error(y_test, final_preds)),
            rmse=float(np.sqrt(mean_squared_error(y_test, final_preds))),
            r2=float(r2_score(y_test, final_preds)),
            mape=float(np.mean(np.abs((y_test - final_preds) / y_test)) * 100),
            top_features=top_features
        )

        self.model_artifact = {
            'models': final_model if isinstance(final_model, dict) else {'rf': final_model},
            'encoders': encoders,
            'feature_order': list(X_engineered.columns),
            'feature_engineering_params': {
                'poly_features': self.config.salary_poly_features,
                'poly_degree': self.config.salary_poly_degree
            }
        }

        return self.model_artifact

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)

        if isinstance(self.model_artifact['models'], dict):
            for name, model in self.model_artifact['models'].items():
                joblib.dump(model, os.path.join(model_dir, f'salary_{name}.pkl'))

            # Save primary RF model as salary_predictor.pkl for backward compatibility
            if 'rf' in self.model_artifact['models']:
                joblib.dump(self.model_artifact['models']['rf'], os.path.join(model_dir, 'salary_predictor.pkl'))
            else:
                joblib.dump(list(self.model_artifact['models'].values())[0], os.path.join(model_dir, 'salary_predictor.pkl'))
        else:
            joblib.dump(self.model_artifact['models'], os.path.join(model_dir, 'salary_predictor.pkl'))

        joblib.dump(self.model_artifact['encoders'], os.path.join(model_dir, 'salary_encoders.pkl'))
        joblib.dump(self.model_artifact['feature_order'], os.path.join(model_dir, 'salary_feature_order.pkl'))

        metadata = {
            'feature_order': self.model_artifact['feature_order'],
            'feature_engineering': self.model_artifact['feature_engineering_params'],
            'metrics': self.metrics.to_dict() if self.metrics else {},
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(model_dir, 'salary_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n💾 Saved salary predictor to {model_dir}/")


class TrainingPipeline:
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.results = {}

    def run(self):
        print("\n" + "="*70)
        print("🚀 Production Training Pipeline")
        print("="*70)

        os.makedirs(self.config.model_dir, exist_ok=True)

        print("\n" + "-"*70)
        print("📋 PHASE 1: Category Classifier")
        print("-"*70)

        cat_trainer = CategoryClassifierTrainer(self.config)
        df, y = cat_trainer.load_data()

        if df is not None:
            artifact = cat_trainer.train(df, y)
            cat_trainer.save(self.config.model_dir)
            self.results['category'] = {
                'metrics': cat_trainer.metrics.to_dict() if cat_trainer.metrics else {},
                'model_path': os.path.join(self.config.model_dir, 'category_classifier.pkl')
            }

        print("\n" + "-"*70)
        print("💰 PHASE 2: Salary Predictor")
        print("-"*70)

        sal_trainer = SalaryPredictorTrainer(self.config)
        X_sal, y_sal = sal_trainer.load_data()

        if X_sal is not None:
            artifact = sal_trainer.train(X_sal, y_sal)
            sal_trainer.save(self.config.model_dir)
            self.results['salary'] = {
                'metrics': sal_trainer.metrics.to_dict() if sal_trainer.metrics else {},
                'model_path': os.path.join(self.config.model_dir, 'salary_predictor.pkl')
            }

        self._save_summary()

        print("\n" + "="*70)
        print("✅ Training Complete!")
        print("="*70)
        self._print_summary()

    def _save_summary(self):
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'results': self.results
        }

        with open(os.path.join(self.config.model_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    def _print_summary(self):
        print("\n📊 Training Summary:")
        for phase, data in self.results.items():
            print(f"\n  {phase.upper()}:")
            if 'metrics' in data:
                metrics = data['metrics']
                for key, value in metrics.items():
                    if value is not None and key not in ['timestamp', 'config', 'top_features', 'model_name']:
                        if isinstance(value, float):
                            print(f"    {key}: {value:.4f}")
                        else:
                            print(f"    {key}: {value}")
            if 'model_path' in data:
                print(f"    Model: {data['model_path']}")


# Backward compatible API
def train_category_classifier(csv_path: str = 'dataset/resume/resume.csv',
                               model_dir: str = 'models') -> bool:
    config = TrainingConfig(resume_csv=csv_path, model_dir=model_dir)
    trainer = CategoryClassifierTrainer(config)

    df, y = trainer.load_data()
    if df is None:
        return False

    trainer.train(df, y)
    trainer.save(model_dir)
    return True


def train_salary_predictor(csv_path: str = 'dataset/salary/job_salary_prediction_dataset.csv',
                            model_dir: str = 'models') -> bool:
    config = TrainingConfig(salary_csv=csv_path, model_dir=model_dir)
    trainer = SalaryPredictorTrainer(config)

    X, y = trainer.load_data()
    if X is None:
        return False

    trainer.train(X, y)
    trainer.save(model_dir)
    return True


if __name__ == "__main__":
    config = TrainingConfig(
        use_optuna=False,
        use_stacking=False,
        build_embedding_index=False
    )

    pipeline = TrainingPipeline(config)
    pipeline.run()