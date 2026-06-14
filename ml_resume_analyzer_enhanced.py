"""
ml_resume_analyzer_enhanced.py
Production-Grade AI Resume Analysis Model with Enhanced Scoring
Includes job recommendation boost in the final match score.
"""

import re
import numpy as np
import joblib
import os
import json
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        sentence_model = None
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    sentence_model = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class SkillMatch:
    skill: str
    category: str
    context_score: float
    recency_score: float
    frequency: int
    proficiency_indicators: List[str] = field(default_factory=list)
    years_of_use: float = 0.0


@dataclass
class JobEntry:
    title: str
    company: str
    company_tier: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    is_current: bool
    description: str
    skills_found: List[str] = field(default_factory=list)
    years: float = 0.0


@dataclass
class EducationEntry:
    degree: str
    field: str
    institution: str
    tier: str
    graduation_date: Optional[datetime]
    level: int


class ProductionResumeAnalyzer:
    FAANG_COMPANIES = {
        'meta', 'facebook', 'google', 'alphabet', 'amazon', 'apple',
        'netflix', 'microsoft', 'twitter', 'x', 'linkedin', 'nvidia'
    }

    FORTUNE500_KEYWORDS = [
        'walmart', 'amazon', 'exxon', 'apple', 'unitedhealth', 'cvs', 'berkshire',
        'alphabet', 'mckesson', 'chevron', 'amerisource', 'costco', 'cigna',
        'microsoft', 'cardinal', 'walgreen', 'home depot', 'jpmorgan', 'verizon',
        'ford', 'general motors', 'anthem', 'target', 'comcast', 'metlife',
        'hca', 'philips', 'cisco', 'intel', 'oracle', 'salesforce', 'adobe',
        'ibm', 'dell', 'hp', 'accenture', 'deloitte', 'pwc', 'ey', 'kpmg'
    ]

    STARTUP_INDICATORS = ['startup', 'founded', 'seed', 'series a', 'series b', 'early stage']
    TOP_UNIVERSITIES = {
        'mit', 'stanford', 'harvard', 'caltech', 'princeton', 'yale', 'columbia',
        'chicago', 'pennsylvania', 'northwestern', 'duke', 'johns hopkins',
        'dartmouth', 'brown', 'vanderbilt', 'rice', 'notre dame', 'uc berkeley',
        'ucla', 'michigan', 'carnegie mellon', 'georgia tech', 'illinois', 'texas',
        'washington', 'wisconsin', 'cornell', 'oxford', 'cambridge', 'imperial',
        'eth zurich', 'tsinghua', 'peking', 'national university singapore'
    }

    CATEGORY_SKILL_WEIGHTS = {
        # ==========================================
        # CORE SOFTWARE DEVELOPMENT
        # ==========================================
        'Information-Technology': {
            'programming': 1.5, 'frontend': 1.3, 'backend': 1.3,
            'database': 1.2, 'devops': 1.2, 'data': 1.0, 'cloud': 1.2,
            'security': 0.9, 'architecture': 1.1, 'testing': 1.0, 'other': 0.8
        },
        'Software-Development': {
            'programming': 1.7, 'frontend': 1.4, 'backend': 1.5,
            'database': 1.2, 'devops': 1.1, 'data': 0.8, 'cloud': 1.1,
            'security': 0.9, 'architecture': 1.3, 'testing': 1.2, 'other': 1.0
        },
        'Frontend-Developer': {
            'programming': 1.4, 'frontend': 1.8, 'backend': 0.7,
            'database': 0.6, 'devops': 0.7, 'data': 0.5, 'cloud': 0.6,
            'security': 0.7, 'architecture': 0.9, 'testing': 1.2, 'other': 0.9
        },
        'Backend-Developer': {
            'programming': 1.6, 'frontend': 0.5, 'backend': 1.8,
            'database': 1.5, 'devops': 1.2, 'data': 0.9, 'cloud': 1.3,
            'security': 1.1, 'architecture': 1.4, 'testing': 1.2, 'other': 0.9
        },
        'Full-Stack-Developer': {
            'programming': 1.5, 'frontend': 1.5, 'backend': 1.5,
            'database': 1.3, 'devops': 1.2, 'data': 0.8, 'cloud': 1.1,
            'security': 1.0, 'architecture': 1.2, 'testing': 1.2, 'other': 1.0
        },
        'Mobile-Application-Developer': {
            'programming': 1.5, 'frontend': 1.5, 'backend': 1.1,
            'database': 1.0, 'devops': 1.0, 'data': 0.7, 'cloud': 1.0,
            'security': 0.9, 'architecture': 1.1, 'testing': 1.2, 'other': 1.0
        },
        'Embedded-Systems-Engineer': {
            'programming': 1.6, 'other': 1.2, 'backend': 1.1,
            'database': 0.7, 'devops': 0.8, 'data': 0.7, 'cloud': 0.5,
            'security': 1.0, 'architecture': 1.0, 'testing': 1.1, 'frontend': 0.4
        },
        'QA-Automation-Engineer': {
            'programming': 1.4, 'testing': 1.9, 'devops': 1.3,
            'backend': 1.1, 'database': 1.0, 'data': 0.7, 'cloud': 0.9,
            'security': 0.9, 'architecture': 0.8, 'other': 1.1, 'frontend': 1.0
        },

        # ==========================================
        # DATA, AI & ANALYTICS
        # ==========================================
        'Data-AI': {
            'data': 1.8, 'programming': 1.3, 'database': 1.2,
            'devops': 0.9, 'cloud': 1.2, 'frontend': 0.4,
            'security': 0.7, 'architecture': 0.9, 'testing': 0.7, 'other': 0.8
        },
        'Data-Scientist': {
            'data': 1.8, 'programming': 1.4, 'database': 1.1,
            'devops': 0.7, 'cloud': 1.0, 'frontend': 0.4,
            'security': 0.6, 'architecture': 0.8, 'testing': 0.7, 'other': 0.8
        },
        'Data-Engineer': {
            'data': 1.6, 'programming': 1.4, 'database': 1.5,
            'devops': 1.3, 'cloud': 1.4, 'frontend': 0.3,
            'security': 0.8, 'architecture': 1.2, 'testing': 1.0, 'other': 0.9
        },
        'Machine-Learning-Engineer': {
            'data': 1.8, 'programming': 1.5, 'database': 1.0,
            'devops': 1.1, 'cloud': 1.3, 'frontend': 0.4,
            'security': 0.7, 'architecture': 1.1, 'testing': 1.0, 'other': 0.8
        },
        'Database-Administrator': {
            'database': 1.8, 'data': 1.3, 'programming': 1.1,
            'devops': 1.2, 'cloud': 1.1, 'backend': 1.0,
            'security': 1.1, 'architecture': 1.1, 'testing': 0.9, 'frontend': 0.3
        },

        # ==========================================
        # INFRASTRUCTURE, DEVOPS & CLOUD
        # ==========================================
        'DevOps-Infrastructure': {
            'devops': 1.8, 'programming': 1.2, 'database': 1.0,
            'cloud': 1.6, 'backend': 1.1, 'data': 0.7,
            'security': 1.2, 'architecture': 1.2, 'testing': 1.0, 'frontend': 0.3
        },
        'DevOps-Engineer': {
            'devops': 1.8, 'programming': 1.2, 'database': 1.0,
            'cloud': 1.6, 'backend': 1.1, 'data': 0.7,
            'security': 1.2, 'architecture': 1.2, 'testing': 1.0, 'frontend': 0.3
        },
        'Site-Reliability-Engineer': {
            'devops': 1.7, 'programming': 1.3, 'database': 1.1,
            'cloud': 1.5, 'backend': 1.3, 'data': 0.8,
            'security': 1.2, 'architecture': 1.3, 'testing': 1.1, 'frontend': 0.4
        },
        'Platform-Engineer': {
            'devops': 1.6, 'programming': 1.3, 'cloud': 1.5,
            'database': 1.1, 'backend': 1.3, 'data': 0.8,
            'security': 1.1, 'architecture': 1.3, 'testing': 1.0, 'frontend': 0.4
        },
        'Cloud-Engineer': {
            'cloud': 1.8, 'devops': 1.5, 'programming': 1.2,
            'database': 1.0, 'backend': 1.0, 'data': 0.8,
            'security': 1.3, 'architecture': 1.3, 'testing': 0.9, 'frontend': 0.3
        },
        'Systems-Administrator': {
            'devops': 1.5, 'programming': 1.0, 'cloud': 1.2,
            'database': 1.1, 'backend': 0.9, 'data': 0.7,
            'security': 1.2, 'architecture': 1.0, 'testing': 0.8, 'frontend': 0.4
        },

        # ==========================================
        # CYBERSECURITY
        # ==========================================
        'Cybersecurity': {
            'security': 1.9, 'devops': 1.2, 'programming': 1.2,
            'cloud': 1.3, 'database': 1.0, 'backend': 1.1,
            'architecture': 1.2, 'testing': 1.1, 'data': 0.9, 'frontend': 0.7
        },
        'DevSecOps-Engineer': {
            'security': 1.7, 'devops': 1.6, 'cloud': 1.3, 'programming': 1.2,
            'database': 0.9, 'backend': 1.0,
            'architecture': 1.1, 'testing': 1.2, 'data': 0.8, 'frontend': 0.5
        },
        'Cloud-Security-Engineer': {
            'security': 1.8, 'cloud': 1.6, 'devops': 1.3, 'programming': 1.1,
            'database': 0.9, 'backend': 1.0,
            'architecture': 1.2, 'testing': 1.0, 'data': 0.8, 'frontend': 0.4
        },
        'Penetration-Tester': {
            'security': 1.9, 'programming': 1.4, 'devops': 1.2,
            'cloud': 1.1, 'backend': 1.2, 'database': 1.0,
            'architecture': 1.0, 'testing': 1.3, 'data': 0.8, 'frontend': 0.8
        },

        # ==========================================
        # NETWORKING & TELECOM
        # ==========================================
        'Networking-Telecom': {
            'devops': 1.3, 'cloud': 1.1, 'programming': 1.0,
            'database': 0.8, 'backend': 0.9, 'data': 0.7,
            'security': 1.3, 'architecture': 1.1, 'testing': 0.8, 'frontend': 0.4
        },
        'Network-Engineer': {
            'devops': 1.3, 'cloud': 1.1, 'programming': 1.0,
            'database': 0.8, 'backend': 0.9, 'data': 0.7,
            'security': 1.3, 'architecture': 1.1, 'testing': 0.8, 'frontend': 0.4
        },
        'Network-Architect': {
            'devops': 1.4, 'cloud': 1.2, 'programming': 1.1,
            'database': 0.9, 'backend': 1.0, 'data': 0.8,
            'security': 1.4, 'architecture': 1.4, 'testing': 0.8, 'frontend': 0.4
        },
        'Network-Security-Engineer': {
            'security': 1.8, 'devops': 1.4, 'cloud': 1.2, 'programming': 1.1,
            'database': 0.9, 'backend': 1.0,
            'architecture': 1.2, 'testing': 1.0, 'data': 0.8, 'frontend': 0.3
        },

        # ==========================================
        # TECH MANAGEMENT & ARCHITECTURE
        # ==========================================
        'Tech-Management': {
            'other': 1.3, 'programming': 1.2, 'devops': 1.1,
            'data': 1.1, 'cloud': 1.1, 'database': 1.0,
            'security': 1.0, 'architecture': 1.3, 'testing': 0.9, 'frontend': 0.9
        },
        'Solutions-Architect': {
            'cloud': 1.5, 'devops': 1.4, 'programming': 1.2,
            'database': 1.2, 'backend': 1.2, 'data': 1.0,
            'security': 1.2, 'architecture': 1.7, 'testing': 1.0, 'frontend': 0.7
        },
        'IT-Project-Manager': {
            'other': 1.4, 'programming': 1.1, 'devops': 1.1,
            'data': 1.0, 'cloud': 1.0, 'database': 1.0,
            'security': 0.9, 'architecture': 1.0, 'testing': 1.0, 'frontend': 0.8
        },
        'Product-Owner': {
            'other': 1.3, 'data': 1.2, 'programming': 1.1,
            'frontend': 1.1, 'backend': 1.0, 'devops': 1.0, 'database': 0.9,
            'security': 0.8, 'architecture': 1.0, 'testing': 0.9, 'cloud': 1.0
        },
        'Chief-Technology-Officer': {
            'other': 1.4, 'programming': 1.2, 'devops': 1.2,
            'cloud': 1.2, 'data': 1.1, 'database': 1.1,
            'security': 1.2, 'architecture': 1.5, 'testing': 1.0, 'frontend': 0.9
        },
        'VP-of-Engineering': {
            'other': 1.4, 'programming': 1.2, 'devops': 1.2,
            'cloud': 1.2, 'data': 1.1, 'database': 1.1,
            'security': 1.1, 'architecture': 1.4, 'testing': 1.0, 'frontend': 0.9
        },
        'IT-Operations-Manager': {
            'devops': 1.4, 'other': 1.3, 'cloud': 1.2,
            'programming': 1.1, 'database': 1.1, 'backend': 1.0,
            'security': 1.2, 'architecture': 1.1, 'testing': 0.9, 'data': 0.9
        },

        # ==========================================
        # GAMING, VFX & CREATIVE TECH
        # ==========================================
        'Gaming-VFX': {
            'programming': 1.4, 'frontend': 1.2, 'other': 1.2,
            'database': 0.9, 'devops': 0.8, 'data': 0.7, 'cloud': 0.8,
            'security': 0.7, 'architecture': 1.1, 'testing': 1.0, 'backend': 1.1
        },

        # ==========================================
        # EXISTING NON-IT CATEGORIES (expanded)
        # ==========================================
        'Engineering': {
            'programming': 1.2, 'other': 1.0, 'data': 0.9, 'devops': 0.9, 'cloud': 0.9,
            'security': 0.8, 'architecture': 1.0, 'testing': 0.9
        },
        'Finance': {
            'data': 1.3, 'programming': 1.0, 'other': 0.9,
            'database': 1.1, 'devops': 0.7, 'cloud': 0.8,
            'security': 0.8, 'architecture': 0.7, 'testing': 0.6
        },
        'Sales': {
            'other': 1.2, 'data': 1.0, 'programming': 0.7, 'database': 0.8,
            'security': 0.5, 'architecture': 0.5, 'testing': 0.4
        },
        'HR': {
            'other': 1.2, 'data': 1.0, 'programming': 0.7, 'database': 0.9,
            'security': 0.5, 'architecture': 0.5, 'testing': 0.4
        },
        'Healthcare': {
            'other': 1.2, 'data': 1.1, 'programming': 0.8, 'database': 1.0,
            'security': 0.9, 'architecture': 0.6, 'testing': 0.5
        },
        'Design': {
            'frontend': 1.3, 'other': 1.2, 'programming': 0.9, 'data': 0.7,
            'security': 0.5, 'architecture': 0.7, 'testing': 0.6
        },
        'Designer': {
            'frontend': 1.3, 'other': 1.2, 'programming': 0.9, 'data': 0.7,
            'security': 0.5, 'architecture': 0.7, 'testing': 0.6
        },
        'Business-Development': {
            'other': 1.2, 'data': 1.1, 'programming': 0.9, 'database': 0.9,
            'security': 0.5, 'architecture': 0.6, 'testing': 0.4
        },
        'Marketing': {
            'data': 1.2, 'frontend': 1.1, 'other': 1.2, 'programming': 0.8,
            'security': 0.5, 'architecture': 0.5, 'testing': 0.4
        },
        'default': {
            'programming': 1.0, 'frontend': 1.0, 'backend': 1.0,
            'database': 1.0, 'devops': 1.0, 'data': 1.0, 'cloud': 1.0,
            'security': 1.0, 'architecture': 1.0, 'testing': 1.0, 'other': 1.0
        }
    }

    def __init__(self, use_trained_model: bool = False, model_dir: str = 'models',
                 use_semantic: bool = True, use_spacy: bool = True):
        self.model_dir = model_dir
        self.use_semantic = use_semantic and SENTENCE_TRANSFORMERS_AVAILABLE and sentence_model is not None
        self.use_spacy = use_spacy and SPACY_AVAILABLE and nlp is not None
        self._embedding_cache = {}
        self._init_skills_database()
        self.salary_model = None
        self.salary_encoders = None
        self.salary_feature_order = None
        self._load_salary_model()

        self.use_trained_model = use_trained_model
        self.tfidf_vectorizer = None
        self.category_classifier = None 
        self.job_categories = [
            # ==========================================
            # TECH & INFRASTRUCTURE (Expanded)
            # ==========================================
            'Information-Technology',      # General IT, Helpdesk, Support
            'Software-Development',        # Frontend, Backend, Full-Stack, Mobile
            'DevOps-Infrastructure',       # DevOps, SRE, Platform, Cloud Engineers
            'Data-AI',                     # Data Scientists, ML, Analytics, DBAs
            'Networking-Telecom',          # Network Architects, Telecom Engineers
            'Cybersecurity',               # Security Analysts, Pen Testers, DevSecOps
            'Tech-Management',             # CTO, Scrum Masters, Product Owners
            'Gaming-VFX',                  # Game Dev, 3D Animators, VFX Artists
            
            # ==========================================
            # CORPORATE & BUSINESS OPERATIONS
            # ==========================================
            'HR',                          # HR Specialists, Recruiters, HRBP
            'Business-Development',        # BizDev, Partnerships, Strategy
            'Sales',                       # Inside/Outside Sales, Account Executives
            'Marketing',                   # Growth, Performance, SEO, Content Marketing
            'Public-Relations',            # PR, Communications, Media Relations
            'Consultant',                  # Strategy, Management, Tech Consulting
            'BPO',                         # Customer Support, Telemarketing, Call Centers
            'Legal-Advocate',              # Lawyers, Paralegals, Compliance Officers
            'Logistics-Supply-Chain',      # Procurement, Inventory, Operations
            
            # ==========================================
            # FINANCE & BANKING
            # ==========================================
            'Finance',                     # Financial Analysts, Corporate Finance
            'Accountant',                  # CPAs, Bookkeepers, Auditors
            'Banking',                     # Investment Banking, Tellers, Loan Officers
            'Insurance',                   # Underwriters, Actuaries, Claims Adjusters
            'Real-Estate',                 # Agents, Property Managers, Brokers

            # ==========================================
            # ENGINEERING & CONSTRUCTION
            # ==========================================
            'Engineering',                 # Civil, Mechanical, Electrical, Chemical
            'Construction',                # Construction Managers, Foremen, Safety
            'Architecture',                # Urban Planners, Architects, BIM Modelers
            
            # ==========================================
            # CREATIVE, DESIGN & MEDIA
            # ==========================================
            'Designer',                    # UI/UX, Graphic, Product Designers
            'Apparel',                     # Fashion Designers, Textile, Stylists
            'Digital-Media',               # Videographers, Editors, Podcasters
            'Arts',                        # Fine Artists, Musicians, Actors, Writers

            # ==========================================
            # HEALTHCARE & WELLNESS
            # ==========================================
            'Healthcare',                  # Doctors, Nurses, Medical Technicians
            'Pharmaceuticals',             # Pharmacists, Clinical Researchers
            'Fitness',                     # Personal Trainers, Nutritionists, Coaches
            'Mental-Health',               # Psychologists, Therapists, Counselors

            # ==========================================
            # EDUCATION, SOCIAL & PUBLIC SECTOR
            # ==========================================
            'Teacher',                     # K-12, Professors, Tutors, E-Learning
            'NGO-Non-Profit',              # Social Workers, Fundraising, Activists
            'Government-Public-Service',   # Civil Servants, Policy Analysts, Defense

            # ==========================================
            # SERVICES & BLUE COLLAR
            # ==========================================
            'Automobile',                  # Mechanics, Automotive Technicians
            'Chef',                        # Culinary Arts, Bakers, Kitchen Managers
            'Hospitality-Tourism',         # Hotel Staff, Travel Agents, Event Planners
            'Agriculture',                 # Farmers, Agronomists, Livestock Specialists
            'Aviation',                    # Pilots, Flight Attendants, Air Traffic Control
            'Maritime'                     # Shipping, Marine Engineers, Coast Guard
        ]   

        self.hiring_classifier = None
        self.hiring_scaler = None
        self._load_hiring_model()

        if use_trained_model:
            self.load_trained_models()

    # ---------------------------------------------------------------
    # IT SKILL TIER DEFINITIONS
    # ---------------------------------------------------------------
    IT_CORE_SKILLS = {
        'python', 'javascript', 'typescript', 'java', 'c#', 'c++', 'go', 'rust',
        'react', 'nodejs', 'sql', 'postgresql', 'mongodb', 'docker', 'kubernetes',
        'aws', 'azure', 'gcp', 'git', 'linux', 'rest api', 'microservices',
        'ci/cd', 'agile', 'scrum', 'terraform', 'django', 'spring boot', 'fastapi',
    }
    IT_SECONDARY_SKILLS = {
        'typescript', 'graphql', 'redis', 'kafka', 'elasticsearch', 'spark',
        'airflow', 'dbt', 'pytorch', 'tensorflow', 'scikit-learn', 'pandas',
        'numpy', 'github actions', 'gitlab ci', 'helm', 'ansible', 'vault',
        'grpc', 'websocket', 'oauth', 'jwt', 'openapi', 'swagger',
        'datadog', 'prometheus', 'grafana', 'sentry', 'new relic',
        'celery', 'rabbitmq', 'nats', 'pulsar',
    }
    IT_BONUS_SKILLS = {
        'wasm', 'webassembly', 'solid.js', 'qwik', 'htmx', 'deno', 'bun',
        'zig', 'nim', 'elixir', 'erlang', 'haskell', 'ocaml', 'clojure',
        'neo4j', 'dgraph', 'clickhouse', 'tidb', 'cockroachdb',
        'istio', 'envoy', 'linkerd', 'argo', 'flux', 'tekton',
        'triton', 'onnx', 'mlflow', 'kubeflow', 'feast', 'ray',
    }
    IT_CERT_BONUS_MAP = {
        'aws certified solutions architect': 8,
        'aws certified developer': 7,
        'aws certified': 6,
        'google cloud professional': 7,
        'azure solutions architect': 7,
        'azure developer': 6,
        'azure certified': 5,
        'gcp certified': 5,
        'certified kubernetes administrator': 8,
        'cka': 8,
        'ckad': 7,
        'cks': 8,
        'terraform associate': 5,
        'hashicorp certified': 5,
        'github actions': 4,
        'gitlab certified': 4,
        'docker certified': 5,
        'oracle certified': 5,
        'java certified': 5,
        'spring certified': 5,
        'comptia security': 5,
        'comptia network': 4,
        'comptia a+': 3,
        'ccna': 5,
        'ccnp': 7,
        'pmp': 4,
        'scrum master': 4,
        'certified scrum': 4,
        'safe': 3,
    }

    def _init_skills_database(self):
        self.technical_skills = {
            'programming': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                'php', 'ruby', 'swift', 'kotlin', 'scala', 'perl', 'r', 'matlab',
                'sql', 'plsql', 'nosql', 'graphql', 'bash', 'powershell', 'lua',
                'dart', 'elixir', 'erlang', 'haskell', 'ocaml', 'clojure', 'zig',
                'solidity', 'move', 'wasm', 'webassembly',
            ],
            'frontend': [
                'react', 'vue', 'angular', 'svelte', 'solid.js', 'qwik', 'htmx',
                'html', 'css', 'tailwind', 'bootstrap', 'material ui', 'shadcn',
                'webpack', 'vite', 'rollup', 'esbuild', 'parcel',
                'next.js', 'nuxt', 'remix', 'gatsby', 'astro',
                'flutter', 'ionic', 'react native', 'xamarin', 'expo',
                'storybook', 'chromatic', 'cypress', 'playwright', 'vitest',
                'pwa', 'web components', 'web3',
            ],
            'backend': [
                'django', 'flask', 'fastapi', 'starlette', 'litestar',
                'spring', 'spring boot', 'quarkus', 'micronaut', 'ktor',
                'express', 'nestjs', 'fastify', 'koa', 'hono',
                'laravel', 'symfony', 'asp.net', 'asp.net core', 'rails', 'sinatra',
                'gin', 'echo', 'fiber', 'actix', 'axum', 'rocket',
                'grpc', 'graphql', 'rest api', 'websocket', 'oauth', 'jwt',
                'openapi', 'swagger', 'celery', 'rq', 'dramatiq',
            ],
            'database': [
                'postgresql', 'mysql', 'oracle', 'mssql', 'sqlite', 'mariadb',
                'mongodb', 'firestore', 'couchdb', 'dynamodb',
                'redis', 'memcached', 'dragonfly',
                'elasticsearch', 'opensearch', 'solr', 'meilisearch', 'typesense',
                'cassandra', 'hbase', 'clickhouse', 'redshift', 'bigquery',
                'snowflake', 'databricks', 'duckdb', 'tidb', 'cockroachdb',
                'influxdb', 'timescaledb', 'questdb',
                'neo4j', 'dgraph', 'amazon neptune',
                'kafka', 'rabbitmq', 'nats', 'pulsar', 'sqs', 'pubsub',
            ],
            'devops': [
                'docker', 'kubernetes', 'helm', 'kustomize', 'podman',
                'aws', 'azure', 'gcp', 'cloudflare', 'vercel', 'netlify', 'fly.io',
                'terraform', 'pulumi', 'cloudformation', 'cdk', 'ansible', 'chef', 'puppet',
                'jenkins', 'gitlab ci', 'github actions', 'circleci', 'travis ci',
                'argocd', 'flux', 'tekton', 'spinnaker',
                'istio', 'envoy', 'linkerd', 'consul',
                'datadog', 'prometheus', 'grafana', 'loki', 'jaeger', 'opentelemetry',
                'new relic', 'dynatrace', 'sentry', 'pagerduty',
                'vault', 'aws secrets manager', 'sops',
                'devops', 'ci/cd', 'gitops', 'linux', 'bash', 'sre',
            ],
            'data': [
                'pandas', 'numpy', 'scipy', 'polars', 'dask',
                'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'jax', 'xgboost', 'lightgbm',
                'mlflow', 'kubeflow', 'sagemaker', 'vertex ai', 'weights and biases', 'dvc', 'feast',
                'ray', 'triton', 'onnx', 'bentoml', 'seldon',
                'spark', 'hadoop', 'hive', 'flink', 'beam', 'pig',
                'airflow', 'prefect', 'dagster', 'luigi', 'dbt', 'fivetran', 'airbyte',
                'looker', 'tableau', 'power bi', 'metabase', 'superset', 'mode',
                'data analysis', 'machine learning', 'deep learning', 'nlp',
                'natural language processing', 'computer vision', 'big data',
                'statistical analysis', 'a/b testing', 'feature engineering',
                'llm', 'generative ai', 'rag', 'langchain', 'llamaindex',
                'time series', 'recommendation systems',
            ],
            'security': [
                'owasp', 'penetration testing', 'siem', 'soc', 'devsecops',
                'zero trust', 'iam', 'rbac', 'mfa', 'ssl/tls', 'pki',
                'sonarqube', 'snyk', 'trivy', 'aqua security', 'falco',
                'nmap', 'burp suite', 'metasploit', 'wireshark',
                'iso 27001', 'soc 2', 'gdpr', 'hipaa', 'pci dss',
            ],
            'other': [
                'git', 'github', 'gitlab', 'bitbucket',
                'jira', 'confluence', 'notion', 'linear', 'asana',
                'agile', 'scrum', 'kanban', 'safe', 'shape up', 'xp',
                'unit testing', 'integration testing', 'e2e testing', 'tdd', 'bdd',
                'junit', 'pytest', 'jest', 'mocha', 'rspec', 'go test',
                'design patterns', 'solid principles', 'ddd', 'cqrs', 'event sourcing',
                'microservices', 'monorepo', 'api gateway', 'service mesh',
                'code review', 'pair programming', 'technical writing',
                'system design', 'architecture', 'performance optimization', 'debugging',
                'soap', 'xml', 'json', 'yaml', 'protobuf', 'avro',
            ],
        }

        self.multi_word_skills = {
            'ecommerce': ['amazon seller central', 'shopify store', 'ebay seller', 'woocommerce',
                         'magento', 'bigcommerce', 'prestashop', 'opencart', 'etsy seller',
                         'amazon fba', 'inventory management', 'product listing', 'sales optimization'],
            'business': ['business analysis', 'market research', 'product analysis', 'competitive analysis',
                        'business strategy', 'strategic planning', 'business development', 'market analysis',
                        'project management', 'vendor management', 'supplier management', 'account management',
                        'customer relations', 'stakeholder management', 'requirements gathering'],
            'design': ['web design', 'graphic design', 'ui design', 'ux design', 'interaction design',
                      'product design', 'visual design', 'responsive design', 'design thinking',
                      'adobe creative suite', 'figma', 'sketch', 'prototyping', 'wireframing',
                      'user research', 'usability testing'],
            'marketing': ['digital marketing', 'content marketing', 'social media marketing', 'email marketing',
                         'seo optimization', 'paid advertising', 'google ads', 'facebook ads',
                         'marketing automation', 'brand management', 'market positioning', 'customer acquisition',
                         'conversion optimization', 'growth hacking', 'analytics tracking'],
            'sales': ['sales management', 'territory management', 'pipeline management', 'sales forecasting',
                     'customer acquisition', 'client retention', 'sales strategy', 'enterprise sales',
                     'saas sales', 'inside sales', 'outside sales', 'negotiation skills', 'crm management'],
            'finance': ['financial modeling', 'financial analysis', 'budget management', 'cost analysis',
                       'risk management', 'portfolio management', 'investment analysis', 'forex trading',
                       'stock trading', 'financial planning', 'tax accounting', 'audit support', 'fpa'],
            'hr': ['talent acquisition', 'recruitment process', 'employee engagement', 'performance management',
                  'payroll management', 'benefits administration', 'compensation planning', 'learning development',
                  'organizational development', 'hr operations', 'compliance management', 'onboarding'],
            'data': ['data visualization', 'data engineering', 'data science', 'big data',
                    'data modeling', 'business intelligence', 'analytics', 'statistical analysis',
                    'predictive modeling', 'machine learning', 'deep learning', 'data pipeline',
                    'feature engineering', 'model training', 'model deployment', 'mlops',
                    'generative ai', 'large language model', 'retrieval augmented generation',
                    'time series analysis', 'recommendation systems', 'a/b testing'],
            'cloud': ['cloud architecture', 'cloud migration', 'aws certified', 'azure certified',
                     'gcp certified', 'infrastructure as code', 'serverless architecture',
                     'cloud security', 'disaster recovery', 'high availability', 'auto scaling',
                     'cloud native', 'multi cloud', 'hybrid cloud', 'cloud cost optimization',
                     'cloud networking', 'service mesh', 'cloud observability'],
            'content': ['content creation', 'content management', 'content strategy', 'copywriting',
                       'technical writing', 'blog writing', 'article writing', 'seo writing', 'editing'],
            'customer_service': ['customer support', 'customer service', 'technical support', 'customer success',
                                'help desk', 'ticket management', 'customer relations', 'customer satisfaction', 'support'],
            'development': ['web development', 'mobile development', 'full stack development', 'java developer',
                           'python developer', 'software development', 'application development', 'api development',
                           'full stack', 'backend development', 'frontend development', 'mobile app development',
                           'cross platform development', 'embedded software development'],
            'architecture': ['system design', 'software architecture', 'microservices architecture',
                            'event driven architecture', 'domain driven design', 'cqrs', 'event sourcing',
                            'api gateway', 'service oriented architecture', 'hexagonal architecture',
                            'clean architecture', 'distributed systems', 'high performance systems',
                            'low latency systems', 'scalable systems', 'fault tolerant systems'],
            'testing': ['test driven development', 'behaviour driven development', 'unit testing',
                       'integration testing', 'end to end testing', 'performance testing', 'load testing',
                       'security testing', 'smoke testing', 'regression testing', 'contract testing',
                       'mutation testing', 'test automation', 'qa engineering'],
            'security': ['devsecops', 'zero trust', 'penetration testing', 'vulnerability assessment',
                        'threat modeling', 'security audit', 'soc 2 compliance', 'iso 27001',
                        'gdpr compliance', 'hipaa compliance', 'pci dss', 'cloud security',
                        'application security', 'network security', 'identity management'],
            'it_ops': ['site reliability engineering', 'incident management', 'on call',
                      'runbook automation', 'chaos engineering', 'performance optimization',
                      'capacity planning', 'database administration', 'linux administration',
                      'network administration', 'it support', 'it operations'],
        }

        self.skill_synonyms = {
            'ml': 'machine learning', 'ai': 'artificial intelligence',
            'js': 'javascript', 'ts': 'typescript', 'py': 'python',
            'rb': 'ruby', 'rs': 'rust', 'kt': 'kotlin',
            'k8s': 'kubernetes', 'tf': 'terraform', 'iac': 'infrastructure as code',
            'gha': 'github actions', 'gh actions': 'github actions',
            'postgres': 'postgresql', 'mongo': 'mongodb', 'es': 'elasticsearch',
            'msql': 'mysql', 'mssql': 'sql server',
            'pytorch': 'pytorch', 'torch': 'pytorch', 'cv': 'computer vision',
            'nlp': 'natural language processing', 'llm': 'large language model',
            'genai': 'generative ai', 'rag': 'retrieval augmented generation',
            'xgb': 'xgboost', 'lgbm': 'lightgbm',
            'rest': 'rest api', 'api': 'rest api', 'gql': 'graphql',
            'reactjs': 'react', 'vuejs': 'vue', 'ng': 'angular',
            'nextjs': 'next.js', 'gatsbyjs': 'gatsby',
            'node': 'nodejs', 'node.js': 'nodejs', 'fastapi': 'fastapi',
            'flask': 'flask', 'django': 'django', 'springboot': 'spring boot',
            'sb': 'spring boot',
            'ci cd': 'ci/cd', 'cicd': 'ci/cd', 'cd': 'ci/cd',
            'dev ops': 'devops', 'gcp': 'gcp', 'aws': 'aws', 'azure': 'azure',
            'argocd': 'argocd', 'argo cd': 'argocd',
            'rn': 'react native',
            'tdd': 'test driven development', 'bdd': 'behaviour driven development',
            'e2e': 'end to end testing',
            'ddd': 'domain driven design', 'cqrs': 'cqrs', 'eda': 'event driven architecture',
            'fullstack': 'full stack development', 'full-stack': 'full stack development',
            'frontend': 'frontend development', 'front-end': 'frontend development',
            'backend': 'backend development', 'back-end': 'backend development',
            'sre': 'site reliability engineering',
        }

        self.proficiency_levels = {
            'expert': 1.0, 'advanced': 0.9, 'proficient': 0.8, 'skilled': 0.8,
            'experienced': 0.75, 'intermediate': 0.6, 'familiar': 0.4,
            'beginner': 0.2, 'novice': 0.2, 'basic': 0.3
        }

        self.all_single_skills = []
        for cat, skills in self.technical_skills.items():
            self.all_single_skills.extend(skills)
        self.all_multi_skills = []
        for cat, skills in self.multi_word_skills.items():
            self.all_multi_skills.extend(skills)
        self.all_skills = self.all_single_skills + self.all_multi_skills

        self.skill_categories = {}
        for cat, skills in {**self.technical_skills, **self.multi_word_skills}.items():
            for skill in skills:
                self.skill_categories[skill] = cat

    # ------------------------------------------------------------------
    # Parsing and extraction methods
    # ------------------------------------------------------------------
    def _parse_with_spacy(self, text: str) -> Dict:
        if not self.use_spacy or nlp is None:
            return self._fallback_section_extraction(text)

        doc = nlp(text)
        sections = {
            'header': '',
            'summary': '',
            'experience': [],
            'education': [],
            'skills': '',
            'certifications': [],
            'projects': [],
            'full_text': text
        }

        organizations = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        dates = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['DATE', 'TIME']]

        sentences = list(doc.sents)
        current_section = 'header'

        section_indicators = {
            'experience': ['experience', 'employment', 'work history', 'career', 'professional background'],
            'education': ['education', 'academic', 'qualifications', 'degree', 'university', 'college'],
            'skills': ['skills', 'technical skills', 'competencies', 'technologies', 'expertise', 'proficiencies'],
            'certifications': ['certifications', 'certificates', 'licenses', 'accreditations'],
            'projects': ['projects', 'portfolio', 'personal projects', 'open source'],
            'summary': ['summary', 'objective', 'profile', 'about', 'overview']
        }

        for sent in sentences:
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            is_header = False
            for section, keywords in section_indicators.items():
                if any(keyword in sent_lower for keyword in keywords) and len(sent_text) < 50:
                    current_section = section
                    is_header = True
                    break

            if not is_header:
                if current_section in ['experience', 'education', 'projects', 'certifications']:
                    sections[current_section].append(sent_text)
                else:
                    sections[current_section] += sent_text + '\n'

        for key in ['experience', 'education', 'projects', 'certifications']:
            if isinstance(sections[key], list):
                sections[key] = '\n'.join(sections[key])

        sections['parsed_jobs'] = self._extract_job_entries(sections['experience'], organizations, dates)
        sections['parsed_education'] = self._extract_education_entries(sections['education'])

        return sections

    def _fallback_section_extraction(self, text: str) -> Dict:
        sections = {
            'header': '',
            'summary': '',
            'experience': '',
            'education': '',
            'skills': '',
            'certifications': '',
            'projects': '',
            'full_text': text,
            'parsed_jobs': [],
            'parsed_education': []
        }

        lines = text.split('\n')
        current_section = 'header'

        section_keywords = {
            'experience': ['experience', 'work experience', 'employment', 'work history', 'professional experience'],
            'education': ['education', 'academic', 'qualifications', 'degree'],
            'skills': ['skills', 'technical skills', 'competencies', 'technologies', 'expertise'],
            'certifications': ['certifications', 'certificates', 'licenses', 'accreditations'],
            'projects': ['projects', 'portfolio', 'personal projects'],
            'summary': ['summary', 'objective', 'profile', 'about']
        }

        for line in lines:
            line_lower = line.strip().lower()
            if not line_lower:
                continue

            for section, keywords in section_keywords.items():
                if any(keyword in line_lower for keyword in keywords) and len(line_lower) < 40:
                    current_section = section
                    break

            sections[current_section] += line + '\n'

        sections['parsed_jobs'] = self._extract_job_entries_fallback(sections['experience'])
        sections['parsed_education'] = self._extract_education_entries_fallback(sections['education'])

        return sections

    def _extract_job_entries(self, experience_text: str, organizations: List[str], dates: List[Tuple]) -> List[JobEntry]:
        jobs = []
        if not experience_text.strip():
            return jobs

        job_blocks = re.split(r'\n(?=[A-Z][a-zA-Z\s]*(?:Developer|Engineer|Manager|Analyst|Designer|Lead|Director|Specialist|Consultant|Architect))', experience_text)

        for block in job_blocks:
            if not block.strip():
                continue

            lines = block.strip().split('\n')
            if not lines:
                continue

            title = lines[0].strip()
            title = re.sub(r'^(current|present|job|role|position):\s*', '', title, flags=re.IGNORECASE)

            company = 'Unknown'
            for org in organizations:
                if org.lower() in block.lower() and org.lower() not in title.lower():
                    company = org
                    break

            date_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s.,]*(\d{4})'
            date_matches = re.findall(date_pattern, block.lower())

            start_date = None
            end_date = None
            is_current = 'present' in block.lower() or 'current' in block.lower()

            if date_matches:
                try:
                    start_date = datetime.strptime(f"{date_matches[0][0]} {date_matches[0][1]}", "%b %Y")
                    if len(date_matches) > 1 and not is_current:
                        end_date = datetime.strptime(f"{date_matches[1][0]} {date_matches[1][1]}", "%b %Y")
                except:
                    pass

            years = 0
            if start_date:
                end = end_date if end_date else datetime.now()
                years = (end - start_date).days / 365.25

            company_tier = self._detect_company_tier(company, block)
            skills_in_job = self._extract_skills_from_text(block)

            jobs.append(JobEntry(
                title=title, company=company, company_tier=company_tier,
                start_date=start_date, end_date=end_date, is_current=is_current,
                description=block, skills_found=skills_in_job, years=years
            ))

        return jobs

    def _extract_job_entries_fallback(self, experience_text: str) -> List[JobEntry]:
        return self._extract_job_entries(experience_text, [], [])

    def _detect_company_tier(self, company: str, context: str) -> str:
        company_lower = company.lower()
        context_lower = context.lower()

        if any(f in company_lower for f in self.FAANG_COMPANIES):
            return 'faang'
        if any(f in company_lower for f in self.FORTUNE500_KEYWORDS):
            return 'fortune500'
        if any(s in context_lower for s in self.STARTUP_INDICATORS):
            return 'startup'
        return 'unknown'

    def _extract_education_entries(self, education_text: str) -> List[EducationEntry]:
        entries = []
        if not education_text.strip():
            return entries

        edu_blocks = re.split(r'\n(?=(?:Bachelor|Master|PhD|Doctorate|BS|BA|MS|MBA|B\.S\.|M\.S\.|B\.A\.|M\.A\.))', education_text)

        for block in edu_blocks:
            if not block.strip():
                continue

            degree_match = re.search(r'(bachelor|master|phd|doctorate|bs|ba|ms|mba|b\.s\.|m\.s\.|b\.a\.|m\.a\.)', block, re.IGNORECASE)
            degree = degree_match.group(1) if degree_match else 'Unknown'

            inst_match = re.search(r'(?:at|from|of|University|College|Institute|School)\s+([A-Z][a-zA-Z\s]+)', block)
            institution = inst_match.group(1).strip() if inst_match else 'Unknown'

            inst_lower = institution.lower()
            tier = 'top10' if any(u in inst_lower for u in ['mit', 'stanford', 'harvard', 'caltech', 'princeton']) else \
                   'top50' if any(u in inst_lower for u in self.TOP_UNIVERSITIES) else 'other'

            field_match = re.search(r'(?:in|of)\s+([A-Za-z\s]+?)(?:Engineering|Science|Arts|Technology|Management|Studies|Mathematics)', block, re.IGNORECASE)
            field = field_match.group(1).strip() if field_match else 'Unknown'

            level_map = {
                'phd': 5, 'doctorate': 5, 'doctoral': 5,
                'master': 4, 'mba': 4, 'ms': 4, 'ma': 4, 'm.s.': 4, 'm.a.': 4,
                'bachelor': 3, 'bs': 3, 'ba': 3, 'b.s.': 3, 'b.a.': 3,
                'associate': 2, 'diploma': 2,
                'certificate': 1
            }
            level = level_map.get(degree.lower(), 3)

            date_match = re.search(r'(19|20)\d{2}', block)
            grad_date = datetime(int(date_match.group(0)), 6, 1) if date_match else None

            entries.append(EducationEntry(
                degree=degree, field=field, institution=institution,
                tier=tier, graduation_date=grad_date, level=level
            ))

        return entries

    def _extract_education_entries_fallback(self, education_text: str) -> List[EducationEntry]:
        return self._extract_education_entries(education_text)

    def get_embedding(self, text: str) -> np.ndarray:
        if not self.use_semantic or sentence_model is None:
            return None

        cache_key = hash(text[:500])
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        embedding = sentence_model.encode(text, convert_to_numpy=True)
        self._embedding_cache[cache_key] = embedding
        return embedding

    def calculate_semantic_similarity(self, resume_text: str, job_description: str) -> float:
        if not self.use_semantic:
            return self.calculate_tfidf_similarity(resume_text, job_description)

        clean_resume = self._clean_text_for_embedding(resume_text)
        clean_job = self._clean_text_for_embedding(job_description)

        resume_emb = self.get_embedding(clean_resume)
        job_emb = self.get_embedding(clean_job)

        if resume_emb is None or job_emb is None:
            return self.calculate_tfidf_similarity(resume_text, job_description)

        similarity = np.dot(resume_emb, job_emb) / (np.linalg.norm(resume_emb) * np.linalg.norm(job_emb))
        return float(max(0, similarity))

    def calculate_tfidf_similarity(self, resume_text: str, job_description: str) -> float:
        clean_resume = re.sub(r'[\d]{3}[-.][\d]{3}[-.][\d]{4}', ' ', resume_text)
        clean_resume = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' ', clean_resume)
        clean_resume = re.sub(r'\s+', ' ', clean_resume)

        try:
            docs = [job_description, clean_resume]
            vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 3), max_features=5000, stop_words='english').fit_transform(docs)
            return float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
        except:
            return 0.0

    def _clean_text_for_embedding(self, text: str) -> str:
        text = re.sub(r'[\d]{3}[-.][\d]{3}[-.][\d]{4}', ' ', text)
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' ', text)
        text = re.sub(r'linkedin\.com/\S+', ' ', text)
        text = re.sub(r'github\.com/\S+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_skills_from_text(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = []

        for synonym, canonical in self.skill_synonyms.items():
            if re.search(r'\b' + re.escape(synonym) + r'\b', text_lower):
                found.append(canonical)

        for skill in self.all_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found.append(skill)

        return list(set(found))

    def extract_all_skills(self, resume_text: str) -> Dict:
        sections = self._parse_with_spacy(resume_text)
        parsed_jobs = sections.get('parsed_jobs', [])

        skills_section_skills = self._extract_skills_from_text(sections['skills'])

        skill_profiles = {}

        for job in parsed_jobs:
            recency = 1.0 if job.is_current else max(0.1, 1.0 - (2026 - (job.start_date.year if job.start_date else 2020)) * 0.15)

            for skill in job.skills_found:
                if skill not in skill_profiles:
                    skill_profiles[skill] = SkillMatch(
                        skill=skill,
                        category=self.skill_categories.get(skill, 'other'),
                        context_score=0.8,
                        recency_score=recency,
                        frequency=1,
                        years_of_use=job.years
                    )
                else:
                    skill_profiles[skill].frequency += 1
                    skill_profiles[skill].recency_score = max(skill_profiles[skill].recency_score, recency)
                    skill_profiles[skill].years_of_use += job.years

        for skill in skills_section_skills:
            if skill in skill_profiles:
                skill_profiles[skill].context_score = 1.0
            else:
                skill_profiles[skill] = SkillMatch(
                    skill=skill,
                    category=self.skill_categories.get(skill, 'other'),
                    context_score=1.0,
                    recency_score=0.5,
                    frequency=1,
                    years_of_use=0
                )

        for skill, profile in skill_profiles.items():
            text_lower = resume_text.lower()
            for prof_word, score in self.proficiency_levels.items():
                pattern = rf'{prof_word}\s+(?:in|with)?\s*{re.escape(skill)}|{re.escape(skill)}\s+(?:expert|advanced|proficient)'
                if re.search(pattern, text_lower):
                    profile.proficiency_indicators.append(prof_word)

        all_skills = list(skill_profiles.keys())
        skill_scores = {}
        for skill, profile in skill_profiles.items():
            prof_multiplier = 1.0
            if profile.proficiency_indicators:
                prof_multiplier = max(self.proficiency_levels.get(p, 0.5) for p in profile.proficiency_indicators)

            composite = profile.context_score * profile.recency_score * (1 + np.log1p(profile.frequency)) * prof_multiplier
            skill_scores[skill] = round(composite, 2)

        return {
            'all_skills': all_skills,
            'skill_profiles': {k: v.__dict__ for k, v in skill_profiles.items()},
            'skill_scores': skill_scores,
            'total_count': len(all_skills),
            'section_context': {
                'skills_section': skills_section_skills,
                'jobs': [{'title': j.title, 'company': j.company, 'skills': j.skills_found, 'years': j.years} for j in parsed_jobs]
            }
        }

    def extract_years_experience(self, resume_text: str) -> float:
        sections = self._parse_with_spacy(resume_text)
        parsed_jobs = sections.get('parsed_jobs', [])

        if parsed_jobs:
            total_years = sum(job.years for job in parsed_jobs)
            if 0 < total_years <= 50:
                return round(total_years, 1)

        return self._fallback_years_extraction(resume_text)

    def _fallback_years_extraction(self, text: str) -> float:
        patterns = [
            r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:experience|exp|work|professional)',
            r'(?:experience|exp)(?::|\s+of)?\s+(\d+)\s*\+?\s*years?',
            r'(?:over|more than)\s+(\d+)\s*\+?\s*years?',
            r'(\d+)-(\d+)\s*years?\s+(?:of\s+)?experience',
        ]

        years = []
        for pat in patterns:
            matches = re.findall(pat, text.lower())
            for m in matches:
                if isinstance(m, tuple):
                    vals = [int(x) for x in m if x.isdigit()]
                    if vals:
                        years.append(max(vals))
                else:
                    if m.isdigit():
                        val = int(m)
                        if 0 < val <= 50:
                            years.append(val)

        if not years:
            return 0.0

        years.sort()
        median = years[len(years) // 2]
        max_val = max(years)

        if max_val > median * 3 and max_val > 20:
            return float(median)
        return float(max_val)

    def extract_education(self, resume_text: str) -> Dict:
        sections = self._parse_with_spacy(resume_text)
        parsed_edu = sections.get('parsed_education', [])

        if parsed_edu:
            best_edu = max(parsed_edu, key=lambda e: e.level)
            return {
                'level': best_edu.level,
                'level_name': best_edu.degree,
                'field': best_edu.field,
                'institution': best_edu.institution,
                'tier': best_edu.tier,
                'entries': [e.__dict__ for e in parsed_edu]
            }

        return self._fallback_education_extraction(resume_text)

    def _fallback_education_extraction(self, text: str) -> Dict:
        text_lower = text.lower()

        education_levels = {
            'phd': 5, 'doctorate': 5, 'doctoral': 5,
            'master': 4, 'mba': 4, 'ms': 4, 'ma': 4, 'm.s.': 4, 'm.a.': 4,
            'bachelor': 3, 'bs': 3, 'ba': 3, 'b.s.': 3, 'b.a.': 3,
            'associate': 2, 'diploma': 2,
            'certificate': 1
        }

        level = 0
        level_name = 'Unknown'
        for edu, score in education_levels.items():
            if re.search(r'\b' + re.escape(edu) + r'\b', text_lower):
                if score > level:
                    level = score
                    level_name = edu.capitalize()

        return {
            'level': level,
            'level_name': level_name,
            'field': None,
            'institution': 'Unknown',
            'tier': 'unknown',
            'entries': []
        }

    def extract_certifications(self, resume_text: str) -> List[str]:
        sections = self._parse_with_spacy(resume_text)
        cert_text = sections['certifications'] if sections['certifications'] else resume_text

        cert_patterns = [
            r'(?:certified|certification|certificate)\s+(?:in|as)?\s*([A-Za-z\s\+#]+?)(?:\.|,|;|$)',
            r'([A-Za-z]+)\s+(?:Certified|Certification|Certificate)',
            r'(?:AWS|Azure|GCP|Google|Microsoft|Cisco|CompTIA|PMP|CFA|CPA|Scrum|ITIL)\s+[A-Za-z\s\+#]+'
        ]

        certs = []
        for pat in cert_patterns:
            matches = re.findall(pat, cert_text, re.IGNORECASE)
            for m in matches:
                cert = m.strip()
                if len(cert) > 2 and len(cert) < 50:
                    certs.append(cert)

        known_certs = ['aws', 'azure', 'gcp', 'pmp', 'scrum', 'cfa', 'cpa', 'cisco',
                      'comptia', 'itil', 'six sigma', 'ccna', 'ccnp', 'rhce']
        for cert in known_certs:
            if re.search(r'\b' + cert + r'\b', cert_text.lower()):
                certs.append(cert.upper())

        return list(set(certs))

    def extract_contact_info(self, text: str) -> Dict:
        info = {
            'name': None,
            'email': None,
            'phone': None,
            'linkedin': None,
            'github': None,
            'portfolio': None,
            'location': None
        }

        email_m = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
        if email_m:
            info['email'] = email_m.group(1)

        phone_patterns = [
            r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
        ]
        for pat in phone_patterns:
            phone_m = re.search(pat, text)
            if phone_m:
                info['phone'] = phone_m.group(0)
                break

        linkedin_m = re.search(r'linkedin\.com/in/([a-zA-Z0-9-]+)', text, re.IGNORECASE)
        if linkedin_m:
            info['linkedin'] = f"linkedin.com/in/{linkedin_m.group(1)}"

        github_m = re.search(r'github\.com/([a-zA-Z0-9-]+)', text, re.IGNORECASE)
        if github_m:
            info['github'] = f"github.com/{github_m.group(1)}"

        portfolio_m = re.search(r'(https?://)?([a-zA-Z0-9.-]+\.(?:com|io|dev|me))', text)
        if portfolio_m:
            info['portfolio'] = portfolio_m.group(2)

        lines = text.strip().split('\n')
        for line in lines[:10]:
            cleaned = line.strip()
            if cleaned and 2 <= len(cleaned.split()) <= 5 and len(cleaned) > 2:
                if re.search(r'[\d@.]|http|linkedin|github|www|portfolio', cleaned, re.IGNORECASE):
                    continue
                if re.match(r'^[A-Z\s]+$', cleaned) and len(cleaned) > 8:
                    continue
                if not re.search(r'[a-zA-Z]{2,}', cleaned):
                    continue
                info['name'] = cleaned
                break

        location_match = re.search(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)?),?\s*([A-Z]{2}|[A-Z][a-zA-Z]+)\b', text[:500])
        if location_match:
            info['location'] = f"{location_match.group(1)}, {location_match.group(2)}"

        return info

    def extract_current_job_title(self, text: str) -> Optional[str]:
        sections = self._parse_with_spacy(text)
        parsed_jobs = sections.get('parsed_jobs', [])

        current_jobs = [j for j in parsed_jobs if j.is_current]
        if current_jobs:
            return current_jobs[0].title

        if parsed_jobs:
            sorted_jobs = sorted([j for j in parsed_jobs if j.start_date],
                               key=lambda x: x.start_date or datetime.min, reverse=True)
            if sorted_jobs:
                return sorted_jobs[0].title

        return self._fallback_job_title_extraction(text)

    def _fallback_job_title_extraction(self, text: str) -> Optional[str]:
        lines = text.split('\n')
        job_keywords = ['developer', 'engineer', 'analyst', 'manager', 'specialist',
                       'consultant', 'designer', 'architect', 'administrator', 'director',
                       'lead', 'senior', 'junior', 'associate', 'principal', 'staff']

        for line in lines[:40]:
            line_clean = line.strip()
            if len(line_clean) < 5 or re.search(r'@|\d{3}[-.]\d{3}', line_clean):
                continue

            line_lower = line_clean.lower()
            line_clean = re.sub(r'^(current|job|role|position|title):\s*', '', line_clean, re.IGNORECASE)

            if any(kw in line_lower for kw in job_keywords):
                words = line_clean.split()
                if 1 <= len(words) <= 6:
                    if re.match(r'^[A-Z\s]+$', line_clean) and len(line_clean) > 8:
                        continue
                    if re.search(r'Inc\.?|LLC|Corp\.?|Ltd\.?|Company|Technologies', line_clean, re.IGNORECASE):
                        continue
                    return line_clean

        return None

    def extract_job_requirements(self, job_description: str) -> Dict:
        text = job_description.lower()

        critical_patterns = [
            r'(?:required|must have|essential|necessary|mandatory).*?(?:skills?|experience|knowledge|qualifications).*?[:;](.*?)(?:preferred|nice to have|bonus|optional|$)',
            r'(?:required|must have|essential).*?[:;](.*?)(?:preferred|nice to have|bonus|$)',
            r'(?:qualifications|requirements).*?[:;](.*?)(?:preferred|responsibilities|$)',
        ]

        critical_text = ''
        for pat in critical_patterns:
            match = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if match:
                critical_text = match.group(1)
                break

        required_skills = []
        for skill in self.all_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                required_skills.append(skill)

        critical_skills = []
        if critical_text:
            for skill in self.all_skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', critical_text):
                    critical_skills.append(skill)

        if not critical_skills and required_skills:
            critical_skills = required_skills[:min(3, len(required_skills))]

        years_m = re.search(r'(\d+)\s*\+?\s*years?', text)
        required_years = int(years_m.group(1)) if years_m else 0

        edu_req = None
        edu_patterns = [
            (r'(?:bachelor|bs|ba|b\.s\.|b\.a\.).*?(?:degree|required)', 'Bachelor'),
            (r'(?:master|ms|ma|m\.s\.|m\.a\.|mba).*?(?:degree|preferred|required)', 'Master'),
            (r'(?:phd|doctorate|doctoral).*?(?:degree|preferred|required)', 'PhD')
        ]
        for pat, level in edu_patterns:
            if re.search(pat, text, re.IGNORECASE):
                edu_req = level
                break

        # Predict job category from the job description for better weight alignment
        predicted_category = self._predict_job_category_from_text(job_description)

        return {
            'skills': list(set(required_skills)),
            'critical_skills': list(set(critical_skills)),
            'years': required_years,
            'education_required': edu_req,
            'predicted_category': predicted_category,
            'raw_text': text
        }

    # =================================================================
    # ENHANCED SCORING METHOD — IT-AWARE DYNAMIC WEIGHTING
    # =================================================================
    _IT_CORE_CATEGORIES = {
        'Software-Development', 'Frontend-Developer', 'Backend-Developer',
        'Full-Stack-Developer', 'Mobile-Application-Developer', 'Information-Technology',
        'Embedded-Systems-Engineer', 'QA-Automation-Engineer', 'Gaming-VFX',
    }
    _IT_DATA_CATEGORIES = {
        'Data-AI', 'Data-Scientist', 'Data-Engineer', 'Machine-Learning-Engineer',
        'Database-Administrator',
    }
    _IT_DEVOPS_CATEGORIES = {
        'DevOps-Infrastructure', 'DevOps-Engineer', 'Site-Reliability-Engineer',
        'Platform-Engineer', 'Cloud-Engineer', 'Systems-Administrator',
    }
    _IT_SEC_CATEGORIES = {
        'Cybersecurity', 'DevSecOps-Engineer', 'Cloud-Security-Engineer', 'Penetration-Tester',
        'Network-Security-Engineer',
    }
    _IT_MGMT_CATEGORIES = {
        'Tech-Management', 'Solutions-Architect', 'IT-Project-Manager', 'Product-Owner',
        'Chief-Technology-Officer', 'VP-of-Engineering', 'IT-Operations-Manager',
        'Networking-Telecom', 'Network-Engineer', 'Network-Architect',
    }

    def _get_score_weights(self, job_category: str) -> Dict[str, float]:
        """Return score component weights tailored to the predicted job category."""
        cat = job_category.replace('_', '-').strip()

        if cat in self._IT_CORE_CATEGORIES:
            return {'skill': 0.38, 'sim': 0.12, 'exp': 0.18,
                    'edu': 0.05, 'depth': 0.15, 'align': 0.12}
        if cat in self._IT_DATA_CATEGORIES:
            return {'skill': 0.36, 'sim': 0.12, 'exp': 0.16,
                    'edu': 0.08, 'depth': 0.18, 'align': 0.10}
        if cat in self._IT_DEVOPS_CATEGORIES:
            return {'skill': 0.36, 'sim': 0.12, 'exp': 0.20,
                    'edu': 0.04, 'depth': 0.16, 'align': 0.12}
        if cat in self._IT_SEC_CATEGORIES:
            return {'skill': 0.38, 'sim': 0.10, 'exp': 0.18,
                    'edu': 0.04, 'depth': 0.18, 'align': 0.12}
        if cat in self._IT_MGMT_CATEGORIES:
            return {'skill': 0.30, 'sim': 0.15, 'exp': 0.22,
                    'edu': 0.08, 'depth': 0.13, 'align': 0.12}
        # General / non-IT roles
        return {'skill': 0.30, 'sim': 0.15, 'exp': 0.20,
                'edu': 0.10, 'depth': 0.15, 'align': 0.10}

    def _calc_it_cert_bonus(self, resume_text: str) -> float:
        """Award bonus points for IT certifications found in the resume."""
        text_lower = resume_text.lower()
        best = 0.0
        for cert_phrase, pts in self.IT_CERT_BONUS_MAP.items():
            if cert_phrase in text_lower:
                best = max(best, pts)
        totals = sorted(
            [pts for phrase, pts in self.IT_CERT_BONUS_MAP.items() if phrase in text_lower],
            reverse=True
        )
        bonus = totals[0] if totals else 0.0
        if len(totals) > 1:
            bonus += totals[1] * 0.5
        return min(bonus, 10.0)

    def _calc_it_skill_tier_bonus(self, all_skills: List[str], skill_scores: Dict[str, float],
                                  job_category: str) -> float:
        it_cats = (self._IT_CORE_CATEGORIES | self._IT_DATA_CATEGORIES |
                   self._IT_DEVOPS_CATEGORIES | self._IT_SEC_CATEGORIES |
                   self._IT_MGMT_CATEGORIES)
        if job_category not in it_cats:
            return 0.0

        core_deep   = sum(1 for s in all_skills
                          if s in self.IT_CORE_SKILLS and skill_scores.get(s, 0) >= 1.2)
        second_deep = sum(1 for s in all_skills
                          if s in self.IT_SECONDARY_SKILLS and skill_scores.get(s, 0) >= 1.0)
        bonus_deep  = sum(1 for s in all_skills
                          if s in self.IT_BONUS_SKILLS and skill_scores.get(s, 0) >= 0.8)

        bonus = (core_deep * 1.5) + (second_deep * 0.8) + (bonus_deep * 0.4)
        return min(bonus, 12.0)

    def _calc_open_source_bonus(self, resume_text: str) -> float:
        text_lower = resume_text.lower()
        indicators = [
            'github.com/', 'open source', 'open-source', 'contributor', 'maintainer',
            'pull request', 'merged pr', 'hacktoberfest', 'npm package', 'pypi package',
            'published library', 'personal project', 'side project',
        ]
        count = sum(1 for ind in indicators if ind in text_lower)
        return min(count * 1.5, 6.0)

    def calculate_match_score(self, resume_text: str, job_description: str) -> Dict:
        req = self.extract_job_requirements(job_description)
        resume_skills_obj = self.extract_all_skills(resume_text)

        all_skills    = resume_skills_obj['all_skills']
        skill_scores  = resume_skills_obj['skill_scores']
        skill_profiles = resume_skills_obj['skill_profiles']

        years     = self.extract_years_experience(resume_text)
        education = self.extract_education(resume_text)
        sections  = self._parse_with_spacy(resume_text)
        parsed_jobs = sections.get('parsed_jobs', [])

        req_skills      = req['skills']
        critical_skills = req['critical_skills']
        job_category    = req.get('predicted_category', 'default')

        w = self._get_score_weights(job_category)

        skill_score      = self._calc_skill_match_score(
            all_skills, skill_scores, skill_profiles, req_skills, critical_skills)
        sim              = self.calculate_semantic_similarity(resume_text, job_description) * 100
        exp_score        = self._calc_experience_score(years, req['years'], skill_profiles)
        edu_score        = self._calc_education_score(education, req['education_required'])
        top_skills_score = self._calc_top_skills_score(skill_scores, skill_profiles, req_skills)
        category_alignment = self._calc_category_alignment(skill_profiles, job_category)

        company_bonus    = self._calc_company_bonus(parsed_jobs)
        critical_penalty = self._calc_critical_penalty(critical_skills, all_skills, skill_scores, req)
        diversity_bonus  = self._calc_diversity_bonus(skill_profiles, req_skills)
        cert_bonus       = self._calc_it_cert_bonus(resume_text)
        tier_bonus       = self._calc_it_skill_tier_bonus(all_skills, skill_scores, job_category)
        oss_bonus        = self._calc_open_source_bonus(resume_text)

        total = (skill_score      * w['skill'] +
                 sim              * w['sim']   +
                 exp_score        * w['exp']   +
                 edu_score        * w['edu']   +
                 top_skills_score * w['depth'] +
                 category_alignment * w['align'])

        total += company_bonus + diversity_bonus + cert_bonus + tier_bonus + oss_bonus
        total -= critical_penalty

        # Job recommendation boost (uses the expanded dictionary)
        job_recs = self.recommend_jobs(resume_text, years, all_skills,
                                       current_job_title=None, skill_scores=skill_scores)
        if job_recs:
            top_match = job_recs[0]['match_percentage']
            boost = top_match * 0.1
            total = min(total + boost, 100)

        total = max(0, min(100, total))

        return {
            'total':             round(total),
            'skill_score':       round(skill_score),
            'similarity_score':  round(sim),
            'experience_score':  round(exp_score),
            'education_score':   round(edu_score),
            'skill_depth_score': round(top_skills_score),
            'category_alignment': round(category_alignment),
            'company_tier_bonus': round(company_bonus),
            'diversity_bonus':    round(diversity_bonus),
            'cert_bonus':         round(cert_bonus),
            'it_tier_bonus':      round(tier_bonus),
            'oss_bonus':          round(oss_bonus),
            'critical_penalty':   round(critical_penalty),
            'score_weights_used': w,
            'job_category_detected': job_category,
            'matched_skills':    [s for s in all_skills if s in req_skills],
            'critical_matched':  [s for s in critical_skills if s in all_skills],
            'critical_missing':  [s for s in critical_skills if s not in all_skills],
            'missing_skills':    [s for s in req_skills if s not in all_skills],
            'top_skills':        self._get_top_skills(skill_scores, 5),
        }

    def _calc_skill_match_score(self, all_skills, skill_scores, skill_profiles, req_skills, critical_skills):
        if not req_skills:
            return 50.0

        matched          = [s for s in all_skills if s in req_skills]
        critical_matched = [s for s in matched if s in critical_skills]

        def _skill_weight(skill):
            if skill in self.IT_CORE_SKILLS:
                return 2.0
            if skill in self.IT_SECONDARY_SKILLS:
                return 1.5
            return 1.0

        critical_base_weight = 3.0

        total_weighted_req = sum(
            critical_base_weight * _skill_weight(s) for s in critical_skills
        ) + sum(
            _skill_weight(s) for s in req_skills if s not in critical_skills
        )

        total_weighted_match = 0.0
        for s in critical_matched:
            quality = skill_scores.get(s, 0.5)
            if quality >= 1.5:
                quality *= 1.3
            total_weighted_match += critical_base_weight * _skill_weight(s) * quality

        for s in [m for m in matched if m not in critical_matched]:
            quality = skill_scores.get(s, 0.5)
            total_weighted_match += _skill_weight(s) * quality

        base_score = (total_weighted_match / total_weighted_req) * 100 if total_weighted_req > 0 else 50

        if len(matched) > len(req_skills):
            bonus = min((len(matched) - len(req_skills)) * 3, 15)
            base_score = min(base_score + bonus, 100)

        return min(base_score, 100)

    def _calc_experience_score(self, years, req_years, skill_profiles):
        if req_years > 0:
            ratio = years / req_years
            if ratio >= 1.0:
                base_score = 100
            elif ratio >= 0.5:
                base_score = 70 + (ratio - 0.5) * 60
            else:
                base_score = max(ratio * 140, 0)
            base_score = min(base_score, 100)
        else:
            base_score = 50
        
        deep_skills = sum(1 for p in skill_profiles.values() if p.get('years_of_use', 0) >= 2)
        depth_bonus = min(deep_skills * 2, 10)
        
        return min(base_score + depth_bonus, 100)

    def _calc_education_score(self, education, edu_required):
        if not edu_required:
            return 100
        
        edu_map = {'Certificate': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
        req_level = edu_map.get(edu_required, 3)
        actual_level = education['level']
        
        if actual_level >= req_level:
            score = 100
        elif actual_level == req_level - 1:
            score = 70
        else:
            score = max(30, actual_level * 20)
        
        field = (education.get('field') or '').lower()
        if any(kw in field for kw in ['computer', 'software', 'engineering', 'data', 'science']):
            score = min(score + 10, 100)
        
        return score

    def _calc_top_skills_score(self, skill_scores, skill_profiles, req_skills):
        if not skill_scores:
            return 0
        
        sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        top_score = 0
        for skill, score in sorted_skills:
            req_bonus = 1.5 if skill in req_skills else 1.0
            top_score += score * req_bonus
        
        max_possible = 5 * 3.0 * 1.5
        normalized = (top_score / max_possible) * 100
        
        return min(normalized, 100)

    def _calc_category_alignment(self, skill_profiles, job_category):
        if not skill_profiles:
            return 50

        job_category = job_category.replace('_', '-').replace(' ', '-').strip()

        category_counts = defaultdict(int)
        category_scores = defaultdict(float)

        for skill, profile in skill_profiles.items():
            cat = profile.get('category', 'other')
            category_counts[cat] += 1
            category_scores[cat] += profile.get('context_score', 0.5) * profile.get('recency_score', 0.5)

        weights = self.CATEGORY_SKILL_WEIGHTS.get(job_category)
        if weights is None:
            job_category_lower = job_category.lower()
            weights = next(
                (v for k, v in self.CATEGORY_SKILL_WEIGHTS.items() if k.lower().replace(' ', '-') == job_category_lower),
                None
            )
        if weights is None:
            weights = self.CATEGORY_SKILL_WEIGHTS['default']

        alignment = 0
        total_weight = 0
        for cat, weight in weights.items():
            total_weight += weight
            if cat in category_scores:
                normalized_score = category_scores[cat] / max(category_counts[cat], 1)
                alignment += normalized_score * weight

        if total_weight > 0:
            alignment = (alignment / total_weight) * 100

        return min(alignment, 100)

    def _calc_company_bonus(self, parsed_jobs):
        tier_scores = {'faang': 15, 'fortune500': 10, 'startup': 5, 'unknown': 0}
        for job in parsed_jobs:
            if job.is_current or (not any(j.is_current for j in parsed_jobs) and job == parsed_jobs[0]):
                return tier_scores.get(job.company_tier, 0)
        return 0

    def _calc_critical_penalty(self, critical_skills, all_skills, skill_scores, req):
        missing = [s for s in critical_skills if s not in all_skills]
        if not missing:
            return 0
        
        base_penalty = len(missing) * 12
        
        matched_ratio = len([s for s in req['skills'] if s in all_skills]) / max(len(req['skills']), 1)
        if matched_ratio > 0.7:
            base_penalty *= 0.7
        
        if skill_scores:
            top_score = max(skill_scores.values())
            if top_score >= 2.0:
                base_penalty *= 0.8
        
        return base_penalty

    def _calc_diversity_bonus(self, skill_profiles, req_skills):
        if not skill_profiles or not req_skills:
            return 0
        
        req_categories = set()
        for skill in req_skills:
            cat = self.skill_categories.get(skill, 'other')
            req_categories.add(cat)
        
        matched_categories = set()
        for skill, profile in skill_profiles.items():
            if skill in req_skills:
                matched_categories.add(profile.get('category', 'other'))
        
        coverage = len(matched_categories) / max(len(req_categories), 1)
        return coverage * 10

    def _get_top_skills(self, skill_scores, n=5):
        sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [{'skill': s, 'score': round(score, 2)} for s, score in sorted_skills]

    # ----------------------------------------------------------------
    # RECOMMEND JOBS (EXPANDED DICTIONARY)
    # ----------------------------------------------------------------
    def recommend_jobs(self, resume_text: str, years_experience: float, all_skills: List[str],
                       current_job_title: Optional[str] = None, skill_scores: Optional[Dict] = None) -> List[Dict]:
        job_recommendations = {
            'Software Engineer': {
                'primary_skills': ['python', 'java', 'javascript', 'c++', 'go', 'rust', 'software development', 'coding', 'programming'],
                'secondary_skills': ['git', 'agile', 'rest api', 'microservices', 'design patterns', 'unit testing', 'tdd'],
                'multi_skills': ['web development', 'software development', 'application development'],
                'critical_skills': ['python', 'java', 'javascript'],
                'min_experience': 1,
                'description': 'Develop and maintain software applications'
            },
            'Frontend Developer': {
                'primary_skills': ['javascript', 'react', 'vue', 'angular', 'html', 'css', 'typescript', 'frontend development'],
                'secondary_skills': ['tailwind', 'bootstrap', 'webpack', 'responsive design', 'ui/ux'],
                'multi_skills': ['web development', 'ui design', 'frontend engineering'],
                'critical_skills': ['javascript', 'html', 'css'],
                'min_experience': 1,
                'description': 'Build user-facing web applications'
            },
            'Backend Developer': {
                'primary_skills': ['python', 'java', 'sql', 'rest api', 'node.js', 'django', 'spring boot', 'fastapi', 'backend development'],
                'secondary_skills': ['microservices', 'database', 'api design', 'cloud', 'docker'],
                'multi_skills': ['web development', 'api development', 'server-side development'],
                'critical_skills': ['python', 'java', 'sql'],
                'min_experience': 1,
                'description': 'Build server-side applications and APIs'
            },
            'Full Stack Developer': {
                'primary_skills': ['javascript', 'react', 'node.js', 'python', 'html', 'css', 'sql', 'full stack development'],
                'secondary_skills': ['git', 'rest api', 'database', 'docker', 'aws'],
                'multi_skills': ['web development', 'frontend development', 'backend development'],
                'critical_skills': ['javascript', 'react', 'node.js'],
                'min_experience': 2,
                'description': 'Develop end-to-end web applications'
            },
            'Mobile Developer': {
                'primary_skills': ['swift', 'kotlin', 'react native', 'flutter', 'ios', 'android', 'mobile development'],
                'secondary_skills': ['rest api', 'firebase', 'git', 'ui/ux'],
                'multi_skills': ['mobile app development', 'cross-platform development'],
                'critical_skills': ['swift', 'kotlin', 'react native'],
                'min_experience': 1,
                'description': 'Build mobile applications for iOS and Android'
            },
            'Data Scientist': {
                'primary_skills': ['python', 'machine learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'statistics', 'data science'],
                'secondary_skills': ['sql', 'data analysis', 'deep learning', 'nlp', 'scikit-learn'],
                'multi_skills': ['data science', 'machine learning', 'predictive modeling'],
                'critical_skills': ['python', 'machine learning'],
                'min_experience': 2,
                'description': 'Develop ML models and analyze complex datasets'
            },
            'Data Engineer': {
                'primary_skills': ['python', 'sql', 'spark', 'hadoop', 'airflow', 'etl', 'data engineering'],
                'secondary_skills': ['aws', 'azure', 'gcp', 'data warehousing', 'big data'],
                'multi_skills': ['data engineering', 'data pipeline', 'data infrastructure'],
                'critical_skills': ['python', 'sql'],
                'min_experience': 2,
                'description': 'Build and maintain data pipelines and infrastructure'
            },
            'Data Analyst': {
                'primary_skills': ['sql', 'excel', 'tableau', 'power bi', 'data analysis', 'statistics', 'python'],
                'secondary_skills': ['pandas', 'data visualization', 'business intelligence'],
                'multi_skills': ['data analysis', 'business analytics', 'reporting'],
                'critical_skills': ['sql', 'data analysis'],
                'min_experience': 1,
                'description': 'Analyze data and create visualizations to support decisions'
            },
            'Machine Learning Engineer': {
                'primary_skills': ['python', 'machine learning', 'tensorflow', 'pytorch', 'mlops', 'model deployment'],
                'secondary_skills': ['docker', 'kubernetes', 'aws', 'ci/cd', 'scikit-learn'],
                'multi_skills': ['machine learning', 'ml engineering', 'model lifecycle'],
                'critical_skills': ['python', 'machine learning'],
                'min_experience': 2,
                'description': 'Deploy and maintain machine learning models in production'
            },
            'DevOps Engineer': {
                'primary_skills': ['docker', 'kubernetes', 'aws', 'terraform', 'ansible', 'jenkins', 'ci/cd', 'devops'],
                'secondary_skills': ['linux', 'python', 'shell', 'git', 'monitoring'],
                'multi_skills': ['cloud architecture', 'infrastructure as code', 'devops practices'],
                'critical_skills': ['docker', 'kubernetes'],
                'min_experience': 2,
                'description': 'Manage infrastructure and deployment pipelines'
            },
            'Cloud Engineer': {
                'primary_skills': ['aws', 'azure', 'gcp', 'cloud architecture', 'terraform', 'cloudformation'],
                'secondary_skills': ['docker', 'kubernetes', 'networking', 'security', 'linux'],
                'multi_skills': ['cloud engineering', 'cloud migration', 'infrastructure as code'],
                'critical_skills': ['aws', 'azure', 'gcp'],
                'min_experience': 2,
                'description': 'Design and manage cloud infrastructure'
            },
            'Site Reliability Engineer': {
                'primary_skills': ['linux', 'python', 'go', 'monitoring', 'prometheus', 'grafana', 'sre', 'incident response'],
                'secondary_skills': ['kubernetes', 'aws', 'terraform', 'ci/cd'],
                'multi_skills': ['site reliability', 'system engineering', 'observability'],
                'critical_skills': ['linux', 'monitoring'],
                'min_experience': 3,
                'description': 'Ensure reliability and performance of production systems'
            },
            'System Administrator': {
                'primary_skills': ['linux', 'windows server', 'networking', 'virtualization', 'backup', 'security', 'system administration'],
                'secondary_skills': ['bash', 'powershell', 'active directory', 'vmware'],
                'multi_skills': ['system administration', 'it operations', 'server management'],
                'critical_skills': ['linux', 'networking'],
                'min_experience': 2,
                'description': 'Manage servers, networks, and IT infrastructure'
            },
            'Cybersecurity Analyst': {
                'primary_skills': ['security', 'vulnerability assessment', 'siem', 'incident response', 'firewall', 'ids/ips'],
                'secondary_skills': ['python', 'linux', 'risk assessment', 'compliance', 'nist'],
                'multi_skills': ['cybersecurity', 'information security', 'threat analysis'],
                'critical_skills': ['security', 'vulnerability assessment'],
                'min_experience': 2,
                'description': 'Protect systems from cyber threats and respond to incidents'
            },
            'Penetration Tester': {
                'primary_skills': ['penetration testing', 'ethical hacking', 'metasploit', 'burp suite', 'kali linux', 'vulnerability assessment'],
                'secondary_skills': ['python', 'linux', 'web security', 'network security'],
                'multi_skills': ['penetration testing', 'offensive security', 'ethical hacking'],
                'critical_skills': ['penetration testing', 'ethical hacking'],
                'min_experience': 2,
                'description': 'Simulate attacks to identify security weaknesses'
            },
            'Product Manager': {
                'primary_skills': ['product strategy', 'market research', 'user stories', 'roadmap', 'agile', 'scrum', 'product management'],
                'secondary_skills': ['data analysis', 'stakeholder management', 'competitive analysis', 'customer development'],
                'multi_skills': ['product management', 'product development', 'agile methodologies'],
                'critical_skills': ['product strategy', 'market research'],
                'min_experience': 3,
                'description': 'Define product strategy and roadmap'
            },
            'Project Manager': {
                'primary_skills': ['project management', 'agile', 'scrum', 'jira', 'risk management', 'budgeting', 'stakeholder communication'],
                'secondary_skills': ['leadership', 'team coordination', 'waterfall', 'microsoft project'],
                'multi_skills': ['project management', 'agile delivery', 'team leadership'],
                'critical_skills': ['project management', 'agile'],
                'min_experience': 3,
                'description': 'Lead projects to successful completion on time and budget'
            },
            'Scrum Master': {
                'primary_skills': ['scrum', 'agile', 'jira', 'team facilitation', 'sprint planning', 'retrospectives'],
                'secondary_skills': ['conflict resolution', 'coaching', 'kanban', 'lean'],
                'multi_skills': ['scrum master', 'agile coaching', 'team empowerment'],
                'critical_skills': ['scrum', 'agile'],
                'min_experience': 2,
                'description': 'Facilitate agile processes and remove impediments'
            },
            'UI/UX Designer': {
                'primary_skills': ['ux design', 'ui design', 'figma', 'sketch', 'wireframing', 'prototyping', 'user research'],
                'secondary_skills': ['adobe xd', 'photoshop', 'illustrator', 'responsive design', 'design thinking'],
                'multi_skills': ['user experience', 'user interface', 'interaction design'],
                'critical_skills': ['ux design', 'figma'],
                'min_experience': 2,
                'description': 'Design user experiences and interfaces'
            },
            'QA Engineer': {
                'primary_skills': ['test automation', 'selenium', 'manual testing', 'bug tracking', 'test cases', 'regression testing'],
                'secondary_skills': ['python', 'java', 'ci/cd', 'junit', 'testng'],
                'multi_skills': ['quality assurance', 'automated testing', 'software testing'],
                'critical_skills': ['test automation', 'selenium'],
                'min_experience': 1,
                'description': 'Ensure software quality through testing and automation'
            },
            'Business Analyst': {
                'primary_skills': ['business analysis', 'requirement analysis', 'data analysis', 'stakeholder management', 'process improvement'],
                'secondary_skills': ['sql', 'excel', 'tableau', 'agile', 'user stories'],
                'multi_skills': ['business analysis', 'requirements engineering', 'process modeling'],
                'critical_skills': ['business analysis', 'requirement analysis'],
                'min_experience': 2,
                'description': 'Bridge business needs with technical solutions'
            },
            'Sales Manager': {
                'primary_skills': ['sales management', 'crm', 'negotiation', 'pipeline management', 'forecasting', 'customer relations'],
                'secondary_skills': ['business development', 'lead generation', 'account management', 'sales strategy'],
                'multi_skills': ['sales management', 'sales operations', 'territory management'],
                'critical_skills': ['sales management', 'crm'],
                'min_experience': 3,
                'description': 'Lead sales team and manage revenue targets'
            },
            'Marketing Manager': {
                'primary_skills': ['digital marketing', 'seo', 'content marketing', 'social media', 'google analytics', 'brand management', 'marketing strategy'],
                'secondary_skills': ['email marketing', 'ppc', 'marketing automation', 'campaign management'],
                'multi_skills': ['marketing strategy', 'digital marketing', 'brand management'],
                'critical_skills': ['digital marketing', 'seo'],
                'min_experience': 3,
                'description': 'Develop and execute marketing campaigns'
            },
            'HR Manager': {
                'primary_skills': ['recruitment', 'onboarding', 'performance management', 'employee relations', 'hr policies', 'payroll'],
                'secondary_skills': ['hris', 'compensation', 'benefits', 'training', 'compliance'],
                'multi_skills': ['human resources', 'talent acquisition', 'employee engagement'],
                'critical_skills': ['recruitment', 'performance management'],
                'min_experience': 3,
                'description': 'Manage human resources functions and employee lifecycle'
            },
            'Financial Analyst': {
                'primary_skills': ['financial modeling', 'excel', 'data analysis', 'budgeting', 'forecasting', 'accounting'],
                'secondary_skills': ['sql', 'tableau', 'risk management', 'investment analysis'],
                'multi_skills': ['financial analysis', 'financial modeling', 'business finance'],
                'critical_skills': ['financial modeling', 'excel'],
                'min_experience': 2,
                'description': 'Analyze financial data and create forecasts'
            },
            'Accountant': {
                'primary_skills': ['accounting', 'bookkeeping', 'tax', 'audit', 'financial reporting', 'quickbooks', 'excel'],
                'secondary_skills': ['gaap', 'ifrs', 'payroll', 'budgeting', 'financial statements'],
                'multi_skills': ['accounting', 'financial reporting', 'tax preparation'],
                'critical_skills': ['accounting', 'tax'],
                'min_experience': 2,
                'description': 'Manage financial records and ensure compliance'
            }
        }

        recommendations = []
        for job, data in job_recommendations.items():
            if years_experience < data['min_experience']:
                continue

            primary_skills = data['primary_skills']
            secondary_skills = data['secondary_skills']
            critical_skills = data.get('critical_skills', primary_skills[:2])
            multi_skills = data.get('multi_skills', [])

            primary_match = [s for s in primary_skills if s in all_skills]
            secondary_match = [s for s in secondary_skills if s in all_skills]
            critical_match = [s for s in critical_skills if s in all_skills]
            multi_match = [s for s in multi_skills if s in all_skills]

            critical_weight = 3.0
            primary_weight = 2.0
            secondary_weight = 1.0
            multi_weight = 1.5

            total_possible = (len(critical_skills) * critical_weight +
                            len(primary_skills) * primary_weight +
                            len(secondary_skills) * secondary_weight +
                            len(multi_skills) * multi_weight)

            total_achieved = (len(critical_match) * critical_weight +
                            len(primary_match) * primary_weight +
                            len(secondary_match) * secondary_weight +
                            len(multi_match) * multi_weight)

            weighted = (total_achieved / total_possible * 100) if total_possible > 0 else 0

            if current_job_title:
                job_lower = job.lower()
                current_lower = current_job_title.lower()
                if job_lower in current_lower or current_lower in job_lower:
                    weighted += 15

            if skill_scores:
                deep_matches = sum(1 for s in primary_match if skill_scores.get(s, 0) >= 1.5)
                weighted += deep_matches * 3

            if weighted >= 15:
                recommendations.append({
                    'job_title': job,
                    'match_percentage': round(min(weighted, 100)),
                    'description': data['description'],
                    'primary_skills_found': primary_match,
                    'critical_skills_found': critical_match,
                    'missing_skills': [s for s in primary_skills if s not in all_skills],
                    'critical_missing': [s for s in critical_skills if s not in all_skills],
                    'experience_requirement': data['min_experience'],
                    'experience_met': years_experience >= data['min_experience']
                })

        recommendations.sort(key=lambda x: x['match_percentage'], reverse=True)
        return recommendations[:5]

    # ----------------------------------------------------------------
    # SALARY, HIRING MODELS, AND OTHER METHODS
    # ----------------------------------------------------------------
    def _load_salary_model(self):
        try:
            model_path = os.path.join(self.model_dir, 'salary_predictor.pkl')
            encoders_path = os.path.join(self.model_dir, 'salary_encoders.pkl')
            order_path = os.path.join(self.model_dir, 'salary_feature_order.pkl')
            if os.path.exists(model_path) and os.path.exists(encoders_path) and os.path.exists(order_path):
                self.salary_model = joblib.load(model_path)
                self.salary_encoders = joblib.load(encoders_path)
                self.salary_feature_order = joblib.load(order_path)
                print("✅ Salary prediction model loaded")
            else:
                print("⚠️ Salary model files not found. Using fallback.")
        except Exception as e:
            print(f"⚠️ Error loading salary model: {e}. Using fallback.")

    def predict_salary(self, job_title: str, years_experience: float, skills_count: int,
                       education_level: str = 'Bachelor', industry: str = 'Technology',
                       company_size: str = 'Medium', location: str = 'United States',
                       remote_work: str = 'No', certifications: int = 0,
                       company_tier: str = 'unknown') -> Dict:
        base_salary = None
        if self.salary_model is not None and self.salary_encoders is not None and self.salary_feature_order is not None:
            try:
                data = {
                    'job_title': job_title,
                    'experience_years': years_experience,
                    'skills_count': skills_count,
                    'education_level': education_level,
                    'industry': industry,
                    'company_size': company_size,
                    'location': location,
                    'remote_work': remote_work,
                    'certifications': certifications
                }
                for col in ['job_title', 'education_level', 'industry', 'company_size', 'location', 'remote_work']:
                    if col in self.salary_encoders:
                        encoder = self.salary_encoders[col]
                        if isinstance(encoder, dict) and 'encoder' in encoder:
                            encoder = encoder['encoder']
                        try:
                            data[col] = encoder.transform([data[col]])[0]
                        except (ValueError, AttributeError):
                            data[col] = 0
                    else:
                        data[col] = 0
                features_df = pd.DataFrame([data], columns=self.salary_feature_order)
                predicted = self.salary_model.predict(features_df)[0]
                base_salary = predicted
            except Exception as e:
                print(f"Salary model inference failed: {e}")
        if base_salary is None:
            fallback_map = {
                'Software Engineer': 100000, 'Data Engineer': 105000, 'Data Scientist': 102000,
                'DevOps Engineer': 108000, 'Product Manager': 115000, 'UX Designer': 95000,
                'Frontend Developer': 98000, 'Backend Developer': 102000, 'Professional': 60000
            }
            base = fallback_map.get(job_title, 60000)
            exp_mult = 1 + min(0.05 * years_experience, 0.5)
            skill_bonus = min(skills_count * 1000, 20000)
            base_salary = base * exp_mult + skill_bonus
        tier_multipliers = {'faang': 1.25, 'fortune500': 1.15, 'startup': 0.95, 'unknown': 1.0}
        adjusted = base_salary * tier_multipliers.get(company_tier, 1.0)
        return {
            'estimated_salary': int(round(adjusted / 1000) * 1000),
            'range_low': int(round(adjusted * 0.85 / 1000) * 1000),
            'range_high': int(round(adjusted * 1.15 / 1000) * 1000),
            'base_salary': int(round(base_salary / 1000) * 1000),
            'company_tier_adjustment': tier_multipliers.get(company_tier, 1.0),
            'confidence': 'high' if self.salary_model else 'low'
        }

    def _load_hiring_model(self):
        try:
            model_path = os.path.join(self.model_dir, 'hiring_classifier.pkl')
            scaler_path = os.path.join(self.model_dir, 'hiring_scaler.pkl')
            if os.path.exists(model_path):
                self.hiring_classifier = joblib.load(model_path)
                if os.path.exists(scaler_path):
                    self.hiring_scaler = joblib.load(scaler_path)
                print("✅ Hiring prediction model loaded")
        except Exception as e:
            print(f"⚠️ Hiring model not loaded: {e}")

    def predict_hire_probability(self, resume_text: str, job_description: str) -> Dict:
        if self.hiring_classifier is None:
            return {'error': 'Hiring model not trained', 'probability': None}
        scores = self.calculate_match_score(resume_text, job_description)
        years = self.extract_years_experience(resume_text)
        skills_obj = self.extract_all_skills(resume_text)
        education = self.extract_education(resume_text)
        sections = self._parse_with_spacy(resume_text)
        parsed_jobs = sections.get('parsed_jobs', [])
        tier_scores = {'faang': 3, 'fortune500': 2, 'startup': 1, 'unknown': 0}
        max_tier = max([tier_scores.get(j.company_tier, 0) for j in parsed_jobs], default=0)
        features = [
            scores['total'], scores['skill_score'], scores['experience_score'],
            scores['education_score'], scores['skill_depth_score'], years,
            len(skills_obj['all_skills']), max_tier, 1 if not scores['critical_missing'] else 0,
            education['level']
        ]
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.hiring_scaler.transform(features_array)
        probability = self.hiring_classifier.predict_proba(features_scaled)[0][1]
        prediction = self.hiring_classifier.predict(features_scaled)[0]
        return {
            'hire_probability': round(float(probability), 3),
            'prediction': 'Hire' if prediction == 1 else 'No Hire',
            'confidence': 'high' if probability > 0.8 or probability < 0.2 else 'medium',
            'key_factors': {
                'match_score': scores['total'],
                'critical_skills_met': not bool(scores['critical_missing']),
                'experience_adequate': scores['experience_score'] >= 70,
                'company_tier': max_tier
            }
        }

    def load_trained_models(self):
        try:
            tfidf_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
            cls_path = os.path.join(self.model_dir, 'category_classifier.pkl')
            if os.path.exists(tfidf_path) and os.path.exists(cls_path):
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                self.category_classifier = joblib.load(cls_path)
                return True
            else:
                self.use_trained_model = False
                return False
        except:
            self.use_trained_model = False
            return False

    def predict_job_category(self, resume_text: str) -> Dict:
        if self.use_trained_model and self.tfidf_vectorizer is not None:
            try:
                vec = self.tfidf_vectorizer.transform([resume_text])
                probs = self.category_classifier.predict_proba(vec)[0]
                top_idx = np.argmax(probs)
                cat = self.category_classifier.classes_[top_idx]
                confidence = float(probs[top_idx])
                top3 = [(self.category_classifier.classes_[i], float(probs[i]))
                       for i in np.argsort(probs)[::-1][:3]]
                return {
                    'predicted_category': cat,
                    'confidence': confidence,
                    'top_categories': top3,
                    'method': 'trained_classifier'
                }
            except:
                pass
        return self._predict_category_by_skills(resume_text)

    def _predict_category_by_skills(self, resume_text: str) -> Dict:
        text = resume_text.lower()
        mapping = {
            'Information-Technology': ['python', 'java', 'javascript', 'react', 'django', 'aws', 'kubernetes', 'docker', 'software', 'developer'],
            'Engineering': ['engineering', 'mechanical', 'electrical', 'civil', 'automotive', 'cad', 'solidworks'],
            'Finance': ['financial', 'accounting', 'auditing', 'tax', 'investment', 'banking', 'excel', 'financial modeling'],
            'Sales': ['sales', 'business development', 'account management', 'pipeline', 'crm', 'negotiation'],
            'HR': ['human resources', 'recruitment', 'hr', 'payroll', 'training', 'employee relations'],
            'Healthcare': ['medical', 'nurse', 'doctor', 'healthcare', 'clinical', 'patient care'],
            'Design': ['design', 'ui', 'ux', 'figma', 'adobe', 'graphic design', 'photoshop'],
            'Business-Development': ['business development', 'partnership', 'strategy', 'market analysis'],
            'Marketing': ['marketing', 'digital marketing', 'seo', 'social media', 'brand', 'campaign'],
        }
        scores = {cat: sum(1 for s in skills if s in text) for cat, skills in mapping.items()}
        best = max(scores, key=scores.get)
        conf = min(scores[best] / len(mapping[best]), 1.0) if scores[best] > 0 else 0.3
        return {'predicted_category': best, 'confidence': conf, 'method': 'skill_based'}

    def _predict_job_category_from_text(self, text: str) -> str:
        text_lower = text.lower()
        mapping = {
            'Software-Development': ['software developer', 'software engineer', 'full stack', 'fullstack', 'web developer', 'application developer', 'software development', 'coding', 'programming'],
            'Frontend-Developer': ['frontend', 'front-end', 'front end', 'ui developer', 'react developer', 'angular developer', 'vue developer'],
            'Backend-Developer': ['backend', 'back-end', 'back end', 'server-side', 'api developer', 'django', 'spring boot', 'node.js developer'],
            'Mobile-Application-Developer': ['mobile developer', 'ios developer', 'android developer', 'react native', 'flutter developer', 'mobile app'],
            'Embedded-Systems-Engineer': ['embedded', 'firmware', 'iot developer', 'hardware software', 'microcontroller', 'rtos'],
            'QA-Automation-Engineer': ['qa engineer', 'automation engineer', 'test engineer', 'selenium', 'cypress', 'quality assurance'],
            'Data-AI': ['data scientist', 'machine learning', 'ml engineer', 'ai engineer', 'deep learning', 'nlp engineer', 'data analyst', 'analytics'],
            'Data-Engineer': ['data engineer', 'etl developer', 'pipeline engineer', 'spark', 'hadoop', 'data infrastructure'],
            'Data-Scientist': ['data scientist', 'statistician', 'predictive modeling', 'research scientist', 'quantitative analyst'],
            'Machine-Learning-Engineer': ['machine learning engineer', 'mlops', 'model deployment', 'tensorflow', 'pytorch production'],
            'Database-Administrator': ['database administrator', 'dba', 'database engineer', 'sql server', 'oracle dba', 'postgres dba'],
            'DevOps-Infrastructure': ['devops', 'sre', 'site reliability', 'platform engineer', 'infrastructure engineer', 'cloud engineer', 'ci/cd'],
            'DevOps-Engineer': ['devops engineer', 'devops specialist', 'release engineer', 'build engineer', 'deployment engineer'],
            'Site-Reliability-Engineer': ['site reliability engineer', 'sre', 'reliability engineer', 'production engineer'],
            'Platform-Engineer': ['platform engineer', 'developer platform', 'internal tools engineer', 'platform team'],
            'Cloud-Engineer': ['cloud engineer', 'aws engineer', 'azure engineer', 'gcp engineer', 'cloud architect', 'cloud specialist'],
            'Systems-Administrator': ['systems administrator', 'sysadmin', 'linux administrator', 'windows administrator', 'server administrator'],
            'Cybersecurity': ['cybersecurity', 'security analyst', 'information security', 'security engineer', 'infosec', 'security architect'],
            'DevSecOps-Engineer': ['devsecops', 'security devops', 'secure ci/cd', 'security automation'],
            'Cloud-Security-Engineer': ['cloud security', 'aws security', 'azure security', 'cloud compliance', 'security architect cloud'],
            'Penetration-Tester': ['penetration tester', 'pen tester', 'ethical hacker', 'red team', 'vulnerability assessor'],
            'Networking-Telecom': ['network engineer', 'network administrator', 'telecom engineer', 'network architect', 'ccna', 'ccnp'],
            'Network-Engineer': ['network engineer', 'networking engineer', 'lan/wan', 'network specialist'],
            'Network-Architect': ['network architect', 'network design', 'network infrastructure architect'],
            'Network-Security-Engineer': ['network security', 'firewall engineer', 'network defense', 'ids/ips'],
            'Tech-Management': ['cto', 'vp engineering', 'tech lead', 'engineering manager', 'development manager', 'head of engineering'],
            'Solutions-Architect': ['solutions architect', 'enterprise architect', 'technical architect', 'cloud architect', 'aws architect'],
            'IT-Project-Manager': ['it project manager', 'technical project manager', 'delivery manager', 'scrum master', 'agile coach'],
            'Product-Owner': ['product owner', 'technical product owner', 'product manager', 'digital product'],
            'Chief-Technology-Officer': ['chief technology officer', 'cto', 'chief technical officer'],
            'VP-of-Engineering': ['vp of engineering', 'vice president engineering', 'head of engineering'],
            'IT-Operations-Manager': ['it operations', 'it manager', 'infrastructure manager', 'operations manager'],
            'Gaming-VFX': ['game developer', 'unity developer', 'unreal engine', 'vfx artist', '3d developer', 'game programmer'],
            'Information-Technology': ['it support', 'helpdesk', 'technical support', 'it specialist', 'desktop support', 'it technician'],
            'Engineering': ['civil engineer', 'mechanical engineer', 'electrical engineer', 'chemical engineer', 'industrial engineer'],
            'Finance': ['financial analyst', 'investment banker', 'financial controller', 'finance manager', 'accountant', 'cpa'],
            'Sales': ['sales executive', 'account executive', 'sales manager', 'business development', 'sales representative'],
            'HR': ['hr manager', 'recruiter', 'talent acquisition', 'human resources', 'hr business partner'],
            'Healthcare': ['nurse', 'doctor', 'physician', 'clinical', 'medical', 'healthcare professional'],
            'Design': ['graphic designer', 'ui designer', 'ux designer', 'visual designer', 'creative designer'],
            'Marketing': ['marketing manager', 'digital marketing', 'growth hacker', 'seo specialist', 'content marketer'],
        }
        scores = {cat: sum(1 for s in skills if s in text_lower) for cat, skills in mapping.items()}
        best = max(scores, key=scores.get)
        conf = min(scores[best] / len(mapping[best]), 1.0) if scores[best] > 0 else 0.3
        if conf >= 0.15:
            return best
        return 'default'

    def analyze_resume(self, resume_text: str, job_description: str) -> Dict:
        sections = self._parse_with_spacy(resume_text)
        parsed_jobs = sections.get('parsed_jobs', [])
        parsed_edu = sections.get('parsed_education', [])

        contact = self.extract_contact_info(resume_text)
        skills_obj = self.extract_all_skills(resume_text)
        all_skills = skills_obj['all_skills']
        skill_scores = skills_obj['skill_scores']
        skill_profiles = skills_obj['skill_profiles']
        years = self.extract_years_experience(resume_text)
        education = self.extract_education(resume_text)
        certifications = self.extract_certifications(resume_text)

        scores = self.calculate_match_score(resume_text, job_description)
        cat_pred = self.predict_job_category(resume_text)
        current_job_title = self.extract_current_job_title(resume_text)

        job_recs = self.recommend_jobs(resume_text, years, all_skills, current_job_title, skill_scores)
        if job_recs:
            primary_job = job_recs[0]['job_title']
        else:
            primary_job = self._suggest_job_title(resume_text, cat_pred['predicted_category'], all_skills[:3])
            if primary_job == 'Professional':
                primary_job = cat_pred['predicted_category'].replace('-', ' ') or 'Candidate'

        current_tier = 'unknown'
        for job in parsed_jobs:
            if job.is_current or (not any(j.is_current for j in parsed_jobs) and job == parsed_jobs[0]):
                current_tier = job.company_tier
                break

        edu_level = education['level_name'] if education['level'] > 0 else 'Bachelor'
        salary = self.predict_salary(primary_job, years, len(all_skills),
                                     education_level=edu_level, company_tier=current_tier)

        strengths = self.identify_strengths(resume_text, job_description)
        gaps = self.identify_gaps(resume_text, job_description)
        hire_pred = self.predict_hire_probability(resume_text, job_description)

        match = scores['total']
        if match >= 85: fit = "Strong fit — candidate meets or exceeds most requirements"
        elif match >= 70: fit = "Good fit — solid match with minor development areas"
        elif match >= 50: fit = "Moderate fit — relevant background but notable gaps"
        elif match >= 30: fit = "Partial fit — some transferable skills, significant gaps"
        else: fit = "Limited fit — major skill and experience gaps"

        summary = self._generate_summary(contact, match, years, all_skills, scores, gaps, hire_pred)

        return {
            'summary': summary,
            'match_score': match,
            'match_breakdown': {
                'skill_match': scores['skill_score'],
                'semantic_similarity': scores['similarity_score'],
                'experience_match': scores['experience_score'],
                'education_match': scores['education_score'],
                'skill_depth': scores['skill_depth_score'],
                'company_tier_bonus': scores['company_tier_bonus'],
                'cert_bonus': scores.get('cert_bonus', 0),
                'it_tier_bonus': scores.get('it_tier_bonus', 0),
                'oss_bonus': scores.get('oss_bonus', 0),
                'critical_penalty': scores['critical_penalty'],
                'score_weights_used': scores.get('score_weights_used', {}),
                'job_category_detected': scores.get('job_category_detected', ''),
            },
            'job_category_prediction': cat_pred,
            'strengths': strengths,
            'gaps': gaps,
            'hire_prediction': hire_pred,
            'key_details': {
                'name': contact['name'] or 'Unknown Candidate',
                'email': contact['email'] or 'Not provided',
                'phone': contact['phone'] or 'Not provided',
                'linkedin': contact.get('linkedin') or 'Not provided',
                'github': contact.get('github') or 'Not provided',
                'location': contact.get('location') or 'Not specified',
                'years_experience': f"{years} years" if years > 0 else "Not specified",
                'top_skills': all_skills[:5],
                'predicted_category': cat_pred['predicted_category'],
                'category_confidence': cat_pred['confidence'],
                'education': education,
                'certifications': certifications,
                'current_job_title': current_job_title or 'Not detected',
                'current_company_tier': current_tier
            },
            'parsed_structure': {
                'jobs': [self._job_to_dict(j) for j in parsed_jobs],
                'education': [e.__dict__ for e in parsed_edu],
                'has_skills_section': bool(sections['skills']),
                'has_experience_section': bool(sections['experience']),
                'has_education_section': bool(sections['education'])
            },
            'all_skills_found': all_skills,
            'skill_profiles': skill_profiles,
            'skill_context_scores': skill_scores,
            'years_experience': years,
            'education': education,
            'certifications': certifications,
            'suggested_job': primary_job,
            'suggested_salary': salary['estimated_salary'],
            'salary_prediction': salary,
            'current_job_title': current_job_title,
            'job_recommendations': job_recs,
            'semantic_features': {
                'using_semantic_embeddings': self.use_semantic,
                'using_spacy_parsing': self.use_spacy
            }
        }

    def _generate_summary(self, contact, match, years, all_skills, scores, gaps, hire_pred) -> str:
        name = contact['name'] or 'Candidate'
        if match >= 85:
            summary = f"{name} is a strong candidate with a {match}% match score. "
        elif match >= 60:
            summary = f"{name} shows good potential with a {match}% match score. "
        else:
            summary = f"{name} has a {match}% match score. "
        if years > 0:
            summary += f"Brings {years} years of experience. "
        summary += f"Identified {len(all_skills)} relevant skills. "
        if scores['critical_missing']:
            summary += f"Missing {len(scores['critical_missing'])} critical requirements. "
        else:
            summary += "Covers all critical requirements. "
        if hire_pred.get('hire_probability') is not None:
            prob = hire_pred['hire_probability']
            if prob > 0.7:
                summary += f"Hire probability: {prob:.0%} (recommended)."
            elif prob > 0.4:
                summary += f"Hire probability: {prob:.0%} (consider with reservations)."
            else:
                summary += f"Hire probability: {prob:.0%} (not recommended)."
        return summary.strip()

    def _suggest_job_title(self, resume_text: str, predicted_category: str, top_skills: List[str]) -> str:
        generic_titles = {
            'Information-Technology': 'Software Developer',
            'Finance': 'Financial Analyst',
            'HR': 'HR Specialist',
            'Sales': 'Sales Representative',
            'Marketing': 'Marketing Specialist',
            'Engineering': 'Engineer',
            'Healthcare': 'Healthcare Professional',
            'Designer': 'UI/UX Designer',
            'Business-Development': 'Business Development Manager',
            'Consultant': 'Consultant',
            'Teacher': 'Teacher',
            'Advocate': 'Legal Advocate',
            'Fitness': 'Fitness Trainer',
            'Agriculture': 'Agricultural Specialist',
            'BPO': 'Customer Service Representative',
            'Digital-Media': 'Digital Media Specialist',
            'Automobile': 'Automotive Technician',
            'Chef': 'Chef',
            'Apparel': 'Fashion Designer',
            'Accountant': 'Accountant',
            'Construction': 'Construction Manager',
            'Public-Relations': 'PR Specialist',
            'Banking': 'Banking Officer',
            'Arts': 'Artist',
            'Aviation': 'Pilot',
            'Python Developer': 'Software Development',
            'Frontend Engineer': 'Software Development',
            'Full-Stack Developer': 'Software Development',
            'Data Scientist': 'Data & AI',
            'Machine Learning Engineer': 'Data & AI',
            'Data Analyst': 'Data & AI',
            'DevOps Engineer': 'Infrastructure & Cloud',
            'Cloud Solutions Architect': 'Infrastructure & Cloud',
            'Systems Administrator': 'Infrastructure & Cloud',
            'Cybersecurity Analyst': 'Security',
            'Chief Information Security Officer': 'Security',
            'IT Support Specialist': 'Support & Operations',
            'Scrum Master': 'Project Management',
            'Backend Engineer': 'Software Development',
            'Mobile Application Developer': 'Software Development',
            'Embedded Systems Engineer': 'Software Development',
            'QA Automation Engineer': 'Software Development',
            'Solutions Architect': 'Software Development',
            'Principal Software Engineer': 'Software Development',
            'API Engineer': 'Software Development',
            'Site Reliability Engineer (SRE)': 'Infrastructure & Cloud',
            'Platform Engineer': 'Infrastructure & Cloud',
            'DevSecOps Engineer': 'Security',
            'Release Manager': 'Infrastructure & Cloud',
            'Cloud Engineer': 'Infrastructure & Cloud',
            'Kubernetes Administrator': 'Infrastructure & Cloud',
            'FinOps Analyst': 'Infrastructure & Cloud',
            'Network Engineer': 'Networking',
            'Network Architect': 'Networking',
            'Network Security Engineer': 'Security',
            'Wireless Network Engineer': 'Networking',
            'Network Administrator': 'Networking',
            'Telecom Engineer': 'Networking',
            'Data Center Engineer': 'Infrastructure & Cloud',
            'Database Administrator (DBA)': 'Data & AI',
            'Data Engineer': 'Data & AI',
            'Cloud Security Engineer': 'Security',
            'Penetration Tester': 'Security',
            'IT Project Manager': 'Project Management',
            'Product Owner': 'Project Management',
            'Chief Technology Officer (CTO)': 'Executive Leadership',
            'VP of Engineering': 'Executive Leadership',
            'IT Operations Manager': 'Support & Operations'
        }
        return generic_titles.get(predicted_category, 'Professional')

    def _job_to_dict(self, job: JobEntry) -> Dict:
        return {
            'title': job.title,
            'company': job.company,
            'company_tier': job.company_tier,
            'start_date': job.start_date.strftime('%Y-%m') if job.start_date else None,
            'end_date': job.end_date.strftime('%Y-%m') if job.end_date else None,
            'is_current': job.is_current,
            'years': round(job.years, 1),
            'skills_found': job.skills_found
        }

    def identify_strengths(self, resume_text: str, job_description: str) -> List[str]:
        scores = self.calculate_match_score(resume_text, job_description)
        strengths = []
        if scores['skill_score'] >= 80:
            strengths.append(f"Strong skill match ({scores['skill_score']}%)")
        if scores['critical_matched'] and len(scores['critical_matched']) == len(scores.get('critical_skills', [])):
            strengths.append("All critical requirements met")
        if scores['experience_score'] >= 80:
            strengths.append("Experience level meets or exceeds requirements")
        if scores['company_tier_bonus'] >= 10:
            strengths.append("Relevant experience at top-tier companies")
        if scores.get('cert_bonus', 0) >= 5:
            strengths.append("Valuable certifications present")
        if scores.get('it_tier_bonus', 0) >= 8:
            strengths.append("Deep expertise in core technologies")
        return strengths[:5] if strengths else ["Basic requirements met"]

    def identify_gaps(self, resume_text: str, job_description: str) -> List[str]:
        scores = self.calculate_match_score(resume_text, job_description)
        gaps = []
        if scores['critical_missing']:
            missing = ', '.join(scores['critical_missing'][:3])
            gaps.append(f"Missing critical skills: {missing}")
        if scores['experience_score'] < 60:
            gaps.append("Experience level below requirements")
        if scores.get('it_tier_bonus', 0) < 4 and scores.get('job_category_detected', '') in self._IT_CORE_CATEGORIES:
            gaps.append("Limited depth in core technologies")
        if len(scores.get('missing_skills', [])) > 5:
            gaps.append(f"Many missing skills ({len(scores['missing_skills'])})")
        if scores.get('cert_bonus', 0) == 0 and 'IT' in scores.get('job_category_detected', ''):
            gaps.append("No relevant certifications found")
        return gaps[:4] if gaps else ["No major gaps identified"]


# =================================================================
# BACKWARD COMPATIBILITY WRAPPER
# =================================================================
class EnhancedResumeAnalyzer(ProductionResumeAnalyzer):
    def __init__(self, use_trained_model=False, model_dir='models'):
        super().__init__(
            use_trained_model=use_trained_model,
            model_dir=model_dir,
            use_semantic=True,
            use_spacy=True
        )
        self.fallback_salary_by_job = {
            'Software Engineer': 100000, 'Data Engineer': 105000, 'Data Scientist': 102000,
            'DevOps Engineer': 108000, 'Product Manager': 115000, 'UX Designer': 95000,
            'Frontend Developer': 98000, 'Backend Developer': 102000, 'E-commerce Manager': 80000,
            'Business Analyst': 85000, 'Marketing Manager': 95000, 'Financial Analyst': 88000,
            'Sales Manager': 100000, 'Professional': 60000
        }
        # Expanded job recommendations for the wrapper
        self.job_recommendations = {
            'Software Engineer': {
                'primary_skills': ['python', 'java', 'javascript', 'c++', 'go', 'rust', 'software development', 'coding', 'programming'],
                'secondary_skills': ['git', 'agile', 'rest api', 'microservices', 'design patterns', 'unit testing', 'tdd'],
                'multi_skills': ['web development', 'software development', 'application development'],
                'min_experience': 1,
                'description': 'Develop and maintain software applications'
            },
            'Frontend Developer': {
                'primary_skills': ['javascript', 'react', 'vue', 'angular', 'html', 'css', 'typescript', 'frontend development'],
                'secondary_skills': ['tailwind', 'bootstrap', 'webpack', 'responsive design', 'ui/ux'],
                'multi_skills': ['web development', 'ui design', 'frontend engineering'],
                'min_experience': 1,
                'description': 'Build user-facing web applications'
            },
            'Backend Developer': {
                'primary_skills': ['python', 'java', 'sql', 'rest api', 'node.js', 'django', 'spring boot', 'fastapi', 'backend development'],
                'secondary_skills': ['microservices', 'database', 'api design', 'cloud', 'docker'],
                'multi_skills': ['web development', 'api development', 'server-side development'],
                'min_experience': 1,
                'description': 'Build server-side applications and APIs'
            },
            'Full Stack Developer': {
                'primary_skills': ['javascript', 'react', 'node.js', 'python', 'html', 'css', 'sql', 'full stack development'],
                'secondary_skills': ['git', 'rest api', 'database', 'docker', 'aws'],
                'multi_skills': ['web development', 'frontend development', 'backend development'],
                'min_experience': 2,
                'description': 'Develop end-to-end web applications'
            },
            'Mobile Developer': {
                'primary_skills': ['swift', 'kotlin', 'react native', 'flutter', 'ios', 'android', 'mobile development'],
                'secondary_skills': ['rest api', 'firebase', 'git', 'ui/ux'],
                'multi_skills': ['mobile app development', 'cross-platform development'],
                'min_experience': 1,
                'description': 'Build mobile applications for iOS and Android'
            },
            'Data Scientist': {
                'primary_skills': ['python', 'machine learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'statistics', 'data science'],
                'secondary_skills': ['sql', 'data analysis', 'deep learning', 'nlp', 'scikit-learn'],
                'multi_skills': ['data science', 'machine learning', 'predictive modeling'],
                'min_experience': 2,
                'description': 'Develop ML models and analyze complex datasets'
            },
            'Data Engineer': {
                'primary_skills': ['python', 'sql', 'spark', 'hadoop', 'airflow', 'etl', 'data engineering'],
                'secondary_skills': ['aws', 'azure', 'gcp', 'data warehousing', 'big data'],
                'multi_skills': ['data engineering', 'data pipeline', 'data infrastructure'],
                'min_experience': 2,
                'description': 'Build and maintain data pipelines and infrastructure'
            },
            'Data Analyst': {
                'primary_skills': ['sql', 'excel', 'tableau', 'power bi', 'data analysis', 'statistics', 'python'],
                'secondary_skills': ['pandas', 'data visualization', 'business intelligence'],
                'multi_skills': ['data analysis', 'business analytics', 'reporting'],
                'min_experience': 1,
                'description': 'Analyze data and create visualizations to support decisions'
            },
            'Machine Learning Engineer': {
                'primary_skills': ['python', 'machine learning', 'tensorflow', 'pytorch', 'mlops', 'model deployment'],
                'secondary_skills': ['docker', 'kubernetes', 'aws', 'ci/cd', 'scikit-learn'],
                'multi_skills': ['machine learning', 'ml engineering', 'model lifecycle'],
                'min_experience': 2,
                'description': 'Deploy and maintain machine learning models in production'
            },
            'DevOps Engineer': {
                'primary_skills': ['docker', 'kubernetes', 'aws', 'terraform', 'ansible', 'jenkins', 'ci/cd', 'devops'],
                'secondary_skills': ['linux', 'python', 'shell', 'git', 'monitoring'],
                'multi_skills': ['cloud architecture', 'infrastructure as code', 'devops practices'],
                'min_experience': 2,
                'description': 'Manage infrastructure and deployment pipelines'
            },
            'Cloud Engineer': {
                'primary_skills': ['aws', 'azure', 'gcp', 'cloud architecture', 'terraform', 'cloudformation'],
                'secondary_skills': ['docker', 'kubernetes', 'networking', 'security', 'linux'],
                'multi_skills': ['cloud engineering', 'cloud migration', 'infrastructure as code'],
                'min_experience': 2,
                'description': 'Design and manage cloud infrastructure'
            },
            'Site Reliability Engineer': {
                'primary_skills': ['linux', 'python', 'go', 'monitoring', 'prometheus', 'grafana', 'sre', 'incident response'],
                'secondary_skills': ['kubernetes', 'aws', 'terraform', 'ci/cd'],
                'multi_skills': ['site reliability', 'system engineering', 'observability'],
                'min_experience': 3,
                'description': 'Ensure reliability and performance of production systems'
            },
            'System Administrator': {
                'primary_skills': ['linux', 'windows server', 'networking', 'virtualization', 'backup', 'security', 'system administration'],
                'secondary_skills': ['bash', 'powershell', 'active directory', 'vmware'],
                'multi_skills': ['system administration', 'it operations', 'server management'],
                'min_experience': 2,
                'description': 'Manage servers, networks, and IT infrastructure'
            },
            'Cybersecurity Analyst': {
                'primary_skills': ['security', 'vulnerability assessment', 'siem', 'incident response', 'firewall', 'ids/ips'],
                'secondary_skills': ['python', 'linux', 'risk assessment', 'compliance', 'nist'],
                'multi_skills': ['cybersecurity', 'information security', 'threat analysis'],
                'min_experience': 2,
                'description': 'Protect systems from cyber threats and respond to incidents'
            },
            'Penetration Tester': {
                'primary_skills': ['penetration testing', 'ethical hacking', 'metasploit', 'burp suite', 'kali linux', 'vulnerability assessment'],
                'secondary_skills': ['python', 'linux', 'web security', 'network security'],
                'multi_skills': ['penetration testing', 'offensive security', 'ethical hacking'],
                'min_experience': 2,
                'description': 'Simulate attacks to identify security weaknesses'
            },
            'Product Manager': {
                'primary_skills': ['product strategy', 'market research', 'user stories', 'roadmap', 'agile', 'scrum', 'product management'],
                'secondary_skills': ['data analysis', 'stakeholder management', 'competitive analysis', 'customer development'],
                'multi_skills': ['product management', 'product development', 'agile methodologies'],
                'min_experience': 3,
                'description': 'Define product strategy and roadmap'
            },
            'Project Manager': {
                'primary_skills': ['project management', 'agile', 'scrum', 'jira', 'risk management', 'budgeting', 'stakeholder communication'],
                'secondary_skills': ['leadership', 'team coordination', 'waterfall', 'microsoft project'],
                'multi_skills': ['project management', 'agile delivery', 'team leadership'],
                'min_experience': 3,
                'description': 'Lead projects to successful completion on time and budget'
            },
            'Scrum Master': {
                'primary_skills': ['scrum', 'agile', 'jira', 'team facilitation', 'sprint planning', 'retrospectives'],
                'secondary_skills': ['conflict resolution', 'coaching', 'kanban', 'lean'],
                'multi_skills': ['scrum master', 'agile coaching', 'team empowerment'],
                'min_experience': 2,
                'description': 'Facilitate agile processes and remove impediments'
            },
            'UI/UX Designer': {
                'primary_skills': ['ux design', 'ui design', 'figma', 'sketch', 'wireframing', 'prototyping', 'user research'],
                'secondary_skills': ['adobe xd', 'photoshop', 'illustrator', 'responsive design', 'design thinking'],
                'multi_skills': ['user experience', 'user interface', 'interaction design'],
                'min_experience': 2,
                'description': 'Design user experiences and interfaces'
            },
            'QA Engineer': {
                'primary_skills': ['test automation', 'selenium', 'manual testing', 'bug tracking', 'test cases', 'regression testing'],
                'secondary_skills': ['python', 'java', 'ci/cd', 'junit', 'testng'],
                'multi_skills': ['quality assurance', 'automated testing', 'software testing'],
                'min_experience': 1,
                'description': 'Ensure software quality through testing and automation'
            },
            'Business Analyst': {
                'primary_skills': ['business analysis', 'requirement analysis', 'data analysis', 'stakeholder management', 'process improvement'],
                'secondary_skills': ['sql', 'excel', 'tableau', 'agile', 'user stories'],
                'multi_skills': ['business analysis', 'requirements engineering', 'process modeling'],
                'min_experience': 2,
                'description': 'Bridge business needs with technical solutions'
            },
            'Sales Manager': {
                'primary_skills': ['sales management', 'crm', 'negotiation', 'pipeline management', 'forecasting', 'customer relations'],
                'secondary_skills': ['business development', 'lead generation', 'account management', 'sales strategy'],
                'multi_skills': ['sales management', 'sales operations', 'territory management'],
                'min_experience': 3,
                'description': 'Lead sales team and manage revenue targets'
            },
            'Marketing Manager': {
                'primary_skills': ['digital marketing', 'seo', 'content marketing', 'social media', 'google analytics', 'brand management', 'marketing strategy'],
                'secondary_skills': ['email marketing', 'ppc', 'marketing automation', 'campaign management'],
                'multi_skills': ['marketing strategy', 'digital marketing', 'brand management'],
                'min_experience': 3,
                'description': 'Develop and execute marketing campaigns'
            },
            'HR Manager': {
                'primary_skills': ['recruitment', 'onboarding', 'performance management', 'employee relations', 'hr policies', 'payroll'],
                'secondary_skills': ['hris', 'compensation', 'benefits', 'training', 'compliance'],
                'multi_skills': ['human resources', 'talent acquisition', 'employee engagement'],
                'min_experience': 3,
                'description': 'Manage human resources functions and employee lifecycle'
            },
            'Financial Analyst': {
                'primary_skills': ['financial modeling', 'excel', 'data analysis', 'budgeting', 'forecasting', 'accounting'],
                'secondary_skills': ['sql', 'tableau', 'risk management', 'investment analysis'],
                'multi_skills': ['financial analysis', 'financial modeling', 'business finance'],
                'min_experience': 2,
                'description': 'Analyze financial data and create forecasts'
            },
            'Accountant': {
                'primary_skills': ['accounting', 'bookkeeping', 'tax', 'audit', 'financial reporting', 'quickbooks', 'excel'],
                'secondary_skills': ['gaap', 'ifrs', 'payroll', 'budgeting', 'financial statements'],
                'multi_skills': ['accounting', 'financial reporting', 'tax preparation'],
                'min_experience': 2,
                'description': 'Manage financial records and ensure compliance'
            }
        }

    def extract_residence(self, text: str) -> str:
        contact = self.extract_contact_info(text)
        return contact.get('location') or "Not specified"

    def calculate_similarity(self, resume_text: str, job_description: str) -> float:
        return self.calculate_semantic_similarity(resume_text, job_description)

    def extract_skills(self, text: str) -> List[str]:
        return self._extract_skills_from_text(text)

    def extract_multi_word_skills(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = []
        for skill in self.all_multi_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found.append(skill)
        return list(set(found))

    def extract_skill_phrases(self, text: str) -> List[str]:
        text_lower = text.lower()
        phrases = []
        patterns = [
            r'(?:expert|skilled|proficient|experienced|advanced)\s+(?:in|with)\s+([a-z\s\+]+?)(?:\.|,|;|and|with|$)',
            r'(?:certified|certification)\s+(?:in|as)?\s*([a-z\s\+]+?)(?:\.|,|;|$)',
            r'(?:knowledge|experience|expertise)\s+(?:in|with|of)\s+([a-z\s\+]+?)(?:\.|,|;|and|$)',
            r'(?:used|utilized|worked with)\s+([a-z\s\+]+?)(?:\.|,|;|to|for|$)',
        ]
        for pat in patterns:
            matches = re.findall(pat, text_lower, re.IGNORECASE)
            for m in matches:
                phrase = m.strip().lower()
                phrase = re.sub(r'^(?:of|in|with|and|or)\s+', '', phrase)
                if 2 < len(phrase) and len(phrase.split()) <= 4:
                    phrases.append(phrase)
        return list(set(phrases))

    def extract_all_skills(self, text: str) -> Dict:
        result = super().extract_all_skills(text)
        result['single_word_skills'] = list(set(self._extract_skills_from_text(text)))
        result['multi_word_skills'] = self.extract_multi_word_skills(text)
        result['discovered_phrases'] = self.extract_skill_phrases(text)
        return result

    def extract_years_experience(self, text: str) -> float:
        return int(super().extract_years_experience(text))

    def extract_contact_info(self, text: str) -> Dict:
        info = super().extract_contact_info(text)
        return {
            'name': info['name'],
            'email': info['email'],
            'phone': info['phone']
        }

    def extract_education(self, text: str) -> Dict:
        edu = super().extract_education(text)
        return {
            'level': edu['level'],
            'level_name': edu['level_name'],
            'field': edu['field'],
            'institution': edu.get('institution', 'Unknown'),
            'tier': edu.get('tier', 'unknown')
        }

    def extract_certifications(self, text: str) -> List[str]:
        return super().extract_certifications(text)

    def calculate_match_score(self, resume_text: str, job_description: str) -> Dict:
        result = super().calculate_match_score(resume_text, job_description)
        return {
            'total': result['total'],
            'skill_score': result['skill_score'],
            'similarity_score': result['similarity_score'],
            'experience_score': result['experience_score'],
            'education_score': result['education_score'],
            'skill_diversity': result['skill_depth_score'],
            'skill_depth_score': result['skill_depth_score'],
            'category_alignment': result.get('category_alignment', 0),
            'company_tier_bonus': result['company_tier_bonus'],
            'diversity_bonus': result.get('diversity_bonus', 0),
            'cert_bonus': result.get('cert_bonus', 0),
            'it_tier_bonus': result.get('it_tier_bonus', 0),
            'oss_bonus': result.get('oss_bonus', 0),
            'critical_penalty': result['critical_penalty'],
            'score_weights_used': result.get('score_weights_used', {}),
            'job_category_detected': result.get('job_category_detected', ''),
            'matched_skills': result['matched_skills'],
            'critical_matched': result.get('critical_matched', []),
            'critical_missing': result.get('critical_missing', []),
            'missing_skills': result['missing_skills']
        }

    def identify_strengths(self, resume_text: str, job_description: str) -> List[str]:
        return super().identify_strengths(resume_text, job_description)

    def identify_gaps(self, resume_text: str, job_description: str) -> List[str]:
        return super().identify_gaps(resume_text, job_description)

    def predict_job_category(self, resume_text: str) -> Dict:
        return super().predict_job_category(resume_text)

    def recommend_jobs(self, resume_text: str, years_experience: int, all_skills: List[str],
                       current_job_title: Optional[str] = None,
                       skill_scores: Optional[Dict] = None) -> List[Dict]:
        recommendations = []
        for job, data in self.job_recommendations.items():
            if years_experience < data['min_experience']:
                continue
            primary_match = [s for s in data['primary_skills'] if s in all_skills]
            secondary_match = [s for s in data['secondary_skills'] if s in all_skills]
            multi_match = [s for s in data.get('multi_skills', []) if s in all_skills]
            weighted = (len(primary_match) / len(data['primary_skills']) * 50) if data['primary_skills'] else 0
            weighted += (len(multi_match) / max(len(data.get('multi_skills', [])), 1) * 30)
            weighted += (len(secondary_match) / len(data['secondary_skills']) * 20) if data['secondary_skills'] else 0
            if current_job_title:
                job_lower = job.lower()
                current_lower = current_job_title.lower()
                if job_lower in current_lower or current_lower in job_lower:
                    weighted += 20
            if weighted >= 10:
                recommendations.append({
                    'job_title': job,
                    'match_percentage': round(weighted),
                    'description': data['description'],
                    'primary_skills_found': primary_match,
                    'missing_skills': [s for s in data['primary_skills'] if s not in all_skills]
                })
        recommendations.sort(key=lambda x: x['match_percentage'], reverse=True)
        return recommendations[:5]

    def predict_salary(self, job_title: str, years_experience: int, skills_count: int,
                       education_level: str = 'Bachelor', industry: str = 'Technology',
                       company_size: str = 'Medium', location: str = 'United States',
                       remote_work: str = 'No', certifications: int = 0,
                       company_tier: str = 'unknown') -> Dict:
        return super().predict_salary(job_title, years_experience, skills_count,
                                      education_level, industry, company_size,
                                      location, remote_work, certifications,
                                      company_tier=company_tier)

    def suggest_job_title(self, resume_text: str, predicted_category: str, top_skills: List[str]) -> str:
        return self._suggest_job_title(resume_text, predicted_category, top_skills)

    def estimate_salary(self, job_title: str, experience_years: int) -> int:
        salary_map = {
            'Software Developer': 60000, 'Python Software Developer': 70000,
            'Financial Analyst': 65000, 'HR Specialist': 55000, 'Sales Representative': 50000,
            'Marketing Specialist': 55000, 'Engineer': 65000, 'Healthcare Professional': 60000,
            'UI/UX Designer': 62000, 'Business Development Manager': 75000, 'Consultant': 70000,
            'Teacher': 45000, 'Legal Advocate': 70000, 'Fitness Trainer': 40000,
            'Agricultural Specialist': 45000, 'Customer Service Representative': 35000,
            'Digital Media Specialist': 52000, 'Automotive Technician': 45000, 'Chef': 48000,
            'Fashion Designer': 55000, 'Accountant': 58000, 'Construction Manager': 70000,
            'PR Specialist': 55000, 'Banking Officer': 60000, 'Artist': 45000, 'Pilot': 90000,
            'Professional': 50000
        }
        base = salary_map.get(job_title, 50000)
        exp_multiplier = 1 + min(0.05 * experience_years, 0.5)
        estimated = int(base * exp_multiplier)
        return round(estimated / 1000) * 1000

    def analyze_resume(self, resume_text: str, job_description: str) -> Dict:
        result = super().analyze_resume(resume_text, job_description)
        return {
            'summary': result['summary'],
            'match_score': result['match_score'],
            'match_breakdown': {
                'skill_match': result['match_breakdown']['skill_match'],
                'text_similarity': result['match_breakdown']['semantic_similarity'],
                'experience_match': result['match_breakdown']['experience_match'],
                'skill_diversity': result['match_breakdown']['skill_depth'],
                'cert_bonus': result['match_breakdown'].get('cert_bonus', 0),
                'it_tier_bonus': result['match_breakdown'].get('it_tier_bonus', 0),
                'oss_bonus': result['match_breakdown'].get('oss_bonus', 0),
                'score_weights_used': result['match_breakdown'].get('score_weights_used', {}),
                'job_category_detected': result['match_breakdown'].get('job_category_detected', ''),
            },
            'job_category_prediction': result['job_category_prediction'],
            'strengths': result['strengths'],
            'gaps': result['gaps'],
            'key_details': {
                'name': result['key_details']['name'],
                'email': result['key_details']['email'],
                'phone': result['key_details']['phone'],
                'years_experience': result['key_details']['years_experience'],
                'top_skills': result['key_details']['top_skills'],
                'predicted_category': result['key_details']['predicted_category'],
                'category_confidence': result['key_details']['category_confidence']
            },
            'all_skills_found': result['all_skills_found'],
            'years_experience': int(result['years_experience']),
            'suggested_job': result['suggested_job'],
            'suggested_salary': result['suggested_salary'],
            'residence': result['key_details'].get('location', 'Not specified'),
            'job_recommendations': result['job_recommendations'],
            'salary_prediction': result['salary_prediction'],
            'multi_word_skills': [s for s in result['all_skills_found'] if ' ' in s],
            'discovered_phrases': result.get('discovered_phrases', []),
            'hire_prediction': result.get('hire_prediction', {}),
            'parsed_structure': result.get('parsed_structure', {}),
            'semantic_features': result.get('semantic_features', {})
        }


if __name__ == "__main__":
    # Example usage (can be removed in production)
    pass