"""
Local AI Analyzer Wrapper for ATS+
Integrates the enhanced trained ML models into the ATS+ system.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ml_resume_analyzer_enhanced import EnhancedResumeAnalyzer
    print("✅ Local AI analyzer module loaded successfully")
except ImportError as e:
    print(f"⚠️ Could not load local analyzer: {e}")
    print("   Local AI option will fall back to pattern matching")

class LocalAnalyzerWrapper:
    def __init__(self, models_dir="models"):
        self.analyzer = None
        self.models_dir = models_dir
        self.use_trained_model = False
        self._initialize()
    
    def _initialize(self):
        try:
            self.analyzer = EnhancedResumeAnalyzer(use_trained_model=True, model_dir=self.models_dir)
            self.use_trained_model = self.analyzer.use_trained_model
            if self.use_trained_model:
                print(f"✅ Local AI: Using trained models from '{self.models_dir}'")
                print(f"   - Categories: {len(self.analyzer.job_categories)}")
                print(f"   - Single-word skills: {len(self.analyzer.all_single_skills)}")
                print(f"   - Multi-word skills: {len(self.analyzer.all_multi_skills)}")
            else:
                print(f"⚠️ Local AI: No trained models found, using pattern matching")
        except Exception as e:
            print(f"⚠️ Local AI initialization error: {e}")
            self.use_trained_model = False
    
    def analyze_resume(self, resume_text, job_description):
        if self.analyzer is None:
            return self._get_error_result("Analyzer not initialized")
        try:
            result = self.analyzer.analyze_resume(resume_text, job_description)
            return {
                "name": result['key_details'].get('name', 'Unknown'),
                "job": result['key_details'].get('predicted_category', result.get('job_category_prediction', {}).get('predicted_category', 'Not Classified')),
                "experience": result.get('years_experience', 0),
                "score": result.get('match_score', 0),
                "summary": result.get('summary', ''),
                "key_strengths": result.get('strengths', []),
                "identified_gaps": result.get('gaps', []),
                "email": result['key_details'].get('email', 'Not found'),
                "phone": result['key_details'].get('phone', 'Not found'),
                "top_skills": result['key_details'].get('top_skills', []),
                "all_skills_found": result.get('all_skills_found', []),
                "match_breakdown": result.get('match_breakdown', {}),
                "category_confidence": result['key_details'].get('category_confidence', 0),
                "method": "local_trained_model" if self.use_trained_model else "local_pattern_matching",
                "suggested_job": result.get('suggested_job', 'Not specified'),
                "suggested_salary": result.get('suggested_salary', 0),
                "salary_range": result.get('salary_range', (0, 0)),
                "job_recommendations": result.get('job_recommendations', []),
                "multi_word_skills": result.get('multi_word_skills', []),
                "discovered_phrases": result.get('discovered_phrases', []),
                "residence": result.get('residence', 'Not specified')
            }
        except Exception as e:
            print(f"Local analysis error: {e}")
            return self._get_error_result(str(e))
    
    def _get_error_result(self, error_msg):
        return {
            "name": "Analysis Error",
            "job": "Processing Failed",
            "experience": 0,
            "score": 0,
            "summary": f"Local AI analysis failed: {error_msg}",
            "key_strengths": [],
            "identified_gaps": ["Check server logs for details"],
            "email": "Not found",
            "phone": "Not found",
            "top_skills": [],
            "all_skills_found": [],
            "method": "error",
            "suggested_job": "Error",
            "suggested_salary": 0,
            "job_recommendations": [],
            "residence": "Not specified"
        }
    
    def get_model_info(self):
        if self.analyzer and self.use_trained_model:
            return {
                "type": "trained_random_forest",
                "categories": self.analyzer.job_categories,
                "num_categories": len(self.analyzer.job_categories),
                "single_word_skills": len(self.analyzer.all_single_skills),
                "multi_word_skills": len(self.analyzer.all_multi_skills),
                "model_dir": self.models_dir
            }
        else:
            return {
                "type": "pattern_matching_fallback",
                "categories": [],
                "num_categories": 0,
                "single_word_skills": 0,
                "multi_word_skills": 0,
                "model_dir": self.models_dir,
                "note": "Run python train_model_enhanced.py to train models"
            }

_local_analyzer = None

def get_local_analyzer():
    global _local_analyzer
    if _local_analyzer is None:
        _local_analyzer = LocalAnalyzerWrapper()
    return _local_analyzer