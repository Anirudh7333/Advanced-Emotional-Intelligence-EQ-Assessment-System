"""
Core logic for EQ Assessment using HuggingFace transformers.
"""
from transformers import pipeline
import numpy as np


class AdvancedEQAssessmentModel:
    """
    Advanced Emotional Intelligence Assessment Model.
    
    This class handles the complete EQ assessment pipeline:
    - Scenario and question generation
    - Response validation
    - Emotion and sentiment analysis using HuggingFace models
    - EQ category scoring
    - Overall EQ interpretation
    """
    
    def __init__(self):
        """Initialize the model by loading HuggingFace pipelines."""
        print("Loading HuggingFace models...")
        
        # Load sentiment analysis pipeline
        # Using a lightweight model for faster inference
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=False
        )
        
        # Load emotion analysis pipeline
        # This model provides multiple emotion labels with scores
        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        
        # Define EQ categories
        self.eq_categories = [
            "self_awareness",
            "emotional_resilience",
            "conflict_resolution",
            "cultural_awareness",
            "empathy",
            "stress_management"
        ]
        
        # Minimum words required per response
        self.min_words_per_response = 10
        
        print("Models loaded successfully!")
    
    def generate_scenario(self, age: int, gender: str, profession: str) -> str:
        """
        Generate a tailored, challenging scenario based on demographics.
        
        Args:
            age: User's age
            gender: User's gender
            profession: User's profession
            
        Returns:
            A scenario text string tailored to the profession
        """
        profession_lower = profession.lower()
        
        # Teacher/Lecturer scenarios
        if "teacher" in profession_lower or "lecturer" in profession_lower or "educator" in profession_lower:
            return (
                "You are in the middle of teaching an important lesson when a student "
                "suddenly starts loudly criticizing your teaching method in front of the entire class, "
                "accusing you of being unfair and biased. Several other students begin nodding in "
                "agreement, and the atmosphere becomes tense and uncomfortable. You have 10 minutes "
                "left in the class and an important topic to cover before the upcoming exam."
            )
        
        # Healthcare scenarios
        elif "nurse" in profession_lower or "doctor" in profession_lower or "physician" in profession_lower:
            return (
                "You are working in a busy hospital ward during a night shift. A patient's family member "
                "approaches you with extreme anger, blaming you for a delayed medication dose that "
                "has caused their relative discomfort. They raise their voice, attracting attention from "
                "other patients and staff. The family member threatens to file a complaint and questions "
                "your competence. Meanwhile, you have other critical patients requiring immediate attention."
            )
        
        # Management/Leadership scenarios
        elif "manager" in profession_lower or "lead" in profession_lower or "director" in profession_lower:
            return (
                "You are leading a team project with a tight deadline. Two of your team members are "
                "engaged in a heated argument during a critical meeting, each blaming the other for "
                "missed deadlines and poor quality work. The conflict escalates, with personal attacks "
                "being exchanged. The rest of the team looks uncomfortable, and the project deadline "
                "is in 48 hours. Your supervisor is expecting a status update in 2 hours."
            )
        
        # Generic workplace scenario
        else:
            return (
                "You have just presented your work to a group of colleagues and stakeholders. "
                "A senior colleague publicly criticizes your approach, pointing out what they perceive "
                "as fundamental flaws in your methodology. Their tone is condescending, and several "
                "others in the room begin questioning your decisions. You spent weeks preparing this "
                "work and believe in its value, but now you're facing public scrutiny and doubt."
            )
    
    def generate_questions(self, scenario: str) -> list[str]:
        """
        Generate reflective, context-aware questions based on the scenario.
        
        Args:
            scenario: The generated scenario text
            
        Returns:
            A list of 3-5 reflective questions
        """
        questions = [
            "What emotions would you feel in this situation, and why? Describe the intensity and sequence of your emotional reactions.",
            "How would you manage your emotions before responding to the situation? What strategies would you use to stay composed?",
            "How would you resolve this conflict or address this challenge? Describe your approach and the reasoning behind it.",
            "How would you cope with the stress and pressure of this situation? What internal and external resources would you draw upon?",
            "What would you learn from this experience, and how might it affect your future behavior in similar situations?"
        ]
        
        return questions
    
    def validate_responses(self, responses: list[str]) -> tuple[bool, str | None]:
        """
        Validate that all responses meet quality requirements.
        
        Args:
            responses: List of response strings
            
        Returns:
            Tuple of (is_valid: bool, error_message: str | None)
        """
        for i, response in enumerate(responses):
            # Check if response is empty or whitespace
            if not response or not response.strip():
                return (False, f"Answer {i + 1} cannot be empty. Please provide a thoughtful response.")
            
            # Count words
            words = response.strip().split()
            if len(words) < self.min_words_per_response:
                return (
                    False,
                    f"Answer {i + 1} is too short. Please provide at least {self.min_words_per_response} words "
                    f"(currently {len(words)} words). Your response should be detailed and reflective."
                )
        
        return (True, None)
    
    def analyze_single_response(self, text: str) -> dict:
        """
        Analyze a single response for sentiment and emotions.
        
        Args:
            text: The response text to analyze
            
        Returns:
            Dictionary containing sentiment and emotion analysis results
        """
        # Sentiment analysis
        sentiment_result_raw = self.sentiment_pipeline(text)
        
        # Handle different return formats
        if isinstance(sentiment_result_raw, list) and len(sentiment_result_raw) > 0:
            sentiment_result = sentiment_result_raw[0]
        elif isinstance(sentiment_result_raw, dict):
            sentiment_result = sentiment_result_raw
        else:
            # Default fallback
            sentiment_result = {'label': 'NEUTRAL', 'score': 0.5}
        
        sentiment_label = sentiment_result.get('label', 'NEUTRAL').upper()
        sentiment_score = float(sentiment_result.get('score', 0.5))
        
        # Map sentiment labels to standard format
        if sentiment_label in ['POSITIVE', 'POS']:
            sentiment_label = 'POSITIVE'
        elif sentiment_label in ['NEGATIVE', 'NEG']:
            sentiment_label = 'NEGATIVE'
        else:
            sentiment_label = 'NEUTRAL'
        
        # Emotion analysis
        try:
            emotion_results = self.emotion_pipeline(text)
        except Exception as e:
            # Fallback if emotion analysis fails
            print(f"Emotion analysis error: {e}")
            emotion_results = []
        
        # Convert emotion results to dictionary
        # Handle different possible return formats
        emotion_scores = {}
        if emotion_results:
            # Check if results is a list
            if isinstance(emotion_results, list):
                for emotion_item in emotion_results:
                    # Handle both dict and tuple formats
                    if isinstance(emotion_item, dict):
                        label = emotion_item.get('label', '').lower()
                        score = emotion_item.get('score', 0.0)
                        if label:
                            emotion_scores[label] = float(score)
                    elif isinstance(emotion_item, (list, tuple)) and len(emotion_item) >= 2:
                        # Handle tuple format: (label, score) or (label, {'score': ...})
                        label = str(emotion_item[0]).lower()
                        score_value = emotion_item[1]
                        
                        # Handle if score is a dictionary
                        if isinstance(score_value, dict):
                            score = float(score_value.get('score', 0.0))
                        elif isinstance(score_value, (int, float)):
                            score = float(score_value)
                        else:
                            # Try to convert to float, default to 0.0 if fails
                            try:
                                score = float(score_value)
                            except (ValueError, TypeError):
                                score = 0.0
                        
                        if label:
                            emotion_scores[label] = score
                    # Skip if format is unexpected
            # Handle case where result is a single dictionary
            elif isinstance(emotion_results, dict):
                # Check if it's a single emotion result
                if 'label' in emotion_results:
                    label = emotion_results.get('label', '').lower()
                    score = emotion_results.get('score', 0.0)
                    if label:
                        emotion_scores[label] = float(score)
                else:
                    # It might be a dictionary of emotions
                    for key, value in emotion_results.items():
                        if isinstance(value, (int, float)):
                            emotion_scores[str(key).lower()] = float(value)
                        elif isinstance(value, dict) and 'score' in value:
                            emotion_scores[str(key).lower()] = float(value['score'])
        
        # Find primary emotion
        primary_emotion = None
        primary_emotion_score = 0.0
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            primary_emotion_score = emotion_scores[primary_emotion]
        
        return {
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "emotion_scores": emotion_scores,
            "primary_emotion": primary_emotion,
            "primary_emotion_score": primary_emotion_score,
        }
    
    def analyze_responses(self, responses: list[str]) -> list[dict]:
        """
        Analyze multiple responses.
        
        Args:
            responses: List of response strings
            
        Returns:
            List of analysis dictionaries
        """
        analyses = []
        for response in responses:
            analysis = self.analyze_single_response(response)
            analyses.append(analysis)
        return analyses
    
    def calculate_eq_scores(
        self, 
        analyses: list[dict], 
        demographics: dict
    ) -> tuple[dict, float]:
        """
        Calculate EQ category scores and overall EQ score from analyses.
        
        Args:
            analyses: List of analysis dictionaries from analyze_responses
            demographics: Dictionary with 'age', 'gender', 'profession'
            
        Returns:
            Tuple of (category_scores: dict, overall_score: float)
        """
        # Aggregate sentiment across all responses
        total_sentiment_positive = 0.0
        total_sentiment_negative = 0.0
        total_sentiment_neutral = 0.0
        sentiment_count = 0
        
        # Aggregate emotion scores across all responses
        emotion_totals = {}
        
        for analysis in analyses:
            # Aggregate sentiment
            sentiment_label = analysis['sentiment_label']
            sentiment_score = analysis['sentiment_score']
            
            if sentiment_label == 'POSITIVE':
                total_sentiment_positive += sentiment_score
            elif sentiment_label == 'NEGATIVE':
                total_sentiment_negative += sentiment_score
            else:
                total_sentiment_neutral += sentiment_score
            
            sentiment_count += 1
            
            # Aggregate emotions
            for emotion_label, score in analysis['emotion_scores'].items():
                if emotion_label not in emotion_totals:
                    emotion_totals[emotion_label] = 0.0
                emotion_totals[emotion_label] += score
        
        # Normalize sentiment to ratios
        if sentiment_count > 0:
            positive_ratio = total_sentiment_positive / sentiment_count
            negative_ratio = total_sentiment_negative / sentiment_count
            neutral_ratio = total_sentiment_neutral / sentiment_count
        else:
            positive_ratio = negative_ratio = neutral_ratio = 0.33
        
        # Normalize emotion scores to ratios (sum of all emotions per response)
        emotion_ratios = {}
        total_emotion_score = sum(emotion_totals.values())
        if total_emotion_score > 0:
            for emotion_label, total_score in emotion_totals.items():
                emotion_ratios[emotion_label] = total_score / total_emotion_score
        else:
            # Default uniform distribution if no emotions detected
            emotion_ratios = {label: 1.0 / len(emotion_totals) if emotion_totals else 0.0 
                            for label in emotion_totals}
        
        # Extract specific emotion ratios (with defaults if not present)
        joy_ratio = emotion_ratios.get('joy', 0.0)
        sadness_ratio = emotion_ratios.get('sadness', 0.0)
        anger_ratio = emotion_ratios.get('anger', 0.0)
        fear_ratio = emotion_ratios.get('fear', 0.0)
        disgust_ratio = emotion_ratios.get('disgust', 0.0)
        surprise_ratio = emotion_ratios.get('surprise', 0.0)
        love_ratio = emotion_ratios.get('love', 0.0)
        neutral_emotion_ratio = emotion_ratios.get('neutral', 0.0)
        
        # Calculate EQ category scores using heuristics
        # Base score of 50, then adjust by ±50 based on emotion/sentiment indicators
        
        # Self-awareness: ability to recognize and understand own emotions
        self_awareness = 50 + 50 * (joy_ratio + love_ratio - sadness_ratio - anger_ratio)
        
        # Emotional resilience: ability to bounce back from negative emotions
        emotional_resilience = 50 + 50 * (positive_ratio - negative_ratio - fear_ratio * 0.5)
        
        # Conflict resolution: ability to handle conflicts constructively
        conflict_resolution = 50 + 50 * (positive_ratio - anger_ratio - disgust_ratio)
        
        # Cultural awareness: openness and adaptability
        cultural_awareness = 50 + 50 * (neutral_ratio + love_ratio - disgust_ratio)
        
        # Empathy: ability to understand and share feelings of others
        empathy = 50 + 50 * (love_ratio + sadness_ratio * 0.7 - disgust_ratio)
        
        # Stress management: ability to manage stress effectively
        stress_management = 50 + 50 * (positive_ratio - fear_ratio - anger_ratio)
        
        # Apply demographic adjustments (mild)
        age = demographics.get('age', 30)
        age_factor = min(1.0, age / 60.0)  # Slightly higher resilience with age
        
        emotional_resilience += age_factor * 5
        stress_management += age_factor * 3
        
        # Clamp all scores to [0, 100]
        category_scores = {
            "self_awareness": max(0, min(100, self_awareness)),
            "emotional_resilience": max(0, min(100, emotional_resilience)),
            "conflict_resolution": max(0, min(100, conflict_resolution)),
            "cultural_awareness": max(0, min(100, cultural_awareness)),
            "empathy": max(0, min(100, empathy)),
            "stress_management": max(0, min(100, stress_management)),
        }
        
        # Overall EQ score is the mean of all category scores
        overall_score = np.mean(list(category_scores.values()))
        
        return (category_scores, float(overall_score))
    
    def interpret_overall_eq(self, overall_score: float) -> str:
        """
        Interpret the overall EQ score into a categorical level.
        
        Args:
            overall_score: Overall EQ score (0-100)
            
        Returns:
            EQ level string: "Low EQ", "Average EQ", or "High EQ"
        """
        if overall_score < 40:
            return "Low EQ"
        elif overall_score <= 70:
            return "Average EQ"
        else:
            return "High EQ"

