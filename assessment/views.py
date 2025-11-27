"""
Views for the EQ Assessment application.
"""
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import DemographicForm
from .eq_model import AdvancedEQAssessmentModel

# Initialize the EQ model as a module-level singleton
# This avoids reloading the models on every request
eq_model = AdvancedEQAssessmentModel()


def landing_view(request):
    """
    Landing page view for collecting user demographics.
    
    GET: Display the demographics form
    POST: Process form, generate scenario and questions, redirect to response view
    """
    if request.method == 'POST':
        form = DemographicForm(request.POST)
        if form.is_valid():
            # Extract demographics
            age = form.cleaned_data['age']
            gender = form.cleaned_data['gender']
            profession = form.cleaned_data['profession']
            
            demographics = {
                'age': age,
                'gender': gender,
                'profession': profession,
            }
            
            # Generate scenario and questions
            scenario = eq_model.generate_scenario(age, gender, profession)
            questions = eq_model.generate_questions(scenario)
            
            # Store in session
            request.session['demographics'] = demographics
            request.session['scenario'] = scenario
            request.session['questions'] = questions
            
            # Redirect to response view
            return redirect('assessment:respond')
    else:
        form = DemographicForm()
    
    return render(request, 'assessment/landing.html', {'form': form})


def response_view(request):
    """
    Response collection view for scenario questions.
    
    GET: Display scenario and questions with textareas
    POST: Validate responses, analyze them, calculate scores, redirect to result view
    """
    # Check if session data exists
    if 'scenario' not in request.session or 'questions' not in request.session:
        messages.error(request, 'Please start the assessment from the beginning.')
        return redirect('assessment:landing')
    
    scenario = request.session['scenario']
    questions = request.session['questions']
    demographics = request.session.get('demographics', {})
    
    if request.method == 'POST':
        # Collect responses from form
        responses = []
        for i in range(len(questions)):
            answer_key = f'answer_{i}'
            response_text = request.POST.get(answer_key, '').strip()
            responses.append(response_text)
        
        # Validate responses
        is_valid, error_message = eq_model.validate_responses(responses)
        
        if not is_valid:
            return render(request, 'assessment/scenario.html', {
                'scenario': scenario,
                'questions': questions,
                'error_message': error_message,
            })
        
        # Analyze responses
        analyses = eq_model.analyze_responses(responses)
        
        # Calculate EQ scores
        category_scores, overall_score = eq_model.calculate_eq_scores(analyses, demographics)
        
        # Interpret overall EQ
        eq_level = eq_model.interpret_overall_eq(overall_score)
        
        # Compute sentiment summary (aggregate across all responses)
        sentiment_counts = {'POSITIVE': 0.0, 'NEGATIVE': 0.0, 'NEUTRAL': 0.0}
        for analysis in analyses:
            label = analysis['sentiment_label']
            score = analysis['sentiment_score']
            if label in sentiment_counts:
                sentiment_counts[label] += score
        
        # Normalize sentiment to percentages
        total_sentiment = sum(sentiment_counts.values())
        if total_sentiment > 0:
            sentiment_percent = {
                label: (count / total_sentiment) * 100
                for label, count in sentiment_counts.items()
            }
        else:
            sentiment_percent = {'POSITIVE': 33.3, 'NEGATIVE': 33.3, 'NEUTRAL': 33.3}
        
        # Compute emotion summary (aggregate across all responses)
        emotion_totals = {}
        for analysis in analyses:
            for emotion_label, score in analysis['emotion_scores'].items():
                if emotion_label not in emotion_totals:
                    emotion_totals[emotion_label] = 0.0
                emotion_totals[emotion_label] += score
        
        # Normalize emotions to percentages
        total_emotion = sum(emotion_totals.values())
        if total_emotion > 0:
            emotion_percent = {
                label: (total / total_emotion) * 100
                for label, total in emotion_totals.items()
            }
        else:
            emotion_percent = {}
        
        # Sort emotions by percentage (descending)
        emotion_percent = dict(sorted(
            emotion_percent.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Store results in session
        request.session['category_scores'] = category_scores
        request.session['overall_score'] = float(overall_score)
        request.session['eq_level'] = eq_level
        request.session['sentiment_percent'] = sentiment_percent
        request.session['emotion_percent'] = emotion_percent
        
        # Redirect to result view
        return redirect('assessment:result')
    
    # GET request - display scenario and questions
    return render(request, 'assessment/scenario.html', {
        'scenario': scenario,
        'questions': questions,
    })


def result_view(request):
    """
    Results display view showing EQ scores and interpretations.
    
    GET: Display comprehensive results with visualizations
    """
    # Check if session data exists
    if 'overall_score' not in request.session:
        messages.error(request, 'Please complete the assessment to view results.')
        return redirect('assessment:landing')
    
    demographics = request.session.get('demographics', {})
    scenario = request.session.get('scenario', '')
    category_scores = request.session.get('category_scores', {})
    overall_score = request.session.get('overall_score', 0.0)
    eq_level = request.session.get('eq_level', 'Average EQ')
    sentiment_percent = request.session.get('sentiment_percent', {})
    emotion_percent = request.session.get('emotion_percent', {})
    
    # Format gender for display
    gender_display = {
        'male': 'Male',
        'female': 'Female',
        'other': 'Other',
        'prefer_not_say': 'Prefer not to say',
    }.get(demographics.get('gender', ''), demographics.get('gender', ''))
    
    context = {
        'demographics': demographics,
        'gender_display': gender_display,
        'scenario': scenario,
        'category_scores': category_scores,
        'overall_score': overall_score,
        'eq_level': eq_level,
        'sentiment_percent': sentiment_percent,
        'emotion_percent': emotion_percent,
    }
    
    return render(request, 'assessment/result.html', context)

