"""
Forms for the EQ Assessment application.
"""
from django import forms


class DemographicForm(forms.Form):
    """Form for collecting user demographics."""
    
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other'),
        ('prefer_not_say', 'Prefer not to say'),
    ]
    
    age = forms.IntegerField(
        label='Age',
        min_value=10,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your age',
            'required': True
        })
    )
    
    gender = forms.ChoiceField(
        label='Gender',
        choices=GENDER_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control',
            'required': True
        })
    )
    
    profession = forms.CharField(
        label='Profession',
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Teacher, Nurse, Manager, Software Engineer',
            'required': True
        })
    )


