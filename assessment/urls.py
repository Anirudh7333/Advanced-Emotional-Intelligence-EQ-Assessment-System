"""
URL configuration for the assessment app.
"""
from django.urls import path
from . import views

app_name = 'assessment'

urlpatterns = [
    path('', views.landing_view, name='landing'),
    path('respond/', views.response_view, name='respond'),
    path('result/', views.result_view, name='result'),
]


