from django.urls import path
from Home import views

urlpatterns = [
    path('', views.index),
]
