from django.contrib import admin
from django.urls import path
from django.urls import include
from blogs import views

app_name = 'blogs'

urlpatterns = [
    path('', views.blog, name='blogs'),
    path('tf', views.tf, name='tf')
]