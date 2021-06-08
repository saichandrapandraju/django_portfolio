from django.contrib import admin
from django.urls import path
from django.urls import include
from projects import views

app_name = 'projects'

urlpatterns = [
    path('', views.index, name='index'),
    path('qgen', views.qgen, name='qgen'),
    path('traffic', views.traffic, name='traffic')
]