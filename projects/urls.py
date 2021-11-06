from django.contrib import admin
from django.urls import path
from django.urls import include
from projects import views

app_name = 'projects'

urlpatterns = [
    path('', views.index, name='index'),
    path(r'qgen', views.qgen, name='qgen'),
    path(r'traffic', views.traffic, name='traffic'),
    path(r'agender', views.agender, name='agender'),
    path(r'en_2_in', views.en_2_in, name='en_2_in'),
    path(r'sketch', views.sketch, name='sketch'),
    path(r'ai4code', views.ai4code, name='ai4code'),
]
