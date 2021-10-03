from django.contrib import admin
from django.urls import path
from django.urls import include
from projects import views

app_name = 'projects'

urlpatterns = [
    path('', views.index, name='index'),
    path(r'qgen', views.qgen, name='qgen'),
    path('traffic', views.traffic, name='traffic'),
    path('agender', views.agender, name='agender'),
    path('en_2_in', views.en_2_in, name='en_2_in'),
    path('sketch', views.sketch, name='sketch'),
]
