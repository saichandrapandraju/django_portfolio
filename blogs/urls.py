from django.urls import path
from blogs import views

app_name = 'blogs'

urlpatterns = [
    path('', views.blog, name='blogs'),
    path('tf', views.tf, name='tf'),
    path('xai', views.xai, name='xai'),
    path('ibmfl', views.ibmfl, name='ibmfl'),
    path('aif', views.aif, name='aif')
]