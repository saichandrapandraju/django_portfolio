"""saichandra URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import include
from portfolio import views as port_v
from projects import views as proj_v

urlpatterns = [
    path('', port_v.index, name='index'),
    path('resume', port_v.resume, name='resume'),
    path('blog', port_v.blog, name='blog'),
    path('projects/', proj_v.index, name='projects'),
    path('projects/', include('projects.urls')),
    path('admin/', admin.site.urls),
]
