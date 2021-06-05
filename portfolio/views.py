from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request,'portfolio/index.html')

def resume(request):
    return render(request,'portfolio/resume.html')

def projects(request):
    return render(request,'portfolio/projects.html')

def blog(request):
    return render(request,'portfolio/blog.html')