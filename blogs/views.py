from django.shortcuts import render

# Create your views here.
def blog(request):
    return render(request,'blogs/blog.html')

def tf(request):
    return render(request,'blogs/tf.html')

def xai(request):
    return render(request,'blogs/xai.html')

def ibmfl(request):
    return render(request,'blogs/ibmfl.html')

def aif(request):
    return render(request,'blogs/aif.html')