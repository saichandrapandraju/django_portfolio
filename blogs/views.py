from django.shortcuts import render

# Create your views here.
def blog(request):
    return render(request,'blogs/blog.html')

def tf(request):
    return render(request,'blogs/tf.html')