from django.shortcuts import render
from . import utils
from . import models
import os

# Create your views here.
def index(request):
    return render(request,'projects/index.html')

def qgen(request):
    if request.method == 'POST':
        data = request.POST.dict()
        context = data.get('context')
        answer = data.get('answer')
        result = utils.generate_question(context=context, answer=answer)
        return render(request,'projects/qgen.html', {'upload':True, 'result':result})


    return render(request,'projects/qgen.html', {'upload':False})

def traffic(request):
    if request.method == 'POST':
        request_file = models.Image_data(image=request.FILES['file_upload'])
        request_file.save()
        result = utils.predict_traffic_sign(request_file.image.name)
        os.remove(request_file.image.name)
        return render(request,'projects/traffic.html',{'upload':True, 'result':result})


    return render(request,'projects/traffic.html', {'upload':False})