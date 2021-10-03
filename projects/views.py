from django.shortcuts import render
from django.http import JsonResponse
from . import utils
from . import models
import os

# Create your views here.


def index(request):
    return render(request, 'projects/index.html')


def qgen(request):
    if request.is_ajax and request.method == 'POST':
        data = request.POST.dict()
        context = data.get('context')
        answer = data.get('answer')
        result = utils.generate_question(context=context, answer=answer)
        response = {'question': result}
        return JsonResponse(response)
        # return render(request,'projects/qgen.html', {'upload':True, 'result':result})

    return render(request, 'projects/qgen.html')


def traffic(request):
    if request.is_ajax and request.method == 'POST':
        request_file = models.Image_data(
            image=request.FILES.get('file'))
        request_file.save()
        result = utils.predict_traffic_sign(request_file.image.name)
        os.remove(request_file.image.name)
        response = {'sign': result}
        return JsonResponse(response)

    return render(request, 'projects/traffic.html')


def agender(request):
    result = {'age': 33, 'gender': 'MALE'}
    if request.method == 'POST':
        print(request.FILES)
        # request_file = models.Image_data(image=request.FILES['file_upload'])
        # request_file.save()
        # print(request_file.image.name)
        # result = utils.predict_agender(request_file.image.name)
        # os.remove(request_file.image.name)
        return render(request, 'projects/age_gender.html', {'upload': True, 'result': result})

    return render(request, 'projects/age_gender.html', {'upload': False})


def en_2_in(request):
    if request.is_ajax and request.method == 'POST':
        data = request.POST.dict()
        en_txt = data.get('english')
        response = {'output': utils.en_2_in(en_txt)}
        return JsonResponse(response)
    return render(request, 'projects/en_2_in.html')


def sketch(request):
    if request.method == 'POST':
        request_file = models.Image_data(image=request.FILES['file_upload'])
        request_file.save()
        result = utils.sketch(request_file.image.name)
        print(request_file.image.name)
        os.remove(request_file.image.name)
        return render(request, 'projects/sketch.html', {'upload': True, 'result': result})

    return render(request, 'projects/sketch.html')
