from django.shortcuts import render
from . import utils

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