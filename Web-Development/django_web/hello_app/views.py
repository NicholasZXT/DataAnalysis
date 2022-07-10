from django.shortcuts import render
from django.shortcuts import HttpResponse

# Create your views here.
def hello_index(response):
    return HttpResponse('Hello, this is django hello index.')