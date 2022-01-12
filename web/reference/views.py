from django.shortcuts import render
from .models import Prediction_tool
from django.views.generic.list import ListView

# Create your views here.
class Prediction_toolListView(ListView):

    model = Prediction_tool
    paginate_by = 50  # if pagination is desired

    context_object_name = 'tools'
    template_name = 'reference/prediction_tools_list.html'  # Specify your own template name/location