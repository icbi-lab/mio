"""mirtariniel_v2 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from os import name
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    ### Correlation View ####
    path("results_detail/<slug:session_slug>/workflow_filter/<int:pk>", views.ReultsDetailView.as_view(), name = "results_detail"),
    path("results_detail/<slug:session_slug>/cytoscape/<slug:identifier_data>/<slug:identifier_style>", views.CytoscapeView.as_view(), name = "cytoscape_full"),

    #path("results_detail/<slug:session_slug>/workflow_filter/<int:pk>", views.TestView.as_view(), name = "results_detail"),

    #### Synthetic Lethal View####
    path("results_detail/<slug:session_slug>/synthetic_filter/", views.ReultsSyntheticView.as_view(), name = "results_synthetic"),

    #### Target Prediction ####
    path("results_detail/<slug:session_slug>/target_predictor/", views.TargetView.as_view(), name = "results_target"),
    
    #### Kaplan Meier ####
    path("survival/<slug:session_slug>/kaplan_meier_form/", views.KaplanMeierFormView.as_view(), name = "survival_formkm"),
    path("survival/<slug:session_slug>/kaplan_meier_plot/", views.KaplanMeierView.as_view(), name = "survival_km"),

    ### Feature View ####
    path("feature_list/<slug:identifier>", views.FeatureListJson, name = "results_json"),
    path("results_detail/<slug:session_slug>/workflow_feature/<int:pk>", views.FeatureDetailView.as_view(), name = "results_feature"),

    ### Survival View ####

    path("feature_list/", views.FeatureListJson, name = "FeatureListJson"),
    path("results_detail/<slug:session_slug>/workflow_survival/<int:pk>", views.SurvivalDetailView.as_view(), name = "results_survival"),

    ### Classification View ####

    path("results_detail/<slug:session_slug>/workflow_classification/<int:pk>", views.ClassificationView.as_view(), name = "results_classification"),

    path("cytoscape/<slug:identifier>", views.CytoListJson, name = "cytoscape_json"),

        
]