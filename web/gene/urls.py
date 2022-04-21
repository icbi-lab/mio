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
from django.contrib import admin
from django.urls import path
from django.views.generic.base import View
from . import views
from django.contrib.auth.decorators import login_required, permission_required
urlpatterns = [
    ####Genes####
    path('gene_view/<slug:gene_id>', views.GeneDetailView.as_view(), name = "gene_view"),
    path('geneset_view/<slug:geneset_id>', views.GenesetDetailView.as_view(), name = "geneset_view"),
    path("create_gene/", views.GeneUploadView.as_view(),name="create_genes"),
    path("data_detail/<slug:user_slug>/create_geneset/", views.CreateSetView.as_view(), name="create_geneset"),
    path("data_detail/<slug:user_slug>/delet_geneset/<int:pk>", views.GenesetDeleteView.as_view(), name="delet_geneset"),
    #path("data_detail/<slug:user_slug>/update_geneset/<int:pk>", views.GenesetUpdateView.as_view(), name="update_geneset"),
    path("geneset_list/", views.GenesetListView.as_view(), name="geneset_list"),
    path("geneset_list/<slug:search>", views.GenesetListView.as_view(), name="geneset_list"),
    path("data_detail/<slug:user_slug>/create_geneset_gmt/", views.CreateGeneSetFromGMTView.as_view(), name="create_geneset_gmt"),
    ###MIRNA####
    path("data_detail/<slug:user_slug>/create_mirset/", views.CreateSetView.as_view(), name="create_mirset"),
    
]

