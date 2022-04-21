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
from django.urls import path
from . import views
from django.contrib.auth.decorators import login_required, permission_required
urlpatterns = [
    path('', views.index, name = "index"),
    path('privacy/de', views.privacy_de, name = "privacy_de"),
    path('privacy/en', views.privacy_en, name = "privacy_en"),
    path('impressum', views.impressum, name = "impressum"),


    ###Sesion###
    path('sesion_index/', views.SessionIndexView.as_view(), name="analysis"),

    path("session_create/", views.SessionCreateView.as_view(),name="new_session"),
    path('session_detail/<slug:session_slug>/', views.SessionDetailView.as_view(), name='session_detail'),
    path('data_detail/<slug:user_slug>/', views.DataDetailView.as_view(), name='data_detail'),
    path('session_detail/<pk>/delete/', views.SessionDeleteView.as_view(), name="session_delete"),


    ####Workflow###
    path("session_detail/<slug:session_slug>/workflow_delete/<int:pk>", views.WorkflowDeleteView.as_view(),name="workflow_delete"),
    path("session_detail/<slug:session_slug>/workflow_correlation/", views.CorrelationWorkflow.as_view(),name="workflow_correlation"),
    path("session_detail/<slug:session_slug>/all_correlation/", views.AllCorrelationWorkflow.as_view(),name="all_correlation"),
    path("session_detail/<slug:session_slug>/workflow_synthetic/", views.SyntheticLethalityWorkflow.as_view(),name="workflow_synthetic"),
    path("session_detail/<slug:session_slug>/workflow_target/", views.TargtePredictorWorkflow.as_view(),name="workflow_target"),
    path("session_detail/<slug:session_slug>/workflow_genesetscore/", views.GeneSetScoreWorkflow.as_view(),name="workflow_genesetscore"),
    path("session_detail/<slug:session_slug>/workflow_immunophenoscore/", views.ImmuneScoreWorkflow.as_view(),name="workflow_immunephenoscore"),
    path("session_detail/<slug:session_slug>/workflow_immunecellinfiltration/", views.ImmuneCellInfiltrationWorkflow.as_view(),name="workflow_immunecellinfiltration"),
    path("session_detail/<slug:session_slug>/workflow_filter/<int:pk>", views.WorkflowFilterView.as_view(),name="workflow_filter"),
    path("session_detail/<slug:session_slug>/workflow_filte_basic/<int:pk>", views.WorkflowFilterBasicView.as_view(),name="workflow_filter_basic"),
    path("session_detail/<slug:session_slug>/workflow_feature/", views.FeatureWorkflow.as_view(),name="workflow_feature"),
    path("session_detail/<slug:session_slug>/workflow_classification/", views.ClassificationWorkflow.as_view(),name="workflow_classification"),
    path("session_detail/<slug:session_slug>/workflow_feature_ratio/", views.FeatureRatioWorkflow.as_view(),name="workflow_feature_ratio"),
    path("session_detail/<slug:session_slug>/workflow_survival/", views.SurvivalFeatureWorkflow.as_view(),name="workflow_survival"),
    path("session_detail/<slug:session_slug>/workflow_survival_ratio/", views.SurvivalFeatureRatioWorkflow.as_view(),name="workflow_survival_ratio"),


    ####Genes####

    path("geneset_list/download_geneset/<int:pk>/<str:identifier>/<str:set_type>/", views.DownloadSet, name="download_geneset"),
    path("geneset_list/download_geneset_gmt/<str:identifier>/<str:set_type>/", views.DownloadAllGMT, name="download_geneset_gmt"),
    path("feature_set/<slug:session_slug>/<int:pk>/", views.FeatureToSet, name="feature_set"),

    ###MIRNA####
    path("data_detail/<slug:user_slug>/delet_mirnaset/<int:pk>", views.MirnasetDeleteView.as_view(), name="delet_mirset"),

    ####File###
    path("session_detail/<slug:session_slug>/download_file/<int:pk_file>", views.DataDownload,name="data_download"),
    path("session_detail/<slug:session_slug>/file_delete/<int:pk>", views.FileDeletView.as_view() ,name="file_delete"),
    path("download_all", views.getfiles, name="download_all"),
    path("save_model/<slug:session_slug>/<int:pk>/", views.SaveModel, name="model_save"),

    ###HTML###
    path("session_detail/<slug:session_slug>/download_html/<int:pk_wrkl>", views.HtmlDownload,name="html_download"),


    ####Dataset###
    path("data_detail/<slug:user_slug>/create_dataset/", views.CreateDatasetView.as_view(), name="create_dataset"),
    path("data_detail/<slug:user_slug>/delet_dataset/<int:pk>", views.DatasetDeleteView.as_view(), name="delet_dataset"),
    path("dataset_list/", views.DatasetListView.as_view(), name="dataset_list"),
    path("dataset_list/<slug:search>", views.DatasetListView.as_view(), name="dataset_list"),
    path("metadata_list/", views.DatasetListJson, name = "MetadataListJson"),
    path("metadata_detail/<int:pk>", views.DatasetDetailView.as_view(), name = "metadata_detail"),
    path("not_owner/", views.handlernotowner, name="not_owner"),
    path("dataset_download/<int:pk>/<slug:file_name>", views.dataset_download, name = "dataset_download"),

    #### Search ####
    path("search/<slug:query>", views.SearchDetailView.as_view(), name = "search_view"),



]

