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
from django.urls import path, include
from . import views
from django.conf.urls import url

urlpatterns = [
        ###User###
    #path('accounts/profile/', views.UserDetailView.as_view(),name="user_detail"),
    path('accounts/profile/<slug:user_slug>/update_profile', views.UserUpdateView.as_view(),name="user_update"),
    path('accounts/temporal/<slug:user_slug>/make_permanent', views.UserUpdateTemporalView.as_view(),name="temporal_update"),
    path('accounts/password/change/', views.PasswordChangeView.as_view(), name="password_change"),
    path('accounts/temporal', views.UserCreateTemporal.as_view(), name="create_temporal"),

    #path('/users/reset/NA/$', password_reset, name='password-reset')
    ]