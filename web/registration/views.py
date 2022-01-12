from django.views.generic import (View, CreateView)
from django.views.generic.edit import UpdateView
from allauth.account.views import PasswordChangeView
from django.shortcuts import render, redirect, get_object_or_404
from .models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login

# Create your views here.

class UserCreateTemporal(View):
    """
    Main View of the page. We obtain all the information related with the Session thanks to
    the session identifier
    """
    template_name = 'registration/user_temporal.html'

    def get(self, request):  # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        if request.user.is_anonymous:
            try:
                user = User()
                user.create_temporal()

            except Exception as error:
                print(error)
                return redirect('index')

            else:
                login(request,user, backend = 'allauth.account.auth_backends.AuthenticationBackend')
                context = {"pass" : user.password, "username" : user.username}
                return render(request, self.template_name, context)
        else:
            return redirect('index')

class UserUpdateTemporalView(UpdateView):
    #Data Model
    model = User
    fields = ['username', 'first_name','last_name','email']

    #Get Data
    slug_url_kwarg = 'user_slug'
    slug_field = 'identifier'

    #Template
    template_name = 'registration/user_update.html'
    success_url = "/"
    def get_object(self, queryset=None):
        object = super(UserUpdateTemporalView, self).get_object(queryset)
        if not self.request.user or self.request.user.pk != object.pk:
            redirect("index")
        return object

    def form_valid(self, form):
        """If the form is valid, redirect to the supplied URL."""
        messages.success(self.request,"Datos actualizados correctamente")
        self.object = form.save()
        self.object.set_not_temporal()
        return super().form_valid(form)

class UserUpdateView(UpdateView):
    #Data Model
    model = User
    fields = ['username', 'first_name','last_name','email']

    #Get Data
    slug_url_kwarg = 'user_slug'
    slug_field = 'identifier'

    #Template
    template_name = 'registration/user_update.html'
    success_url = "/"
    def get_object(self, queryset=None):
        object = super(UserUpdateView, self).get_object(queryset)
        if not self.request.user or self.request.user.pk != object.pk:
            redirect("index")
        return object

    def form_valid(self, form):
        """If the form is valid, redirect to the supplied URL."""
        messages.success(self.request,"Datos actualizados correctamente")
        self.object = form.save()
        return super().form_valid(form)
