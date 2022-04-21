from django.shortcuts import render, redirect
from django.urls import reverse
import pandas as pd
from django.http import HttpResponse, Http404
from .forms import (GeneForm, GenesetForm, MirnasetForm, GenesetGMTForm)
from microrna.models import Mirnaset
from .models import  Geneset, Gene
from microrna.models import Target
from django.views.generic import (View, CreateView, ListView, DeleteView)
from django.views.generic.edit import FormView
from django.contrib import messages
from mirWeb.settings import DATA_DIR
from registration.models import User
from analysis.task import QueueSqlite, parse_file
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.db.models import Q
from django.http import HttpResponse, Http404, HttpResponseForbidden
import uuid
# Create your views here.

#### Genes ###

class GeneDetailView(View):
    """
    Main View of the page. We obtain all the information related with the Session thanks to
    the session identifier
    """
    template_name = 'gene/gene_view.html'

    def get(self, request,gene_id):  # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        try:
            gene = Gene.objects.get(symbol__iexact=gene_id)
        except Exception as error:
            print(error)
            try:
                gene = Gene.objects.get(entrez_id__iexact=gene_id)
            except Exception as error2:
                gene = False
                
        finally:
            if gene:
                context = {"gene" : gene}
                context["bGene"] = True
                #Target
                target = pd.DataFrame(Target.objects.filter(gene_id = gene, number_target__gte = 1).values("id" , "gene_id_id__symbol", "mirna_id_id__mature_name", "target", "number_target"))
                
                if not target.empty:
                    context["col"] = ["ID" , "Gene Symbol", "microRNA Mature ID", "Prediction Tools", "Number Prediction Tools"]
                    target.columns =  ["ID" , "Gene Symbol", "microRNA Mature ID", "Prediction Tools", "Number Prediction Tools"]
                    
                    #Get Json File
                    result = target.to_json(orient='values')
                    identifier = str(uuid.uuid4())
                    context["identifier"] = identifier
                    request.session[identifier]=result

                #Mirset
                geneset = pd.DataFrame(gene.geneset_set.all().values("external_id" , "name", "description","ref_link","genes_id__symbol"))
                if not geneset.empty:
                    print(geneset)
                    result = geneset.to_json(orient='values')
                    identifier = str(uuid.uuid4())
                    context["identifier2"] = identifier
                    request.session[identifier]=result
                    context["col2"] = ("external_id" , "name", "description","ref_link","genes_id__symbol")
            else:
                context = {"bGene" : False}
                context["gene"] = gene_id

        return render(request, self.template_name, context)


class GeneUploadView(UserPassesTestMixin, FormView):
    template_name = 'gene/create_gene.html'
    form_class = GeneForm

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        print(form.cleaned_data)
        file = form.cleaned_data["file"]
        df = parse_file(file)
        table = form.cleaned_data["table"]
        QueueSqlite(df,table)
        return redirect('index')

    def test_func(self):
        return self.request.user.is_staff


class CreateSetView(LoginRequiredMixin, CreateView):
    template_name = 'gene/create_gene.html'

    def get(self, request, user_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            user = User.objects.get(identifier=user_slug)
            print("##########################")
            print(request.META.get('PATH_INFO', ''))
            form = GenesetForm(user=request.user) if "geneset" in request.META.get('PATH_INFO', '') else MirnasetForm(user=request.user)

            context = {'form': form}

        except Exception as error:
            print(error)
            raise Http404('User not found...!')

        else:
            # We pase the session to the template with the Context Dyct
            context['user'] = user
            context['title'] = "Upload Set"

            return render(request, self.template_name, context)

        return render(request, self.template_name, context)

    def post(self, request, user_slug):
        # We get the filter condition
        form = GenesetForm(request.POST, request.FILES, user=request.user) if "geneset" in request.META.get('PATH_INFO', '') else MirnasetForm(request.POST, request.FILES, user=request.user)
        context = {'form': form}
        #context['user'] = user
        context['title'] = "Upload Geneset"

        if form.is_valid():
            # This method is called when valid form data has been POSTed.
            # It should return an HttpResponse.
            formDict = form.cleaned_data
            try:
                if "geneset" in request.META.get('HTTP_REFERER', ''):
                    object = Geneset()
                else:
                    object = Mirnaset()

                print("Create Geneset")
                object.user_id = User.objects.get(identifier=user_slug)
                print("Add User")
                object.public = formDict["public"]
                
                object.from_form(name = formDict["name"], description = formDict["description"],
                                        ref = formDict["ref_link"], lFeature = formDict["file"].read().decode('utf-8').split("\n"), identifier = formDict["format"],
                                        user_slug = user_slug, public=formDict["public"])

            except Exception as error:
                print(error)
                messages.warning(request, "We detected the following errors: %s"%error)
                return render(request, self.template_name, context)

        elif not form.is_valid():
            messages.warning(request, "We detected the following errors: %s"%form.errors)
            return render(request, self.template_name, context)

        return redirect('data_detail', user_slug=user_slug)



class CreateGeneSetFromGMTView(LoginRequiredMixin, CreateView):
    template_name = 'gene/create_gene.html'

    def get(self, request, user_slug):
        try:
            print("as")
            # Obtain the session from the DB with the session_slug (identifier)
            user = User.objects.get(identifier=user_slug)
            form = GenesetGMTForm(user=request.user)
            context = {'form': form}

        except Exception as error:
            print(error)
            raise Http404('User not found...!')

        else:
            # We pase the session to the template with the Context Dyct
            context['user'] = user
            context['title'] = "Upload Geneset"

            return render(request, self.template_name, context)

        return render(request, self.template_name, context)

    def post(self, request, user_slug):
        # We get the filter condition
        form = GenesetGMTForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            # This method is called when valid form data has been POSTed.
            # It should return an HttpResponse.
            formDict = form.cleaned_data
            try:
                gtm = formDict["file"].read().decode('utf-8').split("\n")
                for gs in gtm:

                    lGs = gs.split("\t")
                    name = lGs[0]
                    description = lGs[1]
                    lGene = lGs[2:]
                    print(lGene)
                    
                    GS = Geneset()
                    GS.from_form(name = name, description = description, ref = "#", \
                         lFeature = lGene, public = formDict["public"], identifier = formDict["geneFormat"], user_slug = user_slug)
                    print(GS)
            except Exception as error:
                print("We detected the following errors: %s"%error)
                messages.warning(request, "We detected the following errors: %s"%error)
                return render(request, self.template_name)

        elif not form.is_valid():
            messages.warning(request, "We detected the following errors: %s"%form.errors)
            return render(request, self.template_name)
            
        return redirect('data_detail', user_slug=user_slug)


class GenesetDeleteView(DeleteView, LoginRequiredMixin):
            
    # specify the model you want to use
    model = Geneset
     
    # can specify success url
    # url to redirect after successfully
    # deleting object
    def get_success_url(self):
        return reverse('data_detail',kwargs={"user_slug":self.request.user.get_identifier()})
    def dispatch(self, request, *args, **kwargs):
        # safety checks go here ex: is user allowed to delete?
        ## Get Session
        geneset = Geneset.objects.get(pk=kwargs['pk'])
        if not geneset.is_owner(request.user):
            return HttpResponseForbidden()
        else:
            handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
            return handler(request, *args, **kwargs)


class GenesetListView(ListView):
    #https://simpleisbetterthancomplex.com/tutorial/2016/08/03/how-to-paginate-with-django.html
    model = Geneset
    context_object_name = 'genesets'
    paginate_by = 5
    template_name = 'gene/geneset_list.html'  # Specify your own template name/location
    count = 0

    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(*args, **kwargs)
        context['query'] = self.request.GET.get('search')
        return context

    def get_queryset(self):
        request = self.request
        result = super(GenesetListView, self).get_queryset()
        query = self.request.GET.get('search')

        if query:
            print("Query")
            try:
                queryset = Geneset.objects.filter(Q(public=True) & Q(user_id=User.objects.get(username = "root"))\
                   & (Q(name__icontains=query)|Q(description__icontains=query))).order_by("name")
            except Exception as error:
                print(error)
                queryset = []
        else:
            print("Not")
            try:
                queryset = Geneset.objects.filter(Q(public=True, user_id=User.objects.get(username = "root"))).order_by("name")
            except Exception as error:
                print(error)
                queryset = []
        print(queryset)
        return queryset



class GenesetDetailView(View):
    """
    Main View of the page. We obtain all the information related with the Session thanks to
    the session identifier
    """
    template_name = 'gene/geneset_view.html'

    def get(self, request, geneset_id):  # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        try:
            geneset = Geneset.objects.get(name__iexact=geneset_id)
        except Exception as error:
            print(error)
            try:
                geneset = Geneset.objects.get(pk=geneset_id)
            except Exception as error2:
                geneset = False
                
        finally:
            if geneset and geneset.public:
                context = {"geneset" : geneset}
                context["bGeneset"] = True
                #Target
                #in_set = pd.DataFrame(Target.objects.filter(gene_id = gene, number_target__gte = 1).values("id" , "gene_id_id__symbol", "mirna_id_id__mature_name", "target", "number_target"))
                

            else:
                context = {"bGeneset" : False}
                context["geneset"] = geneset_id

        return render(request, self.template_name, context)
