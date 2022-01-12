from django.shortcuts import render, redirect
from django.urls import reverse_lazy,reverse

from django.http import HttpResponse, Http404
from .forms import (SessionSearchForm, WorkflowFilterForm,
                    GeneForm, GenesetForm, MirnasetForm, GenesetGMTForm, WorkflowCorrelationForm, WorkflowGenesetCorForm, AllCorrelationForm,
                    SyntheticLethalityForm, TargetPredictorForm,
                    DatasetForm, SessionCreateForm, 
                    WorkFlowClassificationForm, WorkFlowFeaturesForm, WorkFlowFeaturesRatioForm)

from .models import Mirnaset, Session, File, Workflow, Geneset, Dataset
from django.views.generic import (View, CreateView, ListView, DeleteView)
import os
from wsgiref.util import FileWrapper
from django.views.generic.edit import FormView
from django.contrib import messages
from mirWeb.settings import DATA_DIR
from registration.models import User
from .task import QueueSqlite, parse_file, QueueCorrelation, QueueFeature, QueueFeatureRatio, QueueSurvivalFeature, QueueSurvivalFeatureRatio, QueueClassification
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin, PermissionRequiredMixin
from django.db.models import Q
import io
from django.core.paginator import Paginator
import pandas as pd 
from django.http import JsonResponse, HttpResponse, Http404, HttpResponseForbidden
import json
import uuid
import logging

# Create a logger for this file
logger = logging.getLogger(__file__)
#############
## Basic  ###
#############

def index(request):
    """
    Index view
    """
    return render(request, 'analysis/index.html')

def privacy_de(request):
    """
    Daten view
    """
    return render(request, 'registration/privacy_de.html')

def privacy_en(request):
    """
    Daten view
    """
    return render(request, 'registration/privacy_en.html')

def impressum(request):
    """
    Daten view
    """
    return render(request, 'registration/Impressum.html')

##############
## Session  ##
##############

class SessionIndexView(CreateView):
    template_name = 'analysis/session_index.html'

    def get(self, request):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            form = SessionSearchForm(user=request.user)
            context = {'form': form}
            # print(request.user.id)

        except Exception as error:
            raise Http404(error)

        else:
            # We pase the session to the template with the Context Dyct
            return render(request, self.template_name, context)
        return render(request, self.template_name, context)

    def post(self, request):
        # We get the filter condition
        form = SessionSearchForm(request.POST, request.FILES,user=request.user)
        if form.is_valid():
            # We add the workflow id and all the filter dict in the session cache
            return redirect('session_detail', session_slug=form.cleaned_data['session'])
        return render(request, self.template_name, {'form': form})


class SessionCreateView(LoginRequiredMixin, CreateView):
    template_name = 'analysis/session_create.html'

    def get(self, request):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            form = SessionCreateForm()
            context = {'form': form}
            context["title"] = "Create New Session"

        except:
            raise Http404('Session not found...!')

        else:
            # We pase the session to the template with the Context Dyct
            return render(request, self.template_name, context)
        return render(request, self.template_name, context)

    def post(self, request):
        # We get the filter condition
        form = SessionCreateForm(request.POST, request.FILES, request.user)
        if form.is_valid():
            # We add the workflow id and all the filter dict in the session cache
            post = form.save(commit=False)
            post.user_id = User.objects.get(identifier=request.user.identifier)
            post.save()
            return redirect('session_detail', session_slug=post.identifier)

        return render(request, self.template_name, {'form': form})


class SessionDetailView(View):
    """
    Main View of the page. We obtain all the information related with the Session thanks to
    the session identifier
    """
    template_name = 'analysis/session_detail.html'

    def get(self, request, session_slug):  # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'

        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)


        except:
            raise Http404('Session not found...!')

        else:
            #Check permissions
            session.check_permissions(request.user)
     
            #Get Workflows
            wrkfl = session.get_workflows()

            #Correlation
            wrkfl_cor = wrkfl.filter(Q(analysis="Correlation")|(Q(analysis="GeneSetScore"))).order_by("analysis","label")
            p = Paginator(wrkfl_cor, 5)  # creating a paginator object 
            # We pase the session to the template with the Context Dyct
            page_number = request.GET.get('page1')
            try:
                page_cor = p.get_page(page_number)  # returns the desired page object
            except PageNotAnInteger:
                # if page_number is not an integer then assign the first page
                page_cor = p.page(1)
            except EmptyPage:
                # if page is empty then return last page
                page_cor = p.page(p.num_pages)


            #Survival
            wrkfl_surv = wrkfl.filter(Q(analysis="Survival")).order_by("analysis","label")
            p = Paginator(wrkfl_surv, 5)  # creating a paginator object 
            # We pase the session to the template with the Context Dyct
            page_number = request.GET.get('page2')
            try:
                page_surv = p.get_page(page_number)  # returns the desired page object
            except PageNotAnInteger:
                # if page_number is not an integer then assign the first page
                page_surv = p.page(1)
            except EmptyPage:
                # if page is empty then return last page
                page_surv = p.page(p.num_pages)

            #Clasification 
            wrkfl_feat = wrkfl.filter(Q(analysis="Feature")|Q(analysis="Classification")).order_by("analysis","label")
            p = Paginator(wrkfl_feat, 5)  # creating a paginator object 
            # We pase the session to the template with the Context Dyct
            page_number = request.GET.get('page3')
            try:
                page_feat = p.get_page(page_number)  # returns the desired page object
            except PageNotAnInteger:
                # if page_number is not an integer then assign the first page
                page_feat = p.page(1)
            except EmptyPage:
                # if page is empty then return last page
                page_feat = p.page(p.num_pages)

            context = {'session_detail': session}
            context['page_cor'] = page_cor
            context['page_surv'] = page_surv
            context['page_feat'] = page_feat

            return render(request, self.template_name, context)

        return redirect('analysis:session_detail', session_slug)

class SessionDeleteView(DeleteView, LoginRequiredMixin):
            
    # specify the model you want to use
    model = Session
     
    # can specify success url
    # url to redirect after successfully
    # deleting object
    success_url ="/"

    def dispatch(self, request, *args, **kwargs):
        # safety checks go here ex: is user allowed to delete?
        ## Get Session
        session = Session.objects.get(pk=kwargs['pk'])
        if not session.is_owner(request.user):
            return HttpResponseForbidden()
        else:
            handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
            return handler(request, *args, **kwargs)



#############
## Files  ###
#############
# Files


class FileDeletView(DeleteView, LoginRequiredMixin):
            
    # specify the model you want to use
    model = File
     
    # can specify success url
    # url to redirect after successfully
    # deleting object
    success_url ="/"

    def dispatch(self, request, *args, **kwargs):
        # safety checks go here ex: is user allowed to delete?
        ## Get Session
        file = File.objects.get(pk=kwargs['pk'])
        if not file.is_owner(request.user):
            return HttpResponseForbidden()
        else:
            handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
            return handler(request, *args, **kwargs)


def DataDownload(request, session_slug, pk_file):
    # print(f'\n Data Downlod called')
    file = File.objects.get(id=pk_file)
    file_path = DATA_DIR+"/"+file.data.url
    print(file.data.name)
    # return session_data

    file_wrapper = FileWrapper(open(file_path, 'rb'))
    response = HttpResponse(file_wrapper)
    response['X-Sendfile'] = file_path
    response['Content-Length'] = os.stat(file_path).st_size
    response['Content-Disposition'] = 'attachment; filename=%s.csv' % file.label
    return response

def dataset_download(request, pk, file_name):
        data = Dataset.objects.get(id=pk)
        if not data.is_owner(request.user) and not data.public:
            return HttpResponseForbidden()
        else:
            pass
        return data.file_download(file_name)

def HtmlDownload(request, session_slug, pk_wrkl):
    wrfkl = Workflow.objects.get(pk=pk_wrkl) #Get Workflow

    # print(f'\n Data Downlod called')
    html_path = "%s/%s/Sessions/%s/file/%s.html"%(str(DATA_DIR),str(wrfkl.sesion_id.user_id.identifier), \
        wrfkl.sesion_id.identifier,wrfkl.label)

    
    # return session_data
    file_wrapper = FileWrapper(open(html_path, 'rb'))
    response = HttpResponse(file_wrapper)
    response['X-Sendfile'] = html_path
    response['Content-Length'] = os.stat(html_path).st_size
    response['Content-Disposition'] = 'attachment; filename=%s' % html_path.split("/")[-1]
    return response

class DataDetailView(LoginRequiredMixin, View):
    """
    Main View of the page. We obtain all the information related with the Session thanks to
    the session identifier
    """
    template_name = 'analysis/data_detail.html'

    def get(self, request, user_slug):  # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            user = request.user

        except Exception as error:
            print(error)
            raise Http404('User not found...!')

        else:
            #Datasets
            datasets = user.get_dataset()
            p = Paginator(datasets, 5)  # creating a paginator object 
            # We pase the session to the template with the Context Dyct
            page_number = request.GET.get('page1')
            try:
                page_dataset = p.get_page(page_number)  # returns the desired page object
            except PageNotAnInteger:
                # if page_number is not an integer then assign the first page
                page_geneset = p.page(1)
            except EmptyPage:
                # if page is empty then return last page
                page_geneset = p.page(p.num_pages)


            #Genesets
            genesets = user.get_geneset()
            p = Paginator(genesets, 5)  # creating a paginator object 
            # We pase the session to the template with the Context Dyct
            page_number = request.GET.get('page2')
            try:
                page_geneset = p.get_page(page_number)  # returns the desired page object
            except PageNotAnInteger:
                # if page_number is not an integer then assign the first page
                page_geneset = p.page(1)
            except EmptyPage:
                # if page is empty then return last page
                page_geneset = p.page(p.num_pages)

            #Clasification 
            mirsets = user.get_mirset()
            p = Paginator(mirsets, 5)  # creating a paginator object 
            # We pase the session to the template with the Context Dyct
            page_number = request.GET.get('page3')
            try:
                page_mirset = p.get_page(page_number)  # returns the desired page object
            except PageNotAnInteger:
                # if page_number is not an integer then assign the first page
                page_mirset = p.page(1)
            except EmptyPage:
                # if page is empty then return last page
                page_mirset = p.page(p.num_pages)

            context = {'user': user}
            context['page_dataset'] = page_dataset
            context['page_geneset'] = page_geneset
            context['page_mirset'] = page_mirset

            return render(request, self.template_name, context)

def getfiles(request):
    import os
    import zipfile
    from io import BytesIO
    from django.http import HttpResponse

    # Files (local path) to put in the .zip
    # FIXME: Change this (get paths from DB etc)
    dir_user = "%s/%s"%(DATA_DIR, request.user.get_identifier())
    result = [ os.path.join(dp,file) for dp, dn, filenames in os.walk(dir_user) for file in filenames]
    # Folder name in ZIP archive which contains the above files
    # E.g [thearchive.zip]/somefiles/file2.txt
    # FIXME: Set this to something better
    zip_subdir = "%s"%request.user.get_identifier()
    zip_filename = "%s.zip" % request.user.get_identifier()
    # Open StringIO to grab in-memory ZIP contents
    s = BytesIO()

     # The zip compressor
    zf = zipfile.ZipFile(s, "w")
    print(result)
    for fpath in result:
        print(fpath)
        # Calculate path for file in zip
        #fdir, fname = os.path.split(fpath)
        #d_rel = fdir.replace(DATA_DIR,"")
        zip_path = os.path.join(zip_subdir, fpath)

        # Add file, at correct path
        zf.write(fpath, zip_path)

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    response = HttpResponse(s.getvalue())
    # ..and correct content-disposition
    response['Content-Disposition'] = 'attachment; filename=%s' % zip_filename

    return response


###############
## Dataset  ###
###############

class DatasetDeleteView(DeleteView, LoginRequiredMixin):
            
    # specify the model you want to use
    model = Dataset
     
    # can specify success url
    # url to redirect after successfully
    # deleting object

    def get_success_url(self):
        return reverse('data_detail',kwargs={"user_slug":self.request.user.get_identifier()})

    def dispatch(self, request, *args, **kwargs):
        # safety checks go here ex: is user allowed to delete?
        ## Get Session
        workflow = Dataset.objects.get(pk=kwargs['pk'])
        if not workflow.is_owner(request.user):
            return HttpResponseForbidden()
        else:
            handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
            return handler(request, *args, **kwargs)

class CreateDatasetView(LoginRequiredMixin, CreateView):
    template_name = 'analysis/create_gene.html'

    def get(self, request, user_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            user = User.objects.get(identifier=user_slug)
            form = DatasetForm(user=request.user)
            context = {'form': form}

        except:
            raise Http404('Session not found...!')

        else:
            # We pase the session to the template with the Context Dyct
            context['user'] = user
            context['title'] = "Upload Dataset"

            return render(request, self.template_name, context)

    def post(self, request, user_slug):
        # We get the filter condition
        form = DatasetForm(request.POST, request.FILES, user=request.user)
        context = {"form":form}
        if form.is_valid():
            # This method is called when valid form data has been POSTed.
            # It should return an HttpResponse.
            dForm = form.cleaned_data

            if dForm["geneFile"] is not None and dForm["mirFile"] is not None:
                try:
                    data = Dataset()
                    data.user_id = User.objects.get(identifier=user_slug)
                    data.name = dForm["name"]
                    data.geneFile = dForm["geneFile"]
                    data.mirFile = dForm["mirFile"]
                    data.metadataFile = dForm["clinicalFile"]
                    data.public = dForm["public"]
                    data.technology = dForm["technology"]
                    data.sesion_id = User.objects.get(identifier=user_slug)
                    data.save()

                    #Get fields
                    data.set_number_gene()
                    data.set_number_mir()
                    data.set_number_sample()
                    data.set_metadata_fields()

                    if not data.check_dataset():
                        messages.warning(request,"Error creating Dataset, samples, genes or mir are 0")
                        data.delete()
                        return render(request, self.template_name, context)
                except Exception as error:
                    
                    messages.warning(request,error)
                    return render(request, self.template_name, context)
                    
                else:
                    return redirect('data_detail',user_slug=user_slug)

        elif not form.is_valid():
            print(form.errors)
            messages.warning(request,form.errors)
            return render(request, self.template_name, context)

        return redirect('data_detail', user_slug=user_slug)



class DatasetListView(ListView):
    #https://simpleisbetterthancomplex.com/tutorial/2016/08/03/how-to-paginate-with-django.html
    model = Dataset
    context_object_name = 'datasets'
    paginate_by = 5
    template_name = 'analysis/dataset_list.html'  # Specify your own template name/location

    def get_queryset(self):
        result = super(DatasetListView, self).get_queryset()
        query = self.request.GET.get('search')

        if query:
            print("Query")
            try:
                queryset = Dataset.objects.filter(Q(public=True) & Q(user_id=User.objects.get(username = "root"))\
                    & (Q(name__icontains=query))).order_by("name")
            except Exception as error:
                print(error)
                queryset = None
        else:

            try:
                queryset = Dataset.objects.filter(Q(public=True, user_id=User.objects.get(username = "root"))).order_by("name")
            except:
                queryset = None

        return queryset





#### Feature Results #####

def DatasetListJson(request):
    """
    Function that transform a DF in JsonResponse
    View list to populate the Html Table.
    The Datatable work with Ajax. We need a Json file.
    """

    #Read DF

    #Tranform DF to Json
    result = request.session["json"]
    json_dict = {}
    json_dict["data"] = json.loads(result)
    
    return JsonResponse(json_dict, safe = False)


class DatasetDetailView(View):
    """
    View to plot the result of the filter DF
    """
    template_name = 'results/dataset_view.html'
    
    def get(self,request, pk):

        # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        #Create Dictionary
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            dataset = Dataset.objects.get(pk=pk)

            if not dataset.is_owner(request.user) and not dataset.public:
                return HttpResponseForbidden()
            else:
                pass

            context = {'dataset': dataset}

        except Exception as error:
            print(error)

        else:
            # We pase the session to the template with the Context Dyct

            #Get File


            #Get Df
            metadata =  pd.read_csv(dataset.get_metadatapath(), index_col=0)
            metadata = metadata.reset_index()
            #Get Json File
            result = metadata.to_json(orient='values')
            identifier = str(uuid.uuid4())
            context["identifier"] = identifier
            request.session[identifier]=result
            context["col"] = metadata.columns.tolist()

            return render(request, self.template_name, context)




################
## Workflow  ###
################


class WorkflowDeleteView(DeleteView, LoginRequiredMixin):
            
    # specify the model you want to use
    model = Workflow
     
    # can specify success url
    # url to redirect after successfully
    # deleting object
    def get_success_url(self):
        return reverse('session_detail',kwargs={"session_slug":str(self.object.get_session_identifier())})

    def dispatch(self, request, *args, **kwargs):
        # safety checks go here ex: is user allowed to delete?
        ## Get Session
        workflow = Workflow.objects.get(pk=kwargs['pk'])
        if not workflow.is_owner(request.user):
            return HttpResponseForbidden()
        else:
            handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
            return handler(request, *args, **kwargs)


class CorrelationWorkflow(LoginRequiredMixin, CreateView):
    template_name = 'analysis/workflow_cor_create.html'

    def get(self, request, session_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            form = WorkflowCorrelationForm(user=request.user)
            context = {'form': form}
            context["title"] = "miRNA/Gene Correlation Analysis"

        except:
            raise Http404('Session not found...!')

        else:
            # We pase the session to the template with the Context Dyct
            bPer = session.check_permissions_analysis(request.user)
            if not bPer:
                print("as")
                return render(request,'error/NotOwner.html', context)
            context['session_detail'] = session
            return render(request, self.template_name, context)

        return render(request, self.template_name, context)


    def post(self, request, session_slug):
        # We get the filter condition
        form = WorkflowCorrelationForm(request.POST, request.FILES,user=request.user)
        if form.is_valid():
            print(form.cleaned_data)
            dForm = form.cleaned_data
            dForm["analysis"] = "Correlation"
            dForm["analysis_type"] = "microRNA/Gene Corr."

            try:
                wrkf = Workflow()
                wrkf.assign_workflow(dForm, session_slug)

                QueueCorrelation(
                    wrkf, method=dForm["analysis"], FilterChoice=dForm["filter"], normal=dForm["normal"], \
                    filter_sample = dForm["filter_sample"], group_sample = dForm["group_sample"], filter_group = dForm["filter_group"], \
                    logfc = dForm["logfc"], pval = dForm["pval"], survival = dForm["survival"], group = dForm["group"])

            except Exception as error:
                print(error)

            finally:
                return redirect('session_detail', session_slug=session_slug)
        else:
            print("Algo ha ido mal")
            print(form.errors)

        return render(request, self.template_name, {'form': form})


class GeneSetScoreWorkflow(LoginRequiredMixin, CreateView):
    template_name = 'analysis/workflow_gs_create.html'

    def get(self, request, session_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            form = WorkflowGenesetCorForm(user=request.user)
            context = {'form': form}
            context["title"] = "miRNA/GeneSet Correlation Analysis"

        except:
            raise Http404('Session not found...!')

        else:
            bPer = session.check_permissions_analysis(request.user)
            if not bPer:
                print("as")
                return render(request,'error/NotOwner.html', context)

            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            return render(request, self.template_name, context)

    def post(self, request, session_slug):
        # We get the filter condition
        form = WorkflowGenesetCorForm(request.POST, request.FILES,user=request.user)
        if form.is_valid():
            print(form.cleaned_data)
            dForm = form.cleaned_data
            dForm["analysis"] = "GeneSetScore"
            dForm["analysis_type"] = "microRNA/Geneset Corr."

            try:
                wrkf = Workflow()
                wrkf.assign_workflow(dForm, session_slug)
                QueueCorrelation(wrkf, method=dForm["analysis"], normal=dForm["normal"],\
                    filter_sample = dForm["filter_sample"], group_sample = dForm["group_sample"], filter_group = dForm["filter_group"])
            except Exception as error:
                print(error)

            finally:
                return redirect('session_detail', session_slug=session_slug)
        else:
            print("Algo ha ido mal")
            print(form.errors)
redirect

class FeatureWorkflow(LoginRequiredMixin, CreateView):
    template_name = 'analysis/workflow_feature.html'

    def get(self, request, session_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            form = WorkFlowFeaturesForm(user=request.user)
            context = {'form': form}
            context["title"] = "Feature selection for classification"

        except Exception as error:
            raise Http404('Session not found...! \n %s'%(error))

        else:
            bPer = session.check_permissions_analysis(request.user)
            if not bPer:
                print("as")
                return render(request,'error/NotOwner.html', context)
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            return render(request, self.template_name, context)

    def post(self, request, session_slug):
        # We get the filter condition
        form = WorkFlowFeaturesForm(request.POST, request.FILES,user=request.user)
        if form.is_valid():
            print(form.cleaned_data)
            dForm = form.cleaned_data
            dForm["analysis"] = "Feature"
            dForm["analysis_type"] = "Classification Feature Selection"

            try:
                print("Enter")
                wrkf = Workflow()
                wrkf.assign_workflow(dForm, session_slug)
                wrkf.set_feature(dForm["feature"])
                QueueFeature(
                    wrkf, topk = dForm["topk"],k = dForm["k"], group=dForm["group"], feature = dForm["feature"],filter_sample = dForm["filter_sample"], group_sample = dForm["group_sample"], filter_group = dForm["filter_group"])
            except Exception as error:
                print("Error: %s"%error)

            finally:
                return redirect('session_detail', session_slug=session_slug)
        else:
            print("Algo ha ido mal")
            print(form.errors)


class ClassificationWorkflow(LoginRequiredMixin, CreateView):
    template_name = 'analysis/workflow_classification.html'

    def get(self, request, session_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            form = WorkFlowClassificationForm(user=request.user)
            context = {'form': form}
            context["title"] = "Classification Analysis"

        except Exception as error:
            raise Http404('Session not found...! \n %s'%(error))

        else:
            bPer = session.check_permissions_analysis(request.user)
            if not bPer:
                print("as")
                return render(request,'error/NotOwner.html', context)            
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            return render(request, self.template_name, context)

    def post(self, request, session_slug):
        # We get the filter condition
        form = WorkFlowClassificationForm(request.POST, request.FILES,user=request.user)
        context = {"form" : form}
        if form.is_valid():
            print(form.cleaned_data)
            dForm = form.cleaned_data
            dForm["analysis"] = "Classification"
            dForm["analysis_type"] = "Classification Analysis"

            context["title"] = "Classification Analysis"

            if len(dForm["publicGeneset"]) == 0 and len(dForm["publicMirnaset"]) == 0 and not dForm["use_fit_model"]:
                print("Algo ha ido mal")
                messages.warning(request, "Select almost one geneset or mirnaset")
                return render(request, self.template_name, context)
            try:
                print("Enter")
                wrkf = Workflow()
                wrkf.assign_workflow(dForm, session_slug)
                print(dForm)
                QueueClassification(wrkf, 
                 k = dForm["k"], model = dForm["model"], group=dForm["group"], use_fit_model = dForm["use_fit_model"], pk = dForm["publicModel"])

            except Exception as error:
                print("Error: %s"%error)

            finally:
                return redirect('session_detail', session_slug=session_slug)
        else:
            print("Algo ha ido mal")
            messages.warning(request, form.errors)
            return render(request, self.template_name, context)



class FeatureRatioWorkflow(LoginRequiredMixin, CreateView):
    template_name = 'analysis/workflow_feature_ratio.html'

    def get(self, request, session_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            form = WorkFlowFeaturesRatioForm(user=request.user)
            context = {'form': form}
            context["title"] = "Feature selection for classification"

        except Exception as error:
            raise Http404('Session not found...! \n %s'%(error))

        else:
            bPer = session.check_permissions_analysis(request.user)
            if not bPer:
                print("as")
                return render(request,'error/NotOwner.html', context)            
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            return render(request, self.template_name, context)

    def post(self, request, session_slug):
        # We get the filter condition
        form = WorkFlowFeaturesRatioForm(request.POST, request.FILES,user=request.user)
        if form.is_valid():
            print(form.cleaned_data)
            dForm = form.cleaned_data
            dForm["analysis"] = "Feature"
            dForm["analysis_type"] = "miRNA/Gene Feature Selection"

            try:
                print("Enter")
                wrkf = Workflow()
                wrkf.assign_workflow(dForm, session_slug)
                QueueFeatureRatio(
                    wrkf, topk = dForm["topk"],k = dForm["k"], group=dForm["group"],  filter_sample = dForm["filter_sample"], \
                         group_sample = dForm["group_sample"], filter_group = dForm["filter_group"], filter_pair = dForm["filter_pair"],low_coef = dForm["low_coef"], min_db = dForm["min_db"])
            except Exception as error:
                print("Error: %s"%error)

            finally:
                return redirect('session_detail', session_slug=session_slug)
        else:
            print("Algo ha ido mal")
            print(form.errors)



class SurvivalFeatureWorkflow(LoginRequiredMixin, CreateView):
    template_name = 'analysis/workflow_feature.html'

    def get(self, request, session_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            form = WorkFlowFeaturesForm(user=request.user)
            context = {'form': form}
            context["title"] = "Feature selection for Survival Analysis"

        except Exception as error:
            raise Http404('Session not found...! \n %s'%(error))

        else:
            bPer = session.check_permissions_analysis(request.user)
            if not bPer:
                print("as")
                return render(request,'error/NotOwner.html', context)            
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            return render(request, self.template_name, context)

    def post(self, request, session_slug):
        # We get the filter condition
        form = WorkFlowFeaturesForm(request.POST, request.FILES,user=request.user)
        if form.is_valid():
            print(form.cleaned_data)
            dForm = form.cleaned_data
            dForm["analysis"] = "Survival"
            dForm["analysis_type"] = "Survival Feature Selection"

            try:
                print("Enter")
                wrkf = Workflow()
                wrkf.assign_workflow(dForm, session_slug)
                wrkf.set_feature(dForm["feature"])
                QueueSurvivalFeature(
                    wrkf, topk = dForm["topk"],k = dForm["k"], group=dForm["group"],feature = dForm["feature"], filter_sample = dForm["filter_sample"], group_sample = dForm["group_sample"], filter_group = dForm["filter_group"])
            except Exception as error:
                print("Error: %s"%error)

            finally:
                return redirect('session_detail', session_slug=session_slug)
        else:
            print("Algo ha ido mal")
            print(form.errors)


class SurvivalFeatureRatioWorkflow(LoginRequiredMixin, CreateView):
    template_name = 'analysis/workflow_feature_ratio.html'

    def get(self, request, session_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            form = WorkFlowFeaturesRatioForm(user=request.user)
            context = {'form': form}
            context["title"] = "Feature selection for Survival Analysis"

        except Exception as error:
            raise Http404('Session not found...! \n %s'%(error))

        else:
            bPer = session.check_permissions_analysis(request.user)
            if not bPer:
                print("as")
                return render(request,'error/NotOwner.html', context)            
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            return render(request, self.template_name, context)

    def post(self, request, session_slug):
        # We get the filter condition
        form = WorkFlowFeaturesRatioForm(request.POST, request.FILES,user=request.user)
        if form.is_valid():
            print(form.cleaned_data)
            dForm = form.cleaned_data
            dForm["analysis"] = "Survival"
            dForm["analysis_type"] = "Survival mirna/Gene Ratio Feature Selection"

            try:
                print("Enter")
                wrkf = Workflow()
                wrkf.assign_workflow(dForm, session_slug)
                QueueSurvivalFeatureRatio(
                    wrkf, topk = dForm["topk"],k = dForm["k"], group=dForm["group"], filter_sample = dForm["filter_sample"], \
                         group_sample = dForm["group_sample"], filter_group = dForm["filter_group"], filter_pair = dForm["filter_pair"],low_coef = dForm["low_coef"], min_db = dForm["min_db"])
            except Exception as error:
                print("Error: %s"%error)

            finally:
                return redirect('session_detail', session_slug=session_slug)
        else:
            print("Algo ha ido mal")
            print(form.errors)


class WorkflowFilterView(CreateView):
    template_name = 'analysis/workflow_filter.html'

    def get(self, request, session_slug, pk):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            wrkfl = Workflow.objects.get(pk=pk)
            print(session)
            form = WorkflowFilterForm()
            context = {'form': form}
            context["title"] = "Filter Table Correlation"
            context["workflow"] = wrkfl
        except:
            raise Http404('Session not found...!')

        else:
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            return render(request, self.template_name, context)


    def post(self, request, session_slug, pk):
        # We get the filter condition
        form = WorkflowFilterForm(request.POST)
        print(request.POST)
        if form.is_valid():
            # We add the workflow id and all the filter dict in the session cache
            form.cleaned_data["pk"] = pk
            print(form.cleaned_data)
            request.session["filter_dict"] = form.cleaned_data

            # Redirect the user to results view
            return redirect('results_detail', session_slug=session_slug, pk = pk)
        else:
            print(form.errors)
        return render(request, self.template_name, {'form': form})


class SyntheticLethalityWorkflow(CreateView):
    template_name = 'analysis/workflow_synthetic.html'

    def get(self, request, session_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            form = SyntheticLethalityForm(session=session, user = request.user)
            context = {'form': form}
            context["title"] = "Get miRNA or Gene targeting"

        except:
            raise Http404('Session not found...!')

        else:
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            return render(request, self.template_name, context)

        return render(request, self.template_name, context)

    def post(self, request, session_slug):
        # Obtain the session from the DB with the session_slug (identifier)
        session = Session.objects.get(identifier=session_slug)
        context = {"title":"SL"}
        context['session_detail'] = session

        # We get the filter condition
        form = SyntheticLethalityForm(request.POST, request.FILES, user = request.user, session=session)
        context["form"] = form
        if form.is_valid():
            formDict = form.cleaned_data
            print(formDict)
            if formDict["use_set"] and formDict["publicGeneset"] == []:
                messages.warning(request=request, message = "Please select almost one Geneset")
                return render(request, self.template_name, context)
            elif not formDict["use_set"] and formDict["tQuery"] == "":
                messages.warning(request=request, message = "Please introduce one Gene or microRNA")
                return render(request, self.template_name, context)

            else:
                # We add the workflow id and all the filter dict in the session cache
                request.session["filter_dict"] = form.cleaned_data

            # Redirect the user to results view
            return redirect('results_synthetic', session_slug=session_slug)
            
        else:
            print(form.errors)
            messages.warning(request=request, message = str(form.errors))
            return render(request, self.template_name, context)


class TargtePredictorWorkflow(CreateView):
    template_name = 'analysis/workflow_target.html'

    def get(self, request, session_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            form = TargetPredictorForm(session=session, user = request.user)
            context = {'form': form}
            context["title"] = "Get miRNA or Gene targeting"

        except:
            raise Http404('Session not found...!')

        else:
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            return render(request, self.template_name, context)

        return render(request, self.template_name, context)

    def post(self, request, session_slug):
        # Obtain the session from the DB with the session_slug (identifier)
        session = Session.objects.get(identifier=session_slug)
        context = {"title":"Get miRNA or Gene targeting"}
        context['session_detail'] = session

        # We get the filter condition
        form = TargetPredictorForm(request.POST, request.FILES, user = request.user, session=session)
        context["form"] = form
        if form.is_valid():
            formDict = form.cleaned_data
            print(formDict)
            if formDict["use_set"] and formDict["publicGeneset"] == [] and formDict["publicMirnaset"] == []:
                messages.warning(request=request, message = "Please select almost one Geneset or microRNAset")
                return render(request, self.template_name, context)
            elif not formDict["use_set"] and formDict["tQuery"] == "":
                messages.warning(request=request, message = "Please introduce one Gene or microRNA")
                return render(request, self.template_name, context)

            else:
                # We add the workflow id and all the filter dict in the session cache
                request.session["filter_dict"] = form.cleaned_data

            # Redirect the user to results view
            return redirect('results_target', session_slug=session_slug)
            
        else:
            print(form.errors)
            messages.warning(request=request, message = str(form.errors))
            return render(request, self.template_name, context)


#### Genes ###
class GeneUploadView(UserPassesTestMixin, FormView):
    template_name = 'analysis/create_gene.html'
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
    template_name = 'analysis/create_gene.html'

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
    template_name = 'analysis/create_gene.html'

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


class MirnasetDeleteView(DeleteView, LoginRequiredMixin):
            
    # specify the model you want to use
    model = Mirnaset
     
    # can specify success url
    # url to redirect after successfully
    # deleting object
    def get_success_url(self):
        return reverse('data_detail',kwargs={"user_slug":self.request.user.get_identifier()})

    def dispatch(self, request, *args, **kwargs):
        # safety checks go here ex: is user allowed to delete?
        ## Get Session
        mirset = Mirnaset.objects.get(pk=kwargs['pk'])
        if not mirset.is_owner(request.user):
            return HttpResponseForbidden()
        else:
            handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
            return handler(request, *args, **kwargs)

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

def DownloadSet(request, pk, identifier, set_type):
    # some code
    if set_type == "mirset":
        object = Mirnaset.objects.get(pk = pk)
    else:
        object = Geneset.objects.get(pk = pk)
    file_data = object.to_txt(identifier)
    response = HttpResponse(file_data, content_type='application/text charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="%s_%s.txt"'%(set_type, str(object.name))
    return response

def DownloadAllGMT(request, identifier, set_type):
   # some code
    if set_type == "mirset":
        query = Mirnaset.objects.filter(public=True, user_id=User.objects.get(username = "root")).order_by("name")
    else:
        query = Geneset.objects.filter(public=True, user_id=User.objects.get(username = "root")).order_by("name")

    lSet = []
    for object in query:
        lSet.append(object.to_gmt(identifier))
    file_data = "\n".join(lSet)
    response = HttpResponse(file_data, content_type='application/text charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="mio_%s_db.gmt"'%set_type
    return response



def FeatureToSet(request, session_slug, pk):
    #Import objects
    session = Session.objects.get(identifier=session_slug)

    #Check permissions
    session.check_permissions(request.user)

    #Get Set
    try:
        wrkfl = Workflow.objects.get(pk=pk)
        fileTop = File.objects.get(workflow_id = pk, type = "Topfeature")

    except Exception as error:
            print(error)

    else:
        #Get Df
        top =  pd.read_csv(fileTop.get_path(), index_col=0)
        lFeature = top.index.tolist()
        try:
            if wrkfl.feature_type == "gene":
                object = Geneset()
                identifier = "symbol"
            else:
                object = Mirnaset()
                identifier = "id"

            object.from_form(name = wrkfl.label, description = wrkfl.label,
                                        ref = "MIO Feature Selection Analysis", lFeature = lFeature, identifier = identifier,
                                        user_slug = request.user.identifier, public=False)
        except Exception as error:
            print(error)
            pass
    finally:
        return redirect('session_detail', session_slug=session_slug)



def SaveModel(request, session_slug, pk):
    import pickle
    #Get Set
    #Import
    session = Session.objects.get(identifier=session_slug)
    wrfkl = Workflow.objects.get(pk=pk) #Get Workflow
    results = File.objects.get(workflow_id = pk, type = "Pickle")
    #Check permissions
    session.check_permissions(request.user)



    results = pickle.load( open(results.get_path(), "rb" ) )            
    score = results["test"]
    models = results["classifier"]
    index = score.index(max(score))
    best_model = models[index] 
    name = results["model"]+"_"+wrfkl.label
    File().set_pickle(workflow = wrfkl, file = best_model, ftype = "Pickle", is_result = False, description = "Models", label = name)
    messages.warning(request,"Model Save")
    return redirect('results_classification', session_slug=session_slug, pk=pk)



class GenesetListView(ListView):
    #https://simpleisbetterthancomplex.com/tutorial/2016/08/03/how-to-paginate-with-django.html
    model = Geneset
    context_object_name = 'genesets'
    paginate_by = 5
    template_name = 'analysis/geneset_list.html'  # Specify your own template name/location

    def get_queryset(self):
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
            try:
                queryset = Geneset.objects.filter(Q(public=True, user_id=User.objects.get(username = "root"))).order_by("name")
            except:
                queryset = []

        return queryset



class MirnasetListView(ListView):
    #https://simpleisbetterthancomplex.com/tutorial/2016/08/03/how-to-paginate-with-django.html
    model = Mirnaset
    context_object_name = 'mirnasets'
    paginate_by = 5
    template_name = 'analysis/mirset_list.html'  # Specify your own template name/location

    def get_queryset(self):
        result = super(MirnasetListView, self).get_queryset()
        query = self.request.GET.get('search')

        if query:
            print("Query")
            try:
                queryset = Mirnaset.objects.filter(Q(public=True) & Q(user_id=User.objects.get(username = "root"))\
                    & (Q(name__icontains=query)|Q(description__icontains=query))).order_by("name")
            except Exception as error:
                print(error)
                queryset = []
        else:
            try:
                queryset = Mirnaset.objects.filter(Q(public=True, user_id=User.objects.get(username = "root"))).order_by("name")
            except:
                queryset = []

        return queryset

#################
#### ADMIN ######
#################


class AllCorrelationWorkflow(LoginRequiredMixin, CreateView):
    template_name = 'analysis/create_gene.html'

    def get(self, request, session_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            form = AllCorrelationForm(user=request.user)
            context = {'form': form}
            context["title"] = "miRNA/Gene Correlation Analysis"

        except:
            raise Http404('Session not found...!')

        else:
            # We pase the session to the template with the Context Dyct
            bPer = session.check_permissions_analysis(request.user)
            if not bPer:
                print("as")
                return render(request,'error/NotOwner.html', context)
            context['session_detail'] = session
            return render(request, self.template_name, context)

        return render(request, self.template_name, context)


    def post(self, request, session_slug):
        # We get the filter condition
        form = AllCorrelationForm(request.POST, request.FILES,user=request.user)
        if form.is_valid():
            print(form.cleaned_data)
            dForm = form.cleaned_data
            dForm["analysis"] = "Correlation"
            dForm["analysis_type"] = "microRNA/Gene Corr."
                
            data = Dataset.objects.get(pk=dForm["publicDataset"])    
            lGeneset = request.user.get_geneset()
            for geneset in lGeneset:
                try:
                    wrkf = Workflow()
                    dForm["publicGeneset"] = [geneset.pk,]
                    dForm["label"] = data.name +"_"+geneset.name
                    wrkf.assign_workflow(dForm, session_slug)

                    QueueCorrelation(
                        wrkf, method=dForm["analysis"], 
                        survival = dForm["survival"])

                except Exception as error:
                    print(error)

            return redirect('session_detail', session_slug=session_slug)
        else:
            print("Algo ha ido mal")
            print(form.errors)

        return render(request, self.template_name, {'form': form})


###########
## Error ##
###########

def handlernotowner(request):
    """
    Index view
    """
    return render(request, 'error/NotOwner.html')

def handler404(request, exception):
    template_name="error/404.html"
    response = render(request, template_name)
    context = {}
    return render(request,template_name, context)

def handler500(request):
    template_name="error/500.html"
    response = render(request, template_name)
    context = {}
    return render(request,template_name, context)

def handler403(request,exception):
    template_name="error/403.html"
    response = render(request, template_name)
    context = {}
    return render(request,template_name, context)