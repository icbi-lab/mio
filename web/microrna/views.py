from cgi import print_environ
from django.shortcuts import render
from django.views.generic import (View, CreateView, ListView, DeleteView)
from matplotlib.style import context
from microrna.models import Mirna_mature, Mirna_pre_mature, Mirna_prefam_id, Mirnaset, Target
import pandas as pd
import uuid
from django.db.models import Q
from registration.models import User
# Create your views here.

class MicrornaDetailView(View):
    """
    Main View of the page. We obtain all the information related with the Session thanks to
    the session identifier
    """
    template_name = 'microrna/microrna_view.html'

    def get(self, request,microrna_id):  # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        try:
            mature = Mirna_mature.objects.get(mature_name__iexact=microrna_id)
        except Exception as error:
            print(error)
            try:
                mature = Mirna_mature.objects.get(previous_mature_id__iexact=microrna_id)
            except Exception as error2:
                mature = False
                
        finally:
            if mature:
                mature_context = mature.mirna_pre_mature_set.all()[0]
                precursor = mature.mirna_set.all()[0]
                sfrom, sto = int(float(mature_context.mature_from)), int(float(mature_context.mature_to))
                xsome = precursor.mirna_chromosome_build_set.all()[0]
                #print(precursor.prefam_id.all())
                if xsome.strand == "-":
                    mature_xsome = f"{xsome.xsome}:{xsome.contig_start+sto}:{xsome.contig_end}"
                context = {"mature":mature}
                context["sequence"] = precursor.sequence
                context["mature_sequence"] = precursor.sequence[sfrom-1:sto]
                context["xsome"] = xsome
                #context["mature_xsome"] = mature_xsome
                context["precursor"] = precursor
                context["bMir"] = True
                #Target
                target = pd.DataFrame(Target.objects.filter(mirna_id = mature, number_target__gte = 1).values("id" , "gene_id_id__symbol", "mirna_id_id__mature_name", "target", "number_target"))
                
                if not target.empty:
                    context["col"] = ["ID" , "Gene Symbol", "microRNA Mature ID", "Prediction Tools", "Number Prediction Tools"]
                    target.columns =  ["ID" , "Gene Symbol", "microRNA Mature ID", "Prediction Tools", "Number Prediction Tools"]
                    target
                                #Get Json File
                    result = target.to_json(orient='values')
                    identifier = str(uuid.uuid4())
                    context["identifier"] = identifier
                    request.session[identifier]=result

                #Mirset
                #mirset = pd.DataFrame(mature).values("id" , "name", "description","ref_link","mirna_id__mature_name"))
                mirset = pd.DataFrame(mature.mirnaset_set.all().values("id" , "name", "description","ref_link","mirna_id__mature_name"))
                if not mirset.empty:
                    print(mirset)
                    result = mirset.to_json(orient='values')
                    identifier = str(uuid.uuid4())
                    context["identifier2"] = identifier
                    request.session[identifier]=result
                    context["col2"] = ("id" , "name", "description","ref_link","mirna_id__mature_name")
            else:
                context = {"bMir" : False}
                context["mature"] = microrna_id

        return render(request, self.template_name, context)


class MirnasetListView(ListView):
    #https://simpleisbetterthancomplex.com/tutorial/2016/08/03/how-to-paginate-with-django.html
    model = Mirnaset
    context_object_name = 'mirnasets'
    paginate_by = 5
    template_name = 'microrna/mirset_list.html'  # Specify your own template name/location

    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(*args, **kwargs)
        context['query'] = self.request.GET.get('search')
        return context

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

class MirnasetDetailView(View):
    """
    Main View of the page. We obtain all the information related with the Session thanks to
    the session identifier
    """
    template_name = 'microrna/mirnaset_view.html'

    def get(self, request, mirnaset_id):  # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        try:
            mirnaset = Mirnaset.objects.get(name__iexact=mirnaset_id)
        except Exception as error:
            print(error)
            try:
                mirnaset = Mirnaset.objects.get(pk=mirnaset_id)
            except Exception as error2:
                mirnaset = False
                
        finally:
            if mirnaset and mirnaset.public:
                context = {"mirnaset" : mirnaset}
                context["bMirnaset"] = True
                #Target
                #in_set = pd.DataFrame(Target.objects.filter(gene_id = gene, number_target__gte = 1).values("id" , "gene_id_id__symbol", "mirna_id_id__mature_name", "target", "number_target"))
                

            else:
                context = {"bMirnaset" : False}
                context["mirnaset"] = mirnaset_id

        return render(request, self.template_name, context)
