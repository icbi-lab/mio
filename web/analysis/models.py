from re import M
from django.db import models
import uuid
from django.db.models.enums import Choices
from django.db.models.fields.related import ForeignKey
from registration.models import User
from django.conf import settings
from django.urls import reverse
from django.core.files.storage import FileSystemStorage
import os
from django.core.files.base import ContentFile
import pandas as pd
import io
from miopy import concat_matrix, read_count, voom_normal, tmm_normal, differential_expression_array, differential_expression_edger, get_mir_gene_ratio, header_list
from django.db.models import Q
from django.http import HttpResponseForbidden
from django.core.exceptions import PermissionDenied
from django.shortcuts import render, redirect
from microrna.models import Mirnaset, Mirna_mature
from gene.models import Gene, Geneset
from mirWeb.settings import DATA_DIR


# Create your models here.
data_root = FileSystemStorage(location=settings.DATA_DIR)
img_root = FileSystemStorage(location=settings.IMG_ROOT)


class Session(models.Model):
    """
    Mean table of the database
    """
    def get_absolute_url(self): # provides a default if Session is called from views.py without a specified reverse or redirect
        return reverse('session_detail', kwargs={'session_slug':self.identifier})

    def __str__(self): # provides a default session string
        return str(self.name)

    def get_workflows(self): # Obtain all Workflows to this session
        return self.workflow_set.all().order_by('id')

    def get_files(self):# Obtain all files to this session
        return File.objects.filter(workflow_id__in=self.get_workflows())

    def get_number_workflows(self):# Obtain all files to this session
        return len(list(self.workflow_set.all()))

    def is_public(self):
        return bool(self.public)

    def is_owner(self, user):
        return True if self.user_id == user else False

    def check_permissions(self, user):
        if self.is_public() or self.is_owner(user):
            #print("Is Public")
            pass
        else:
            #print("Forbidden")
            raise PermissionDenied()

    def check_permissions_analysis(self, user):
        if self.is_owner(user):
            print("Is Owner")
            return True
        else:
            print("Not owner")
            return False

    def session_redirect(self, user):
        lSession = user.get_session()
        if len(lSession) == 0:
            session = Session()
            session.user_id = user
            name = user.username + "_" + str(uuid.uuid4())
            session.name = name[0: 49 if len(name) > 49 else -1]
            session.save()
            identifier = session.identifier
        else:
            identifier = lSession[0].identifier
        return identifier

    #Unic identifier used in the URL
    identifier = models.UUIDField(default = uuid.uuid4, editable = False, unique = True)
    public = models.BooleanField(choices =  [(0, "No"),(1,"Yes")], default=0)
    name = models.CharField(max_length=50,unique=True,null=False)

    user_id = models.ForeignKey(User, on_delete=models.CASCADE)



    

class Dataset(models.Model):

    def get_rnafile_name(self):
        return self.geneFile.name.split("/")[-1]

    def get_mirfile_name(self):
        return self.mirFile.name.split("/")[-1]

    def get_metadatafile_name(self):
        return self.metadataFile.name.split("/")[-1]

    def file_download(self, file):
        from wsgiref.util import FileWrapper
        from django.http import HttpResponse
        # print(f'\n Data Downlod called')

        if file == "mir":
            html_path = os.path.join(DATA_DIR,self.mirFile.name)

        elif file == "gene":
            html_path = os.path.join(DATA_DIR,self.geneFile.name)
        else:
            html_path = os.path.join(DATA_DIR,self.metadataFile.name)

        
        # return session_data
        file_wrapper = FileWrapper(open(html_path, 'rb'))
        response = HttpResponse(file_wrapper)
        response['X-Sendfile'] = html_path
        response['Content-Length'] = os.stat(html_path).st_size
        response['Content-Disposition'] = 'attachment; filename=%s' % html_path.split("/")[-1]
        return response

    def get_session_path(self, filename):
        return os.path.join(str(self.user_id.identifier), 'Dataset',self.name, filename) # version with dash
        # return os.path.join(self.identifier.hex, filename) # version without dash
    
    
    def is_owner(self, user):
        return True if self.user_id == user else False
        
    def __str__(self):
        return '%s' %self.name

    def get_table(self, lGene, lMir):
        table = pd.read_csv(settings.DATA_DIR+"/"+self.corFile.url, index_col=0)
        matrix = pd.read_csv(settings.DATA_DIR+"/"+self.pearFile.url, index_col=0)

        try:
            matrix = matrix.loc[lMir,lGene]#Subset the Correlation matrix to the heatmap
        except:
            matrix = matrix.loc[lGene,lMir]#Subset the Correlation matrix to the heatmap

        table = table[(table["Gene"].isin(lGene)) & (table["Mir"].isin(lMir))]

        return table, matrix
        
    def get_absolute_url(self):
        return reverse('analysis:session_detail', kwargs={'pk':self.pk})

    def delete(self, *args, **kwargs):
        self.geneFile.delete()
        self.mirFile.delete()
        self.metadataFile.delete()
        super(Dataset,self).delete(*args, **kwargs)

    def set_exprFile(self, df):
        self.exprFile = ContentFile(df.to_csv())
        self.exprFile.name = "ExpresionSet.%i.csv"%(self.pk)
        self.save()
    
    def set_featurelist(self,file = None):
        from django.core.files.base import ContentFile
        import pickle
        
        content = pickle.dumps(file)
        fid = ContentFile(content)
        self.featureFile.save(f"{self.name}_feature_list.pickle", fid)
        fid.close()
        self.save()
    
    def set_number_gene(self):
        #Get Paths
        genePath = self.get_genepath()
        sep = "," if genePath.endswith(".csv") else "\t"
        df = read_count(genePath, sep)
        self.number_gene = len(df.index)
        self.save()

    def set_number_mir(self):
        #Get Paths
        genePath = self.get_mirpath()
        sep = "," if genePath.endswith(".csv") else "\t"
        df = read_count(genePath, sep)
        self.number_mir = len(df.index)
        self.save()

    def set_number_sample(self):
        #Get Paths
        df = self.get_expr()
        self.number_sample = len(df.index)
        self.set_featurelist(df.columns.tolist())
        self.save()

    def set_metadata_fields(self):
        #Get Paths
        genePath = self.get_metadatapath()
        sep = "," if genePath.endswith(".csv") else "\t"
        df = read_count(genePath, sep)
        self.metadata_fields = ", ".join(df.columns.tolist())
        self.save()

    def get_features(self):
        import pickle
        return pickle.load( open(self.get_featurepath(), "rb" ) ) 

    def get_number_gene(self):
        if self.number_gene == None:
            self.set_number_gene()
        return self.number_gene

    def get_number_mir(self):
        if self.number_mir == None:
            self.set_number_mir()
        return self.number_mir

    def get_number_sample(self):
        if self.number_sample == None:
            self.set_number_sample()
        return self.number_sample

    def get_metadata_fields(self):
        if self.metadata_fields == None:
            self.set_metadata_fields()
        return self.metadata_fields

    def check_dataset(self):
        if self.get_number_gene() == 0 or self.get_number_mir() == 0 or self.get_number_sample() == 0:
            self.delete()
            return False
        else:
            return True

    def get_mirpath(self):
        return settings.DATA_DIR+"/"+ self.mirFile.url

    def get_featurepath(self):
        return settings.DATA_DIR+"/"+ self.featureFile.url

    def get_genepath(self):
        return settings.DATA_DIR+"/"+ self.geneFile.url

    def get_metadatapath(self):
        return settings.DATA_DIR+"/" + self.metadataFile.url

    def get_expr(self, custom_metadata = "", normal = False, survival = False, classifier = False, group = "event", ratio = False, lGene = None, feature = "all", \
        filter_sample = False, group_sample = "event", filter_group = "0", filter_pair = False, low_coef = 0.5, min_db = 20):
        #Get ExprDF

        if bool(self.exprFile) is False:
            
            if self.mirFile.url.endswith(".csv"):
                sep = ","
            else:
                sep = "\t"

            #Get Paths
            mirPath = self.get_mirpath()
            genePath = self.get_genepath()

            print("Obtenido los ficheros")
            
            #Read DF
            if normal and self.technology == "sequencing":
                mirExpr = tmm_normal(mirPath)
                geneExpr = tmm_normal(genePath)
                print("Normalizados los ficheros")

            elif normal and self.technology == "microarray":
                mirExpr = voom_normal(mirPath)
                geneExpr = voom_normal(genePath)
                print("Normalizados los ficheros")

            else:
                mirExpr = read_count(mirPath, sep)
                geneExpr = read_count(genePath, sep)
                print("Ya normalizados")

            dfExpr = concat_matrix(mirExpr, geneExpr)

            self.set_exprFile(dfExpr)

        else:
            dfExpr = pd.read_csv(settings.DATA_DIR+"/" +self.exprFile.url, index_col=0)
            
        #Read Metadata    
        if bool(custom_metadata) is False:
            print("False")
            metadataPath = self.get_metadatapath()

        else:
            print("True")
            metadataPath = settings.DATA_DIR+"/" + custom_metadata.url
        dfMeta = pd.read_csv(metadataPath, index_col=0)
        
        if filter_sample:
                print("Filtrando muestras")
                try:
                    print("Filtrando Decimal")
                    dfMeta = dfMeta[dfMeta[group_sample]==float(filter_group)]
                except:
                    print("Filtrando Texto")
                    dfMeta = dfMeta[dfMeta[group_sample]==filter_group]
                print(dfMeta[group_sample].value_counts())
                dfExpr = dfExpr.loc[set(dfMeta.index.tolist()).intersection(dfExpr.index.tolist()),: ]

        #Create Ratio mir/gene
        if ratio:
            dfExpr = get_mir_gene_ratio(dfExpr, lGeneUser=lGene, filter_pair = filter_pair, low_coef = low_coef, min_db = min_db)

        #Add Label Column
        if survival or classifier:
            if classifier:
                print(group)
                print(dfMeta)
                dfMeta = dfMeta[set([group])]
            else:
                dfMeta = dfMeta[set([group, "time"])]
                    
            #print(dfExpr[group].value_counts())

            lMir, lGene = header_list(exprDF=dfExpr)

            if feature == "gene":
                dfExpr = dfExpr[lGene]

            elif feature == "mir":
                dfExpr = dfExpr[lMir]

            dfExpr = pd.concat([dfMeta,dfExpr], axis = 1).dropna()


        return dfExpr


    def run_de_analysis(self, FilterChoice = "NF", custom_metadata = "", pval = 0.05, logfc = 1.2, group = "event", normal = False, wrkflw = None):
        bPaired = FilterChoice.endswith("P")
        print(group)
        if bool(custom_metadata) is False:
                metadataPath = self.get_metadatapath()

        else:
            metadataPath = settings.DATA_DIR+"/" + custom_metadata.url

        if FilterChoice != "NF":
            if self.technology == "microarray":
                DEM = differential_expression_array(self.get_mirpath(), metadataPath, paired = bPaired, group = group, bNormal = normal)
                DEG = differential_expression_array(self.get_genepath(), metadataPath, paired = bPaired, group = group, bNormal = normal)
                print("Analisis DEG")

                lDEG = DEG[((DEG["adj.P.Val"] < pval) & ((DEG["logFC"] < -logfc) | (DEG["logFC"] > logfc)))].index.tolist()
                lMir = DEM[((DEM["adj.P.Val"] < pval) & ((DEM["logFC"] < -logfc) | (DEM["logFC"] > logfc)))].index.tolist()


            elif self.technology == "sequencing":
                DEM = differential_expression_edger(self.get_mirpath(), metadataPath, paired = bPaired, group = group, bNormal = normal)
                DEG = differential_expression_edger(self.get_genepath(), metadataPath, paired = bPaired, group = group, bNormal = normal)
                print("Analisis DEG")

                lDEG = DEG[((DEG["adj.P.Val"] < pval) & ((DEG["logFC"] < -logfc) | (DEG["logFC"] > logfc)))].index.tolist()
                lMir = DEM[((DEM["adj.P.Val"] < pval) & ((DEM["logFC"] < -logfc) | (DEM["logFC"] > logfc)))].index.tolist()

            File().set_data(wrkflw, DEM, "DEM", filter, "DEM")
            File().set_data(wrkflw, DEG, "DEG", filter, "DEG")


        return lMir, lDEG


    TECHNOLOGY_CHOICES = [
        ("sequencing", "Sequencing Data"),
        ("microarray", "Microarray Data")
    ]

    name = models.CharField(max_length=250, unique=True)
    geneFile = models.FileField(storage=data_root, upload_to=get_session_path, blank=False, null=False)
    mirFile = models.FileField(storage=data_root, upload_to=get_session_path, blank=False, null=False)
    exprFile = models.FileField(storage=data_root, upload_to=get_session_path, blank=True, null=True)
    metadataFile = models.FileField(storage=data_root,upload_to=get_session_path, blank=False, null=False)
    featureFile = models.FileField(storage=data_root,upload_to=get_session_path, blank=True, null=True)
    public = models.BooleanField(choices =  [(False, "No"),(True,"Yes")], default=True)
    technology = models.CharField(max_length=50, choices=TECHNOLOGY_CHOICES)
    corFile = models.FileField(storage=data_root, upload_to=get_session_path, blank=True, null=True)
    pearFile = models.FileField(storage=data_root, upload_to=get_session_path, blank=True, null=True)
    number_gene = models.PositiveIntegerField(blank=True, null=True)
    number_mir = models.PositiveIntegerField(blank=True, null=True)
    number_sample = models.PositiveIntegerField(blank=True, null=True)
    metadata_fields = models.TextField(blank=True, null=True)

    user_id = models.ForeignKey(User, on_delete=models.CASCADE)



class Workflow(models.Model):
    def get_session_path(self, filename):
        path = os.path.join(str(settings.DATA_DIR),str(self.sesion_id.user_id.identifier), "Sessions", str(self.sesion_id.identifier),"file") # version with dash
        return path
        # return os.path.join(self.identifier.hex, filename) # version without dash
    
    def __str__(self):
        return 'workflow_' + str(self.pk)

    def get_absolute_url(self):
        return reverse('analysis:session_detail', kwargs={'pk':self.pk})

    def set_status(self, i):
        self.status = i
        self.save()

    def delete(self, *args, **kwargs):
        try:
            self.cancel_job()
        except:
            pass
        super(Workflow, self).delete(*args, **kwargs)
    
    def get_genes(self):
        lGene = []
        for gs in self.geneset_id.all():
            lGene += list(gs.get_genes()) 
        return list(set(lGene))

    def get_mir(self):
        lMir = []
        for mr in self.mirnaset_id.all():
            lMir += list(mr.get_mir()) 
        return list(set(lMir))


    def get_mirset(self):
        lMirSet = []
        for ms in self.mirnaset_id.all():
            lMirSet.append(ms.name)
        return lMirSet

    def get_geneset(self):
        lGeneSet = []
        for gs in self.geneset_id.all():
            lGeneSet.append(gs.name)
        return lGeneSet

    def set_log(self, log):
        self.logs = log
        self.save()


    def set_feature(self,feature):
        self.feature_type = feature
        self.save()

    def set_group_data(self,group_data):
        self.group_data = group_data
        self.save()


    def get_log(self):
        return self.logs
    
    def get_files_result(self):# Obtain all files to this session
        return File.objects.get(workflow_id=self,is_result=True)

    def get_files(self):# Obtain all files to this session
        return File.objects.filter(workflow_id=self)

    def is_owner(self, user):
        return True if self.sesion_id.user_id == user else False

    def get_session_identifier(self):
        return str(self.sesion_id.identifier)

    def get_jobid(self):
        return str(Queue.objects.get(workflow_id=self.pk).job_id)

    def cancel_job(self):
        from redis import Redis
        from rq.job import Job

        redis = Redis()
        job = Job.fetch(self.get_jobid(), connection=redis)
        job.cancel()
        job.cleanup()

    def assign_workflow(self, dForm, session_slug):        
        #Obtain dict
                
        #Link the Workflow with the Session
        self.sesion_id = Session.objects.get(identifier=session_slug)

        self.analysis = dForm["analysis"]
        self.label = dForm["label"]
        self.analysis_type = dForm["analysis_type"]

        if dForm["publicDataset"] != "":

            self.dataset_id=Dataset.objects.get(pk=dForm["publicDataset"])

            if "clinicalFile" in list(dForm.keys()) and dForm["clinicalFile"] is not None:
                self.custom_metadata = dForm["clinicalFile"]

            print(Dataset.objects.get(pk=dForm["publicDataset"]))
            
        else:
            try:
                data = Dataset()
                data.sesion_id = Session.objects.get(identifier=session_slug)
                data.name = dForm["name_dataset"]
                data.geneFile = dForm["geneFile"]
                data.mirFile = dForm["mirFile"]

                if dForm["clinicalFile"] is not None:
                    data.metadataFile = dForm["clinicalFile"]
                data.save() 
            except:
                pass
            else:
                self.dataset_id = data

        self.save()

        if "publicGeneset" in list(dForm.keys()) and "Todos" not in dForm["publicGeneset"]:
            print("GS")
            geneset = Geneset.objects.filter(pk__in=dForm["publicGeneset"])
            self.geneset_id.add(*geneset)

        if "publicMirnaset" in list(dForm.keys()) and "Todos" not in dForm["publicMirnaset"]:
            print("MS")
            mirnaset = Mirnaset.objects.filter(pk__in=dForm["publicMirnaset"])
            self.mirnaset_id.add(*mirnaset)

        if "group" in list(dForm.keys()):
            self.set_group_data(dForm["group"])
            
        self.save()


    label = models.CharField(max_length=50, blank=False, null=False)
    analysis = models.CharField(max_length=200)
    analysis_type = models.CharField(max_length=200)
    feature_type = models.CharField(max_length=50, blank=True, null=True)
    group_data = models.CharField(max_length=50, blank=True, null=True)
    status = models.PositiveIntegerField(default=0)
    custom_metadata = models.FileField(storage=data_root, upload_to=get_session_path, blank=True, null=True, max_length=250)
    logs = models.CharField(max_length=300, blank=True, null=True)
    job_id = models.UUIDField( editable = True, unique = True, blank=True, null=True)

    sesion_id = models.ForeignKey(Session, on_delete=models.CASCADE)
    dataset_id = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    geneset_id = models.ManyToManyField(Geneset,  blank=True)
    mirnaset_id = models.ManyToManyField(Mirnaset,  blank=True)


    
class File(models.Model):
    def __str__(self):
        return str(self.label)

    def get_session_path(self, filename): 
        
        path = os.path.join(str(self.workflow_id.sesion_id.user_id.identifier), 'Sessions', \
            str(self.workflow_id.sesion_id.identifier),"file",filename) # version with dash
        # return os.path.join(self.identifier.hex, filename) # version without dash
        return path

    def set_file(self, df):
        self.data = ContentFile(df.to_csv())
        self.data.name = "%s.csv"%(self.label)
    
    def set_data(self, workflow, df, ftype, is_result, description = None):
        self.workflow_id = workflow
        self.type = ftype
        self.label = "%s" %(workflow.label)
        self.is_result = is_result
        self.description = description
        self.set_file(df)
        self.save()

    def load_pickle(self):
        import pickle
        return pickle.load( open(self.get_path(), "rb" ) )  
        
    def set_pickle(self, workflow = None, file = None, ftype = None, is_result = False, description = None, label = None):
        from django.core.files.base import ContentFile
        import pickle
        
        self.workflow_id = workflow
        self.type = ftype
        self.label = label
        self.is_result = is_result
        self.description = description
        content = pickle.dumps(file)
        fid = ContentFile(content)
        self.data.save("ModuleScore.pickle", fid)
        fid.close()
        self.save()

    def get_path(self):
        return "%s/%s"%(str(settings.DATA_DIR),str(self.data.url)) # version with dash

    def delete(self, *args, **kwargs):
        self.data.delete()
        super(File,self).delete(*args, **kwargs)

    def is_owner(self, user):
        return True if self.workflow_id.sesion_id.user_id == user else False

    data = models.FileField(storage=data_root, upload_to=get_session_path, blank=False, null=False, max_length=250)
    label = models.CharField(max_length=70)
    type = models.CharField(max_length=70)
    description = models.CharField(max_length=450)
    is_result = models.BooleanField()
    
    workflow_id = models.ForeignKey(Workflow, on_delete=models.CASCADE)

class Queue(models.Model):
    job_id = models.UUIDField()
    workflow_id = models.ForeignKey(Workflow, on_delete=models.CASCADE)
