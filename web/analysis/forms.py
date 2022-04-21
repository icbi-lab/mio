from re import M
from django import forms
from .models import Session, Geneset, Mirnaset, Dataset
from django.core.exceptions import ValidationError
from mirWeb.settings import CONTENT_TYPES, MAX_UPLOAD_SIZE
from django.forms.widgets import HiddenInput
import os
from mirWeb.settings import BASE_DIR
####FileField####
from django import forms
from django.template.defaultfilters import filesizeformat
from django.utils.translation import ugettext_lazy as _
from django.core.exceptions import ValidationError
from miopy import load_matrix_header


class RestrictedFileField(forms.FileField):
    """
    Same as FileField, but you can specify:
    * content_types - list containing allowed content_types. Example: ['application/pdf', 'image/jpeg']
    * max_upload_size - a number indicating the maximum file size allowed for upload.
        2.5MB - 2621440
        5MB - 5242880
        10MB - 10485760
        20MB - 20971520
        50MB - 5242880
        100MB - 104857600
        250MB - 214958080
        500MB - 429916160
"""

    def __init__(self, *args, **kwargs):
        self.content_types = kwargs.pop("content_types")
        self.max_upload_size = kwargs.pop("max_upload_size")

        super(RestrictedFileField, self).__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        file = super(RestrictedFileField, self).clean(data, initial)

        try:
            print(file.content_type)
            content_type = file.content_type
            if content_type in self.content_types:
                if float(file.size) > float(self.max_upload_size):
                    raise ValidationError(_('Please keep filesize under %s. Current filesize %s') % (
                        filesizeformat(self.max_upload_size), filesizeformat(file._size)))
            else:
                raise ValidationError(_('Filetype not supported.'))
        except Exception as error:
            print(error)
            pass

        return data


##### Survival ####
class KaplanMeierForm(forms.Form):
    """
    Form to select all Available public session in the DB.
    This work with SessionIndexView(FormView)
    """

    ### Data From DB###
    def __init__(self, *args, **kwargs):
        #Get user data
        self.user = kwargs.pop('user', None)
        super(KaplanMeierForm, self).__init__(*args, **kwargs)

        ### Select Public DataSet ###
        try:
            DATASET_CHOICES = list(Dataset.objects.filter(public=1).values_list("pk","name"))
            
            if self.user.is_authenticated:
                DATASET_CHOICES_USER = list(Dataset.objects.filter(user_id=self.user).values_list("pk","name"))

                DATASET_CHOICES += DATASET_CHOICES_USER
                DATASET_CHOICES = list(set(DATASET_CHOICES))

        except Exception as error:
            print(error)
            DATASET_CHOICES =[]

        dataset = forms.ChoiceField(label = "Available Datasets", choices=DATASET_CHOICES)

        self.fields["dataset"] = dataset

    
    #BASE QUERY
    path_dir = os.path.join(BASE_DIR,"static/data/lFeature.txt")
    result = open(path_dir,"r").read().split()
    res = list(zip(result,result))
    target = forms.ChoiceField(choices = res, label="Gene or miRNA name",required=False)
    get_cutoff = forms.BooleanField(label="Determine the optimal cutpoint of variables", required=False)
    q = forms.FloatField(label="Quantile to split the sample in Higher and Lower", initial=0.5, min_value=0, max_value=1,
        widget = forms.NumberInput(attrs={'id': 'form_q', 'step': "0.05"}))
    

##### Session ####
class SessionSearchForm(forms.Form):
    """
    Form to select all Available public session in the DB.
    This work with SessionIndexView(FormView)
    """

    ### Data From DB###
    def __init__(self, *args, **kwargs):
        #Get user data
        self.user = kwargs.pop('user', None)
        super(SessionSearchForm, self).__init__(*args, **kwargs)

        ### Select Public DataSet ###
        try:


            SESSION_CHOICES = list(Session.objects.filter(public=1).values_list("identifier","name"))
            
            if self.user.is_authenticated:
                SESSION_CHOICES_USER = list(Session.objects.filter(user_id=self.user).values_list("identifier","name"))

                SESSION_CHOICES += SESSION_CHOICES_USER
                SESSION_CHOICES = list(set(SESSION_CHOICES))
        except Exception as error:
            print(error)
            SESSION_CHOICES =[]

        session = forms.ChoiceField(label = "Available Analysis", choices=SESSION_CHOICES)

        self.fields["session"] = session 


class SessionCreateForm(forms.ModelForm):
    """
    Form to create a new Workflow.
    The session_id is created in the view.
    This Form is used by WorkflowCreateView
    """

    class Meta:
        model = Session
        fields = ["name"]


##### Dataset #####
class DatasetForm(forms.ModelForm):
    """
    Form to create a new Workflow.
    The session_id is created in the view.
    This Form is used by WorkflowCreateView
    """
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(DatasetForm, self).__init__(*args, **kwargs)  

        if self.user.is_staff is False:
            self.fields["public"].widget = HiddenInput()

    geneFile = RestrictedFileField(label="Select gene File", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)
    mirFile = RestrictedFileField(label="Select miRNA File", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)
    clinicalFile = RestrictedFileField(label="Select clinical File", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)

    class Meta:
        model = Dataset
        exclude = ["exprFile", "user_id","metadataFile","mirFile","geneFile",\
             "corFile", "pearFile", "number_gene","number_mir", "number_sample",\
             "metadata_fields", "featureFile"]


##### Workflow ####

class WorkflowFilterForm(forms.Form):
    """
    Form to get the Correlation data for the analysis.

    Ref:
    https://stackoverflow.com/questions/50960800/searchable-drop-down-for-choice-field-in-django-admin
    """

    pval = forms.FloatField(min_value=0, max_value=1, initial=0.05,label="Maximum adjust P.value", 
        widget = forms.NumberInput(attrs={'id': 'form_pval', 'step': "0.01"}))

    #nDB = forms.IntegerField(min_value=0, max_value=20, initial=6,label="Minimum Number of Predictions Tools")
    low_coef = forms.FloatField(min_value=-1, max_value=1, initial=-0.5, 
        label="Coefficient equal or lower than",     
        widget = forms.NumberInput(attrs={'id': 'form_low_coef', 'step': "0.1"}))

    high_coef = forms.FloatField(min_value=-1, max_value=1, initial=0.5, 
        label="Coefficient equal or higher than",
        widget = forms.NumberInput(attrs={'id': 'form_high_coef', 'step': "0.1"}))

    survival = forms.BooleanField(label="Get log hazard ratio for Gene and miRNA", required=False)

    join = forms.ChoiceField(choices=(("or","OR"),("and","AND")), label="Filter method to target databases",required=False)

    DB_CHOICES = tuple(zip(load_matrix_header(),load_matrix_header()))

    nDB = forms.MultipleChoiceField(choices=DB_CHOICES,label="Minimum number of Predictions Tools", required=False, \
        widget=forms.CheckboxSelectMultiple(attrs={
                "checked": "",
                "class": "form-check form-check-inline"
            }))
    METHOD_CHOICES = ( ("R","Pearson (R)"),
                     ("Rho","Spearman (Rho)"), 
                     ("Tau","Kendall (Tau)"),
                     ("Lasso","Lasso"),
                     ("Ridge","Ridge"),
                     ("ElasticNet","Elastic net"))
                     
    method = forms.ChoiceField(choices=METHOD_CHOICES,label="Select the coefficient for applying the filters", initial="R") 

    min_db = forms.IntegerField(initial=5, max_value=40, min_value=0, required=False)        


class WorkflowFilterBasicForm(forms.Form):
    """
    Form to get the Correlation data for the analysis.

    Ref:
    https://stackoverflow.com/questions/50960800/searchable-drop-down-for-choice-field-in-django-admin
    """

    pval = forms.FloatField(min_value=0, max_value=1, initial=0.05,label="Maximum adjust P.value", 
        widget = forms.NumberInput(attrs={'id': 'form_pval', 'step': "0.01"}))

    #nDB = forms.IntegerField(min_value=0, max_value=20, initial=6,label="Minimum Number of Predictions Tools")
    low_coef = forms.FloatField(min_value=-1, max_value=1, initial=-0.5, 
        label="Coefficient equal or lower than",     
        widget = forms.NumberInput(attrs={'id': 'form_low_coef', 'step': "0.1"}))

    high_coef = forms.FloatField(min_value=-1, max_value=1, initial=0.5, 
        label="Coefficient equal or higher than",
        widget = forms.NumberInput(attrs={'id': 'form_high_coef', 'step': "0.1"}))
                     
    METHOD_CHOICES = ( ("R","Pearson (R)"),
                     ("Rho","Spearman (Rho)"), 
                     ("Tau","Kendall (Tau)"),
                     ("Lasso","Lasso"),
                     ("Ridge","Ridge"),
                     ("ElasticNet","Elastic net"))
                     
    method = forms.ChoiceField(choices=METHOD_CHOICES,label="Select the coefficient for applying the filters", initial="R", required=False)   

class WorkflowIPSCorForm(forms.Form):
    """
    Form to get the Correlation data for the analysis.

    Ref:
    https://stackoverflow.com/questions/50960800/searchable-drop-down-for-choice-field-in-django-admin
    https://stackoverflow.com/questions/57143113/django-how-to-use-the-admin-autocomplete-field-in-a-custom-form
    """

    ### Data From DB###
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(WorkflowIPSCorForm, self).__init__(*args, **kwargs)

        ### Select Public DataSet ###
        try:
            DATASET_CHOICES = list(Dataset.objects.filter(public=True).values_list("id","name"))
            #print(DATASET_CHOICES)

            if self.user.is_authenticated:
                DATASET_USER = list(Dataset.objects.filter(user_id=self.user).values_list("id","name"))
                DATASET_CHOICES += DATASET_USER
            DATASET_CHOICES = list(set(DATASET_CHOICES))

        except Exception as error:
            print(error)
            DATASET_CHOICES =[]

        publicDataset = forms.ChoiceField(choices=[(None,"-----"),]+DATASET_CHOICES, label="Select available dataset", required=False)
        self.fields["publicDataset"] = publicDataset
      
    ### Name ###
    label = forms.CharField(label="Add a name for the Analysis")

    ### Custom Files ###
    TECHNOLOGY_CHOICES = [
        ("sequencing", "Sequencing Data"),
        ("microarray", "Microarray Data")
    ]
    geneFile = RestrictedFileField(label="Select Gene File", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)
    mirFile = RestrictedFileField(label="Select miRNA File", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)
    clinicalFile = RestrictedFileField(label="Add Custom Metadata", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)
    technology = forms.ChoiceField(choices=TECHNOLOGY_CHOICES)

    ##Dataset Chioces##

    PUBLIC_CHOICES = (
        (3,"--------"),
        (0, "Available Dataset"),
        (1, "Provide Own Dataset"),
    )

    dataset = forms.ChoiceField(choices=PUBLIC_CHOICES, label="Select available dataset", required=False)
    name_dataset = forms.CharField(label="Name Dataset", required=False)

    ### Filter Sample
    filter_sample = forms.BooleanField(label="Apply normalization to the data", required=False)
    group_sample = forms.CharField(label="Group name in metadata for filter", initial="sample_type", required=False)
    filter_group = forms.CharField(label="Group name in metadata for Correlation", initial="PrimaryTumor", required=False)

    ### Normalize ###
    normal = forms.BooleanField(label="Apply normalization to the data", required=False)



class WorkflowInfiltratedCorForm(forms.Form):
    """
    Form to get the Correlation data for the analysis.

    Ref:
    https://stackoverflow.com/questions/50960800/searchable-drop-down-for-choice-field-in-django-admin
    https://stackoverflow.com/questions/57143113/django-how-to-use-the-admin-autocomplete-field-in-a-custom-form
    """

    ### Data From DB###
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(WorkflowInfiltratedCorForm, self).__init__(*args, **kwargs)

        ### Select Public DataSet ###
        try:
            DATASET_CHOICES = list(Dataset.objects.filter(name="TCGA-OV").values_list("id","name"))
            #print(DATASET_CHOICES)

        except Exception as error:
            print(error)
            DATASET_CHOICES =[]

        publicDataset = forms.ChoiceField(choices=[(None,"-----"),]+DATASET_CHOICES, label="Select available dataset", required=False)
        self.fields["publicDataset"] = publicDataset
      
    ### Name ###
    label = forms.CharField(label="Add a name for the Analysis")

    #BASE QUERY
    path_dir = os.path.join(BASE_DIR,"static/data/lMethod.txt")
    result = open(path_dir,"r").read().split("\n")

    lTimer = [x for x in result if x.startswith("TIMER")]
    TIMER_CHOICES = list(zip(lTimer,[x.replace("TIMER|","") for x in lTimer]))

    lTimer = [x for x in result if x.startswith("MCP_COUNTER")]
    MCP_COUNTER_CHOICES = list(zip(lTimer,[x.replace("MCP_COUNTER|","") for x in lTimer]))
    #print(MCP_COUNTER_CHOICES)
    lTimer = [x for x in result if x.startswith("QUANTISEQ")]
    QUANTISEQ_CHOICES = list(zip(lTimer,[x.replace("QUANTISEQ|","") for x in lTimer]))

    lTimer = [x for x in result if x.startswith("EPIC")]
    EPICCHOICES = list(zip(lTimer,[x.replace("EPIC|","") for x in lTimer]))

    timer = forms.MultipleChoiceField(choices=TIMER_CHOICES, label="Select Cell", required=False, \
        widget=forms.CheckboxSelectMultiple(attrs={
                "checked": "",
                "class": "form-check form-check-inline"
            }))
    mcp = forms.MultipleChoiceField(choices=MCP_COUNTER_CHOICES, label="Select Cell", required=False, \
        widget=forms.CheckboxSelectMultiple(attrs={
                "checked": "",
                "class": "form-check form-check-inline"
            }))
    quantiseq = forms.MultipleChoiceField(choices=QUANTISEQ_CHOICES, label="Select Cell", required=False, \
        widget=forms.CheckboxSelectMultiple(attrs={
                "checked": "",
                "class": "form-check form-check-inline"
            }))
    epic = forms.MultipleChoiceField(choices=EPICCHOICES, label="Select Cell", required=False, \
        widget=forms.CheckboxSelectMultiple(attrs={
                "checked": "",
                "class": "form-check form-check-inline"
            }))
    ### Filter Sample
    filter_sample = forms.BooleanField(label="Apply normalization to the data", required=False)
    group_sample = forms.CharField(label="Group name in metadata for filter", initial="sample_type", required=False)
    filter_group = forms.CharField(label="Group name in metadata for Correlation", initial="PrimaryTumor", required=False)

    ### Normalize ###
    normal = forms.BooleanField(label="Apply normalization to the data", required=False)




class WorkflowGenesetCorForm(forms.Form):
    """
    Form to get the Correlation data for the analysis.

    Ref:
    https://stackoverflow.com/questions/50960800/searchable-drop-down-for-choice-field-in-django-admin
    https://stackoverflow.com/questions/57143113/django-how-to-use-the-admin-autocomplete-field-in-a-custom-form
    """

    ### Data From DB###
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(WorkflowGenesetCorForm, self).__init__(*args, **kwargs)

        ### Select Public DataSet ###
        try:
            DATASET_CHOICES = list(Dataset.objects.filter(public=True).values_list("id","name"))
            #print(DATASET_CHOICES)

            if self.user.is_authenticated:
                DATASET_USER = list(Dataset.objects.filter(user_id=self.user).values_list("id","name"))
                DATASET_CHOICES += DATASET_USER
            DATASET_CHOICES = list(set(DATASET_CHOICES))

        except Exception as error:
            print(error)
            DATASET_CHOICES =[]

        publicDataset = forms.ChoiceField(choices=[(None,"-----"),]+DATASET_CHOICES, label="Select available dataset", required=False)
        self.fields["publicDataset"] = publicDataset


            ### Select Gene Pathway ###
        try:
            GENESET_CHOICES = list(Geneset.objects.filter(public=True).values_list("id","name"))
            if self.user.is_authenticated:
                GENESET_USER = list(Geneset.objects.filter(user_id=self.user).values_list("id","name"))
                GENESET_CHOICES += GENESET_USER
            GENESET_CHOICES = list(set(GENESET_CHOICES))

        except:
            GENESET_CHOICES =[]

        #GENESET_CHOICES = [("", "Using Genes in the file")] + GENESET_CHOICES
        publicGeneset = forms.MultipleChoiceField(choices=GENESET_CHOICES, label="Select geneset", required=True)
        self.fields["publicGeneset"] = publicGeneset
      
    ### Name ###
    label = forms.CharField(label="Add a name for the Analysis")

    ### Custom Files ###
    TECHNOLOGY_CHOICES = [
        ("sequencing", "Sequencing Data"),
        ("microarray", "Microarray Data")
    ]
    geneFile = RestrictedFileField(label="Select Gene File", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)
    mirFile = RestrictedFileField(label="Select miRNA File", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)
    clinicalFile = RestrictedFileField(label="Add Custom Metadata", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)
    technology = forms.ChoiceField(choices=TECHNOLOGY_CHOICES)

    ##Dataset Chioces##

    PUBLIC_CHOICES = (
        (3,"--------"),
        (0, "Available Dataset"),
        (1, "Provide Own Dataset"),
    )

    dataset = forms.ChoiceField(choices=PUBLIC_CHOICES, label="Select available dataset", required=False)
    name_dataset = forms.CharField(label="Name Dataset", required=False)

    ### Filter Sample
    filter_sample = forms.BooleanField(label="Apply normalization to the data", required=False)
    group_sample = forms.CharField(label="Group name in metadata for filter", initial="sample_type", required=False)
    filter_group = forms.CharField(label="Group name in metadata for Correlation", initial="PrimaryTumor", required=False)

    ### Normalize ###
    normal = forms.BooleanField(label="Apply normalization to the data", required=False)



class WorkflowCorrelationForm(forms.Form):
    """
    Form to get the Correlation data for the analysis.

    Ref:
    https://stackoverflow.com/questions/50960800/searchable-drop-down-for-choice-field-in-django-admin
    https://stackoverflow.com/questions/57143113/django-how-to-use-the-admin-autocomplete-field-in-a-custom-form
    """

    ### Data From DB###
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(WorkflowCorrelationForm, self).__init__(*args, **kwargs)

        ### Select Public DataSet ###
        try:
            DATASET_CHOICES = list(Dataset.objects.filter(public=True).values_list("id","name"))
            #print(DATASET_CHOICES)

            if self.user.is_authenticated:
                DATASET_USER = list(Dataset.objects.filter(user_id=self.user).values_list("id","name"))
                DATASET_CHOICES += DATASET_USER
            DATASET_CHOICES = list(set(DATASET_CHOICES))

        except Exception as error:
            print(error)
            DATASET_CHOICES =[]

        publicDataset = forms.ChoiceField(choices=[(None,"-----"),]+DATASET_CHOICES, label="Select available dataset", required=False)

        self.fields["publicDataset"] = publicDataset

            ### Select Gene Pathway ###
        try:
            GENESET_CHOICES = list(Geneset.objects.filter(public=True).values_list("id","name"))
            if self.user.is_authenticated:
                GENESET_USER = list(Geneset.objects.filter(user_id=self.user).values_list("id","name"))
                GENESET_CHOICES += GENESET_USER
            GENESET_CHOICES = list(set(GENESET_CHOICES))

        except:
            GENESET_CHOICES =[]

        #GENESET_CHOICES = [("", "Using Genes in the file")] + GENESET_CHOICES
        publicGeneset = forms.MultipleChoiceField(choices=GENESET_CHOICES, label="Select geneset", required=False)
        
        #publicGeneset = forms.ModelMultipleChoiceField(label="Available Genesets",
        #queryset=Geneset.objects.all(),
        #widget=AutocompleteSelectMultiple(Workflow._meta.get_field('geneset_id'), admin.site)
        #    )
        self.fields["publicGeneset"] = publicGeneset

            
    ### Name ###
    label = forms.CharField(label="Add a name for the Analysis")

    ### Custom Files ###
    TECHNOLOGY_CHOICES = [
        ("sequencing", "Sequencing Data"),
        ("microarray", "Microarray Data")
    ]
    geneFile = RestrictedFileField(label="Select gene file", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)
    mirFile = RestrictedFileField(label="Select miRNA file", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)
    clinicalFile = RestrictedFileField(label="Add custom metadata", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)
    technology = forms.ChoiceField(choices=TECHNOLOGY_CHOICES)

    ##Dataset Chioces##

    PUBLIC_CHOICES = (
        (3,"--------"),
        (0, "Available dataset"),
        (1, "Provide own dataset"),
    )

    dataset = forms.ChoiceField(choices=PUBLIC_CHOICES, label="Select available dataset", required=False)
    name_dataset = forms.CharField(label="Name Dataset", required=False)

    ### Normalize ###
    normal = forms.BooleanField(label="Apply normalization to the data", required=False)
    survival = forms.BooleanField(label="Get Log Hazard Ratio for gene and miRNA", required=False)

    ## Bacground ##
    background = forms.BooleanField(label="Background modul with non-target gene/microRNA pairs", required=False)


    ### Filter Sample
    filter_sample = forms.BooleanField(label="Apply normalization to the data", required=False)
    group_sample = forms.CharField(label="Group name in metadata for filter", initial="sample_type", required=False)
    filter_group = forms.CharField(label="Group Name in metadata for correlation", initial="PrimaryTumor", required=False)

    ### Select Filter Gene Expression ###
    FILTER_CHOICES = [
        ("NF", "No Filter"),
        ("CCU", "Condition1 vs Condition2 Unpaired"),
        ("CCP", "Condition1 vs Condition2 Paired")]

    filter = forms.ChoiceField(choices = FILTER_CHOICES, label="Differential expression analysis")

    group = forms.CharField(label="Group name in metadata for DEA", initial="event", required=False)
    #filtergroup = forms.CharField(label="Group Name in metadata for COrrelation", initial="event", required=False)

    logfc = forms.FloatField(min_value=0, initial=1.5, label = "Absolute Log Fold Change Value",
        widget = forms.NumberInput(attrs={'id': 'form_logfc', 'step': "0.1"}))

    pval = forms.FloatField(min_value=0, max_value=1, initial=0.01, label = "Minimum P-Value",
        widget = forms.NumberInput(attrs={'id': 'form_pval', 'step': "0.01"}))


class WorkFlowFeaturesForm(forms.Form):
    """
    Form to create a new Workflow.
    The session_id is created in the view.
    This Form is used by WorkflowCreateView
    """
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(WorkFlowFeaturesForm, self).__init__(*args, **kwargs)  

        ### Select Public DataSet ###
        try:
            DATASET_CHOICES = list(Dataset.objects.filter(public=True).values_list("id","name"))
            print(DATASET_CHOICES)

            if self.user.is_authenticated:
                DATASET_USER = list(Dataset.objects.filter(user_id=self.user).values_list("id","name"))
                DATASET_CHOICES += DATASET_USER
            DATASET_CHOICES = list(set(DATASET_CHOICES))

        except Exception as error:
            print(error)
            DATASET_CHOICES =[]

        publicDataset = forms.ChoiceField(choices=DATASET_CHOICES, label="Select available dataset", required=True)

        self.fields["publicDataset"] = publicDataset
    
    topk = forms.IntegerField(min_value=1,max_value=500,label="Obtain top predictors", initial=100)
    k = forms.IntegerField(min_value=1, max_value=10, initial=4, label="Number of cross-validation")

    FEATURE_CHOICES = (("gene", "Gene"),
                        ('mir','miRNAs'),
                        ('all', 'Mir and Genes')
                    )
    feature = forms.ChoiceField(choices=FEATURE_CHOICES, label="Select features", required=True)
    label = forms.CharField(max_length=25, label="Name")
    group = forms.CharField(max_length=25, label="Group name in metadata", initial="event")

    ### Filter Sample
    filter_sample = forms.BooleanField(label="Apply normalization to the data", required=False)
    group_sample = forms.CharField(label="Group name in metadata for filter", initial="sample_type", required=False)
    filter_group = forms.CharField(label="Group Name in metadata for correlation", initial="PrimaryTumor", required=False)


class WorkFlowFeaturesRatioForm(forms.Form):
    """
    Form to create a new Workflow.
    The session_id is created in the view.
    This Form is used by WorkflowCreateView
    """
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(WorkFlowFeaturesRatioForm, self).__init__(*args, **kwargs)  

        ### Select Public DataSet ###
        try:
            DATASET_CHOICES = list(Dataset.objects.filter(public=True).values_list("id","name"))
            print(DATASET_CHOICES)

            if self.user.is_authenticated:
                DATASET_USER = list(Dataset.objects.filter(user_id=self.user).values_list("id","name"))
                DATASET_CHOICES += DATASET_USER
            DATASET_CHOICES = list(set(DATASET_CHOICES))

        except Exception as error:
            print(error)
            DATASET_CHOICES =[]

        publicDataset = forms.ChoiceField(choices=DATASET_CHOICES, label="Select available dataset", required=True)

            ### Select Gene Pathway ###
        try:
            GENESET_CHOICES = list(Geneset.objects.filter(public=True).values_list("id","name"))
            if self.user.is_authenticated:
                GENESET_USER = list(Geneset.objects.filter(user_id=self.user).values_list("id","name"))
                GENESET_CHOICES += GENESET_USER
            GENESET_CHOICES = list(set(GENESET_CHOICES))

        except:
            GENESET_CHOICES =[]

        #GENESET_CHOICES = [("", "Using Genes in the file")] + GENESET_CHOICES
        publicGeneset = forms.MultipleChoiceField(choices=GENESET_CHOICES, label="Select geneset", required=False)
            
        self.fields["publicGeneset"] = publicGeneset
        self.fields["publicDataset"] = publicDataset
    
    topk = forms.IntegerField(min_value=1,max_value=500,label="Obtain top predictors", initial=100)
    k = forms.IntegerField(min_value=1, max_value=10, initial=10, label="Number of cross-validation ")
    label = forms.CharField(max_length=25, label="Name")
    group = forms.CharField(max_length=25, label="Group name in metadata", initial="event")
    
    ### Filter Sample
    filter_sample = forms.BooleanField(label="Apply normalization to the data", required=False)
    group_sample = forms.CharField(label="Group name in metadata for filter", initial="sample_type", required=False)
    filter_group = forms.CharField(label="Group Name in metadata for correlation", initial="PrimaryTumor", required=False)

    ### Filter Pairs
    filter_pair = forms.BooleanField(label="Filter the gene/microRNA pairs", required=False)

    low_coef = forms.FloatField(min_value=-1, max_value=1, initial=-0.5, 
        label="Coefficient equal or lower than",     
        widget = forms.NumberInput(attrs={'id': 'form_low_coef', 'step': "0.1"}))

    min_db = forms.IntegerField(initial=5, max_value=40, min_value=0)        


class WorkFlowClassificationForm(forms.Form):
    """
    Form to create a new Workflow.
    The session_id is created in the view.
    This Form is used by WorkflowCreateView
    """
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(WorkFlowClassificationForm, self).__init__(*args, **kwargs)  

        ### Select Public DataSet ###
        try:
            DATASET_CHOICES = list(Dataset.objects.filter(public=True).values_list("id","name"))
            print(DATASET_CHOICES)

            if self.user.is_authenticated:
                DATASET_USER = list(Dataset.objects.filter(user_id=self.user).values_list("id","name"))
                DATASET_CHOICES += DATASET_USER
            DATASET_CHOICES = list(set(DATASET_CHOICES))

        except Exception as error:
            print(error)
            DATASET_CHOICES =[]

        publicDataset = forms.ChoiceField(choices=DATASET_CHOICES, label="Select available dataset", required=True)

            ### Select Gene Pathway ###
        try:
            GENESET_CHOICES = list(Geneset.objects.filter(public=True).values_list("id","name"))
            if self.user.is_authenticated:
                GENESET_USER = list(Geneset.objects.filter(user_id=self.user).values_list("id","name"))
                GENESET_CHOICES += GENESET_USER
            GENESET_CHOICES = list(set(GENESET_CHOICES))

        except:
            GENESET_CHOICES =[]

        #GENESET_CHOICES = [("", "Using Genes in the file")] + GENESET_CHOICES
        publicGeneset = forms.MultipleChoiceField(choices=GENESET_CHOICES, label="Select geneset", required=False)

            ### Select miRNA Pathway ###
        try:
            MIRNASET_CHOICES = list(Mirnaset.objects.filter(public=True).values_list("id","name"))
            if self.user.is_authenticated:
                MIRNASET_USER = list(Mirnaset.objects.filter(user_id=self.user).values_list("id","name"))
                MIRNASET_CHOICES += MIRNASET_USER
            MIRNASET_CHOICES = list(set(MIRNASET_CHOICES))

        except:
            MIRNASET_CHOICES =[]

        publicMirnaset = forms.MultipleChoiceField(choices=MIRNASET_CHOICES, label="Select geneset", required=False)

            ### Select Model ###
        try:
            if self.user.is_authenticated:
                FITMODEL_CHOICES= list(self.user.get_models().values_list("id","label"))
                FITMODEL_CHOICES = list(set(FITMODEL_CHOICES))

        except:
            FITMODEL_CHOICES =[]

        #GENESET_CHOICES = [("", "Using Genes in the file")] + GENESET_CHOICES
        publicModel = forms.ChoiceField(choices=FITMODEL_CHOICES, label="Select training model", required=False)
            
        self.fields["publicGeneset"] = publicGeneset
        self.fields["publicMirnaset"] = publicMirnaset
        self.fields["publicDataset"] = publicDataset
        self.fields["publicModel"] = publicModel


    MODEL_CHOICE = [("Random Forest","Random Forest"),
                    ("Logistic Regression", "Logistic Regression"),
                    ("Support Vector Machine","Support Vector Machine")]

    model = forms.ChoiceField(choices=MODEL_CHOICE, label="Select Models", required=False)
    k = forms.IntegerField(min_value=1, max_value=10, initial=10, label="Number of cross-validation ")
    label = forms.CharField(max_length=25, label="Name")
    group = forms.CharField(max_length=25, label="Group name in metadata", initial="event")
    use_fit_model = forms.BooleanField(label="Use training model", required=False)
    #use_geneset = forms.BooleanField(label="Uses geneset as feature")
    #use_mirnaset = forms.BooleanField(label="Uses mirnaset as feature")


#################
### TARGET #####
##############

class SyntheticLethalityForm(forms.Form):
    """
    Form to get the Correlation data for the analysis.

    Ref:
    https://stackoverflow.com/questions/50960800/searchable-drop-down-for-choice-field-in-django-admin
    """
    
    def __init__(self, *args, **kwargs):
        self.session = kwargs.pop('session', None)
        self.user = kwargs.pop('user', None)

        super(SyntheticLethalityForm, self).__init__(*args, **kwargs)
        ### Gene ###


            ### Select Gene Pathway ###
        try:
            GENESET_CHOICES = list(Geneset.objects.filter(public=True).values_list("id","name"))
            if self.user.is_authenticated:
                GENESET_USER = list(Geneset.objects.filter(user_id=self.user).values_list("id","name"))
                GENESET_CHOICES += GENESET_USER
            GENESET_CHOICES = list(set(GENESET_CHOICES))

        except:
            GENESET_CHOICES =[]

        #GENESET_CHOICES = [("", "Using Genes in the file")] + GENESET_CHOICES
        publicGeneset = forms.MultipleChoiceField(choices=GENESET_CHOICES, label="Select geneset", required=False)


            ### Select miRNA Pathway ###
            
        try:
            TABLE_CHOICES = list(self.session.get_workflows().filter(analysis = "Correlation").values_list("id","label"))
            print(TABLE_CHOICES)
        except Exception as error:
            print(error)
            TABLE_CHOICES =[]

        publicTable = forms.ChoiceField(choices = TABLE_CHOICES, label="Available analises", required=False)
        self.fields["table"] = publicTable
        self.fields["publicGeneset"] = publicGeneset
        
    #BASE QUERY    
    #BASE QUERY
    path_dir =    path_dir = os.path.join(BASE_DIR,"static/data/lGeneMatrix.txt")
    result = open(path_dir,"r").read().split()
    res = list(zip(result,result))
    tQuery = forms.ChoiceField(choices = res, label="Gene or miRNA name",required=False)
    use_correlation = forms.BooleanField(label="Use correlation result", required=False)
    use_set = forms.BooleanField(label="Predict Geneset or microRNAset target", required=False)

    #CORRELATION QUERY
    METHOD_CHOICES = ( ("R","Pearson (R)"),
                     ("Rho","Spearman (Rho)"), 
                     ("Tau","Kendall (Tau)"),
                     ("Lasso","Lasso"),
                     ("Ridge","Ridge"),
                     ("ElasticNet","Elastic net"))
                     
    method = forms.ChoiceField(choices=METHOD_CHOICES,label="Select the coefficient for applying the filters") 

    pval = forms.FloatField(min_value=0, max_value=1, initial=0.05,label="Maximum adjust P.value", 
        widget = forms.NumberInput(attrs={'id': 'form_pval', 'step': "0.01"}))

    low_coef = forms.FloatField(min_value=-1, max_value=1, initial=-0.5, 
        label="Coefficient equal or lower than",     
        widget = forms.NumberInput(attrs={'id': 'form_low_coef', 'step': "0.1"}))

    high_coef = forms.FloatField(min_value=-1, max_value=1, initial=0.5, 
        label="Coefficient equal or higher than",
        widget = forms.NumberInput(attrs={'id': 'form_high_coef', 'step': "0.1"}))

    survival = forms.BooleanField(label="Get log hazard ratio for Gene and miRNA", required=False)

    ##DB QUERY
    join = forms.ChoiceField(choices=(("or","OR"),("and","AND")), label="Filter method to target Databases",required=False)

    DB_CHOICES = tuple(zip(load_matrix_header(),load_matrix_header()))

    nDB = forms.MultipleChoiceField(choices=DB_CHOICES,label="Minimum number of Predictions Tools", required=False, \
        widget=forms.CheckboxSelectMultiple(attrs={
                "checked": "",
                "class": "form-check form-check-inline"
            }))
    min_db = forms.IntegerField(initial=5, max_value=40, min_value=0)        


class TargetPredictorForm(forms.Form):
    """
    Form to get the Correlation data for the analysis.

    Ref:
    https://stackoverflow.com/questions/50960800/searchable-drop-down-for-choice-field-in-django-admin
    """
    
    def __init__(self, *args, **kwargs):
        self.session = kwargs.pop('session', None)
        self.user = kwargs.pop('user', None)

        super(TargetPredictorForm, self).__init__(*args, **kwargs)
        ### Gene ###

            ### Select Gene Pathway ###
        try:
            GENESET_CHOICES = list(Geneset.objects.filter(public=True).values_list("id","name"))
            if self.user.is_authenticated:
                GENESET_USER = list(Geneset.objects.filter(user_id=self.user).values_list("id","name"))
                GENESET_CHOICES += GENESET_USER
            GENESET_CHOICES = list(set(GENESET_CHOICES))

        except Exception as error:
            print(error)
            GENESET_CHOICES =[]

        #GENESET_CHOICES = [("", "Using Genes in the file")] + GENESET_CHOICES
        publicGeneset = forms.MultipleChoiceField(choices=GENESET_CHOICES, label="Select geneset", required=False)

            ### Select miRNA Pathway ###
        try:
            MIRNASET_CHOICES = list(Mirnaset.objects.filter(public=True).values_list("id","name"))
            if self.user.is_authenticated:
                MIRNASET_USER = list(Mirnaset.objects.filter(user_id=self.user).values_list("id","name"))
                MIRNASET_CHOICES += MIRNASET_USER
            MIRNASET_CHOICES = list(set(MIRNASET_CHOICES))

        except:
            MIRNASET_CHOICES =[]

        #GENESET_CHOICES = [("", "Using Genes in the file")] + GENESET_CHOICES
        publicMirnaset = forms.MultipleChoiceField(choices=MIRNASET_CHOICES, label="Select geneset", required=False)
            
        try:
            TABLE_CHOICES = list(self.session.get_workflows().filter(analysis = "Correlation").values_list("id","label"))
            print(TABLE_CHOICES)
        except Exception as error:
            print(error)
            TABLE_CHOICES =[]

        publicTable = forms.ChoiceField(choices = TABLE_CHOICES, label="Available analises", required=False)
        self.fields["table"] = publicTable
        self.fields["publicGeneset"] = publicGeneset
        self.fields["publicMirnaset"] = publicMirnaset
        
    #BASE QUERY
    path_dir = os.path.join(BASE_DIR,"static/data/lFeature.txt")
    result = open(path_dir,"r").read().split()
    res = list(zip(result,result))
    tQuery = forms.ChoiceField(choices = res, label="Gene or miRNA name",required=False)
    use_correlation = forms.BooleanField(label="Use correlation result", required=False)
    use_set = forms.BooleanField(label="Predict Geneset or microRNAset target", required=False)


    #CORRELATION QUERY
    METHOD_CHOICES = ( ("R","Pearson (R)"),
                     ("Rho","Spearman (Rho)"), 
                     ("Tau","Kendall (Tau)"),
                     ("Lasso","Lasso"),
                     ("Ridge","Ridge"),
                     ("ElasticNet","Elastic net"))
                     
    method = forms.ChoiceField(choices=METHOD_CHOICES,label="Select the coefficient for applying the filters") 

    pval = forms.FloatField(min_value=0, max_value=1, initial=0.05,label="Maximum adjust P.value", 
        widget = forms.NumberInput(attrs={'id': 'form_pval', 'step': "0.01"}))

    low_coef = forms.FloatField(min_value=-1, max_value=1, initial=-0.5, 
        label="Coefficient equal or lower than",     
        widget = forms.NumberInput(attrs={'id': 'form_low_coef', 'step': "0.1"}))

    high_coef = forms.FloatField(min_value=-1, max_value=1, initial=0.5, 
        label="Coefficient equal or higher than",
        widget = forms.NumberInput(attrs={'id': 'form_high_coef', 'step': "0.1"}))

    survival = forms.BooleanField(label="Get log hazard ratio for Gene and miRNA", required=False)

    ##DB QUERY
    join = forms.ChoiceField(choices=(("or","OR"),("and","AND")), label="Filter method to target Databases",required=False)

    DB_CHOICES = tuple(zip(load_matrix_header(),load_matrix_header()))

    nDB = forms.MultipleChoiceField(choices=DB_CHOICES,label="Minimum number of Predictions Tools", required=False, \
        widget=forms.CheckboxSelectMultiple(attrs={
                "checked": "",
                "class": "form-check form-check-inline"
            }))
    min_db = forms.IntegerField(initial=5, max_value=40, min_value=0)        


class AllCorrelationForm(forms.Form):
    """
    Form to get the Correlation data for the analysis.

    Ref:
    https://stackoverflow.com/questions/50960800/searchable-drop-down-for-choice-field-in-django-admin
    https://stackoverflow.com/questions/57143113/django-how-to-use-the-admin-autocomplete-field-in-a-custom-form
    """

    ### Data From DB###
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(AllCorrelationForm, self).__init__(*args, **kwargs)

        ### Select available dataset ###
        try:
            DATASET_CHOICES = list(Dataset.objects.filter(public=True).values_list("id","name"))
            #print(DATASET_CHOICES)

            if self.user.is_authenticated:
                DATASET_USER = list(Dataset.objects.filter(user_id=self.user).values_list("id","name"))
                DATASET_CHOICES += DATASET_USER
            DATASET_CHOICES = list(set(DATASET_CHOICES))

        except Exception as error:
            print(error)
            DATASET_CHOICES =[]

        publicDataset = forms.ChoiceField(choices=[(None,"-----"),]+DATASET_CHOICES, label="Select available dataset", required=False)

        self.fields["publicDataset"] = publicDataset



    ### Normalize ###
    survival = forms.BooleanField(label="Get Log Hazard Ratio for gene and miRNA", required=False)

