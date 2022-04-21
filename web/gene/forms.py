from django import forms
from .models import Gene,  Geneset
from django.core.exceptions import ValidationError
from microrna.models import Mirnaset
from mirWeb.settings import CONTENT_TYPES, MAX_UPLOAD_SIZE
from django.forms.widgets import HiddenInput
from mirWeb.settings import BASE_DIR
####FileField####
from django import forms
from django.template.defaultfilters import filesizeformat
from django.utils.translation import ugettext_lazy as _
from django.core.exceptions import ValidationError


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


##### Gene ####

class GeneForm(forms.Form):
    """
    Form to update the Geneset, Gene, Synthetic Lethality, Gene_set_Gene
    in the DB. This tables has the cBioportal format.
    This form is used by GeneUploadView
    """
    TABLE_CHOICES = [
        #Gene
        ("gene_geneset","GeneSet"),
        ("gene_gene","Gene"),
        ("gene_geneset_genes_id", "GeneSet2Genes"),
        ("gene_gene_synthetic_lethal", "Synthetic_lethal"),
        #MicroRNA
        ("microrna_mirna", "miRNA"),
        ("microrna_mirna_mature", "miRNA_mature"),
        ("microrna_mirna_prefam", "miRNA_prefam"),
        ("microrna_mirna_chromosome_build", "Mirna_chromosome_build"),
        ("microrna_mirna_context", "mirna_context"),
        ("microrna_mirna_pre_mature","mirna_pre_mature"),
        ("microrna_target","target"),
        ("microrna_mirna_prefam_id", "microrna_mirna_prefam_id"),
        ("microrna_mirnaset_mirna_id", "microrna_mirnaset_mirna_id"),
        ("microrna_mirnaset", "microrna_mirnaset"),

        #
        ("reference_prediction_tool","Reference")


    ]
    file = forms.FileField()
    table = forms.ChoiceField(choices=TABLE_CHOICES, required=True)


class GenesetForm(forms.ModelForm):
    """
    Form to create a new GeneSet from the user.
    This form is used by CreateGeneSetView
    """
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(GenesetForm, self).__init__(*args, **kwargs)  

        if self.user.is_staff is False:
            self.fields["public"].widget = HiddenInput()



    ID_CHOICE = [("symbol","Symbol"),("entrezid","Gene_ID")]

    file = RestrictedFileField(label="Select geneset File", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)

    format = forms.ChoiceField(choices=ID_CHOICE,label="Select the Gene Identifier")
    
    class Meta:
        model = Geneset
        exclude = ["external_id", "genes_id","user_id"]


class GenesetGMTForm(forms.Form):
    """
    Form to update the Geneset, Gene, Synthetic Lethality, Gene_set_Gene
    in the DB. This tables has the cBioportal format.
    This form is used by GeneUploadView
    """
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(GenesetGMTForm, self).__init__(*args, **kwargs)  

        if self.user.is_staff is False:
            self.fields["public"].widget = HiddenInput()
    
    ID_CHOICE = [("symbol","Symbol"),("entrezid","Entrez_ID")]     
    file = forms.FileField()
    geneFormat = forms.ChoiceField(choices=ID_CHOICE,label="Select the Gene Identifier")
    public = forms.ChoiceField(choices= [(False, "No"),(True,"Yes")], initial=False)


class MirnasetForm(forms.ModelForm):
    """
    Form to create a new GeneSet from the user.
    This form is used by CreateGeneSetView
    """
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(MirnasetForm, self).__init__(*args, **kwargs)  

        if self.user.is_staff is False:
            self.fields["public"].widget = HiddenInput()



    ID_CHOICE = [("accesion","Accesion (MIMATXXXXXX)"),("id","Id (hsa-miR-XXXX)")]

    file = RestrictedFileField(label="Select mirnaset file", required=False, content_types=CONTENT_TYPES, max_upload_size=MAX_UPLOAD_SIZE)

    format = forms.ChoiceField(choices=ID_CHOICE,label="Select the miRNA Identifier")
    
    class Meta:
        model = Mirnaset
        exclude = ["mirna_id","user_id"]