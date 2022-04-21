from django.db import models
from registration.models import User

# Create your models here.

class Gene(models.Model): #Gene Data from cBioportal
    """
    Gene table. Genes extracted from the CBioPortal.
    """
    def __str__(self):
        return self.symbol

    hgnc_id = models.PositiveIntegerField(primary_key=True) 
    symbol = models.CharField(max_length=255, null=False)
    approved_name = models.TextField(null=False)
    status = models.CharField(max_length=75, null=False) 
    previus_symbol = models.CharField(max_length=255, null=True, blank=True)
    alias_symbols = models.CharField(max_length=255, null=True, blank=True)
    chromosome = models.CharField(max_length=75, null=False) 
    locus_type = models.CharField(max_length=75, null=False) 
    ncbi_gene_id = models.CharField(max_length=75, null=True, blank = True)
    ensembl_gene_id = models.CharField(max_length=75, null=True, blank = True)
    synthetic_lethal = models.ManyToManyField("self", blank=True)


class Geneset (models.Model):
    """
    Geneset Table.
    """
    def __str__(self):
        return self.name
    
    def to_txt(self, gname = "symbol"):
        lFields = [self.name,">"+self.description,">"+self.ref_link]
        lFields += [str(gene) for gene in self.get_genes(gname)]
        txt = "\n".join(lFields)
        return txt

    def to_gmt(self, gname = "symbol"):
        lFields = [self.name,self.description+" "+self.ref_link,"\t".join([str(gene) for gene in self.get_genes(gname)])]
        txt = "\t".join(lFields)
        return txt 

    def get_genes(self, gname="symbol"):# Obtain a gene list to Geneset
        return list(self.genes_id.values_list(gname, flat=True))

    def get_number_genes(self):
        """
        Function to get the number of genes from geneset
        Return:
            Int
        """
        return len(self.get_genes())

    def check_geneset(self):
        if self.get_number_genes() == 0:
            #self.delete()
            return False
        else:
            return True


    def create_geneset(self, name, description, ref, user_slug, public):
        self.external_id = name
        self.name = name
        self.description = description
        self.ref_link = ref
        self.user_id = User.objects.get(identifier=user_slug)
        self.public = public
        self.save()

    def from_form(self, name = None, description = None, ref = None, lFeature = None, identifier = None, public = False, user_slug = None):
        self.create_geneset(name, description, ref, user_slug, public)
        #file = file.read().decode('utf-8')
        #df = pd.read_table(io.StringIO(file), delimiter='\t',names=["Gene",])
        
        try:
            if identifier == "symbol":
                genes = Gene.objects.filter(symbol__in=lFeature)
                self.genes_id.add(*genes)
            else:
                genes = Gene.objects.filter(pk__in=lFeature)
                self.genes_id.set(*genes)


        except Exception as error:
            self.delete()
            print(error)
            return False

        else:
            self.save() if self.check_geneset() else self.delete()

    def is_owner(self, user):
        return True if self.user_id == user else False

    external_id = models.CharField(max_length=500)
    name = models.CharField(max_length=500, unique=True)
    description =  models.TextField(max_length=900000, null=True)
    ref_link = models.TextField(max_length=900000)
    public = models.BooleanField(choices =  [(False, "No"),(True,"Yes")], default=False)
    genes_id = models.ManyToManyField(Gene) #Relation Geneset-Gene
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
