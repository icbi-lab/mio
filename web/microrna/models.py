from django.db import models
from registration.models import User
from gene.models import Gene
# Create your models here.

class Mirna_mature(models.Model):
    def __str__(self): # provides a default session string
        return str(self.mature_name)
    auto_mature = models.PositiveIntegerField(primary_key=True)
    mature_name  = models.CharField(max_length=70, null=False)
    previous_mature_id  = models.CharField(max_length=700, null=True)
    mature_acc  = models.CharField(max_length=70, null=False)
    evidence  = models.TextField(null=True)
    experiment  = models.TextField(null=True)
    similarity  = models.TextField(null=True)
    dead_flag = models.BooleanField()

    class Meta:
        indexes = [models.Index(fields=['auto_mature', 'mature_name','mature_acc'])]

class Mirna_prefam(models.Model):
    def __str__(self):
        return self.prefam_id
    auto_prefam = models.PositiveIntegerField(primary_key=True)
    prefam_acc = models.CharField(max_length=70,)
    prefam_id = models.CharField(max_length=70,)
    description = models.TextField(max_length=70,null=True)


class Mirna(models.Model):
    def __str__(self): # provides a default session string
        return str(self.mirna_id)

    def get_family(self):
        return self.prefam_id

    auto_mirna = models.PositiveIntegerField(primary_key=True)
    mirna_acc = models.CharField(max_length=255, null=False)
    mirna_id = models.CharField(max_length=255, null=True)
    previous_mirna_id =  models.CharField(max_length=255, null=True, blank=True)
    description = models.CharField(max_length=255, null=True)
    sequence = models.CharField(max_length=255)
    comment = models.TextField(null=True, blank=True)
    auto_species = models.PositiveBigIntegerField(default=0)
    dead_flag = models.PositiveBigIntegerField()
    prefam_id = models.ManyToManyField(Mirna_prefam, through="Mirna_prefam_id")
    mature_id = models.ManyToManyField(Mirna_mature, through="Mirna_pre_mature")
    class Meta:
        indexes = [models.Index(fields=['auto_mirna', 'mirna_acc','mirna_id'])]


class Mirna_prefam_id(models.Model):
    def __str__(self):
        return f"{self.auto_mirna}:{self.auto_prefam}"
    auto_mirna = models.ForeignKey(Mirna, on_delete=models.CASCADE)
    auto_prefam = models.ForeignKey(Mirna_prefam, on_delete=models.CASCADE)

    class Meta:
        unique_together = (("auto_mirna", "auto_prefam"),)
        indexes = [models.Index(fields=['auto_mirna', 'auto_prefam'])]

class Mirna_chromosome_build(models.Model):
    def __str__(self):
        return f"{self.xsome}:{self.contig_start}-{self.contig_end}" 
    auto_mirna = models.ForeignKey(Mirna, on_delete=models.CASCADE)
    xsome  = models.CharField(max_length=255, null=False)
    contig_start = models.PositiveIntegerField()
    contig_end = models.PositiveIntegerField()
    strand  = models.CharField(max_length=255, null=False)


class Mirna_context(models.Model):
    def __str__(self):
        return f"{self.auto_mirna}:{self.transcript_id}"
    auto_mirna = models.ForeignKey(Mirna, on_delete=models.CASCADE)
    transcript_id  = models.CharField(max_length=70, null=False)
    overlap_sense  = models.CharField(max_length=10, null=False)
    overlap_type  = models.CharField(max_length=30, null=False)
    number = models.PositiveIntegerField()
    transcript_source  = models.CharField(max_length=70, null=False)
    transcript_name  = models.CharField(max_length=70, null=False)


class Mirna_pre_mature(models.Model):
    def __str__(self):
        return f"{self.auto_mirna}:{self.auto_mature}"
    auto_mirna = models.ForeignKey(Mirna, on_delete=models.CASCADE)
    auto_mature = models.ForeignKey(Mirna_mature, on_delete=models.CASCADE)
    mature_from = models.CharField(max_length=20)
    mature_to = models.CharField(max_length=20)

    class Meta:
        unique_together = (("auto_mirna", "auto_mature"),)
        indexes = [models.Index(fields=['auto_mirna', 'auto_mature'])]

class Mirnaset(models.Model): #Gene Data from cBioportal
    """
    Gene table. Genes extracted from the CBioPortal.
    """
    def __str__(self):
        return self.name

    def check_mirset(self):
        if self.get_number_mir() == 0:
            #self.delete()
            return False
        else:
            return True

    def to_txt(self, mirna = "mature_name"):
        lFields = [self.name,">"+self.description,">"+self.ref_link]
        lFields += [str(mir) for mir in self.get_mir(mirna)]
        txt = "\n".join(lFields)
        return txt

    def to_gmt(self, mirna = "mature_name"):
        lFields = [self.name,self.description+" "+self.ref_link,"\t".join([str(mir) for mir in self.get_mir(mirna)])]
        txt = "\t".join(lFields)
        return txt 

    def get_mir(self, mirna="mature_name"):# Obtain a gene list to Geneset
        return list(self.mirna_id.values_list(mirna, flat=True))


    def get_number_mir(self):
        """
        Function to get the number of genes from geneset
        Return:
            Int
        """
        return len(self.get_mir())

    def create_mirset(self, name, description, ref, user_slug, public):
        self.external_id = name
        self.name = name
        self.description = description
        self.ref_link = ref
        self.user_id = User.objects.get(identifier=user_slug)
        self.public = public
        self.save()

    def from_form(self, name = None, description = None, ref = None, lFeature = None, identifier = None, public = False, user_slug = None):
        self.create_mirset(name, description, ref, user_slug, public)
        #file = file.read().decode('utf-8')
        #df = pd.read_table(io.StringIO(file), delimiter='\t',names=["Gene",])
        
        try:
            if identifier == "id":
                mirna = Mirna_mature.objects.filter(mature_name__in=lFeature)
            else:
                mirna = Mirna_mature.objects.filter(mature_acc__in=lFeature)
                            
            self.mirna_id.add(*mirna)



        except Exception as error:
            self.delete()
            print(error)
            return False

        else:
            self.save() if self.check_mirset() else self.delete()

    def is_owner(self, user):
        return True if self.user_id == user else False

    name = models.CharField(max_length=500, unique=True)
    description =  models.TextField(max_length=900000)
    ref_link = models.TextField(max_length=900000)
    public = models.BooleanField(choices =  [(False, "No"),(True,"Yes")], default=False)

    mirna_id = models.ManyToManyField(Mirna_mature) #Relation Geneset-Gene
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)


class Target(models.Model):
    
    def __str__(self):
        return "%s/%s"%(self.mirna_id,self.gene_id)

    gene_id = models.ForeignKey(Gene, on_delete=models.CASCADE)
    mirna_id = models.ForeignKey(Mirna_mature, on_delete=models.CASCADE)
    target = models.CharField(max_length=41)
    number_target = models.PositiveIntegerField()

    class Meta:
        unique_together = (("gene_id", "mirna_id"),)
        indexes = [models.Index(fields=['gene_id', 'mirna_id'])]