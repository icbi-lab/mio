from django.db import models
from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid
from django.urls import reverse
from itertools import chain

class User(AbstractUser):
    """
    Mean table of the database
    """
    def get_absolute_url(self): # provides a default if Session is called from views.py without a specified reverse or redirect
        return reverse('session_detail', kwargs={'session_slug':self.identifier})

    def __str__(self): # provides a default session string
        return str(self.username)

    def get_models(self):# Obtain all files to this session
        from analysis.models import File

        lModel = []
        for wrkfl in self.get_workflows():
                lModel.append(wrkfl.get_files())
        lModel = [y.pk for x in lModel for y in x]

        return File.objects.filter(pk__in=lModel, description="Models")

    def get_workflows(self): # Obtain all Workflows to this session
        from analysis.models import Workflow

        lWorkflows = []
        for session in self.get_session():
            lWorkflows.append(session.get_workflows())
        lWorkflows = [y.pk for x in lWorkflows for y in x]

        return Workflow.objects.filter(pk__in=lWorkflows)

    def get_session(self): # Obtain all Workflows to this session
        return self.session_set.all()

    def get_dataset(self):# Obtain all files to this session
        return self.dataset_set.all()

    def get_geneset(self):# Obtain all files to this session
        return self.geneset_set.all()

    def get_mirset(self):# Obtain all files to this session
        return self.mirnaset_set.all()

    def set_identifier(self, id):
        self.identifier = id
        self.save()

    def set_not_temporal(self):
        self.is_temporal = False
        self.save()

    def get_identifier(self):
        return str(self.identifier)

    def create_temporal(self):
        self.is_temporal = True
        self.username = str(uuid.uuid4())
        self.password = uuid.uuid4()
        self.save()

    identifier = models.UUIDField(default = uuid.uuid4, editable = False, unique = True)
    is_temporal = models.BooleanField(default=False)

    #class PrivacyMeta:
    #    fields = ['usernam', 'last_name', "email"]
