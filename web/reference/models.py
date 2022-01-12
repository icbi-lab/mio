from django.db import models

# Create your models here.

class Prediction_tool(models.Model):
    def __str__(self):
        return self.name

    name = models.CharField(max_length=20, unique=True, null=True, blank=True)
    url = models.URLField(null=True, blank=True)
    doi = models.URLField(null=True, blank=True)
    pmid = models.PositiveIntegerField(null=True, blank=True)
    reference = models.CharField(max_length=50, null=True, blank=True)
