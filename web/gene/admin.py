from django.contrib import admin
from .models import Gene, Geneset

# Register your models here.
class GeneAdmin(admin.ModelAdmin):
    search_fields = ("symbol",)

class GenesetAdmin(admin.ModelAdmin):
    search_fields = ("name",)
    autocomplete_fields = ["genes_id"]
    
admin.site.register(Gene, GeneAdmin)
admin.site.register(Geneset, GenesetAdmin)