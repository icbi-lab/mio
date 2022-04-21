from django.contrib import admin
from django.db.models import Q
from .models import Session, Workflow, File, Dataset, Queue

class DatasetAdmin(admin.ModelAdmin):
    search_fields = ("name",)


class SessionAdmin(admin.ModelAdmin):
    search_fields = ("name",)


""" 

    def get_search_results(self, request, queryset, search_term):
        print(queryset)
        if 'workflow' in request.META.get('HTTP_REFERER', ''):
            limit_choices_to = Workflow._meta.get_field('geneset_id')
            queryset = queryset.filter(Q(public = True))
            print(queryset)

        return super().get_search_results(request, queryset, search_term)
"""

class WorkflowAdmin(admin.ModelAdmin):
    search_fields = ("name",)
    autocomplete_fields = ["geneset_id", "sesion_id"]


# Register your models here.
admin.site.register(Session, SessionAdmin)
admin.site.register(Workflow, WorkflowAdmin)
admin.site.register(File)
admin.site.register(Queue)
admin.site.register(Dataset, DatasetAdmin)

