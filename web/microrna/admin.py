from django.contrib import admin
from .models import Mirna, Mirna_chromosome_build, Mirna_context, Mirna_mature, \
                    Mirna_pre_mature, Mirna_prefam, Mirna_prefam_id, Mirnaset, Target
# Register your models here.

class MirnasetAdmin(admin.ModelAdmin):
    search_fields = ("name",)
    search_fields = ["auto_mirna", "mirna_acc", "mirna_id", "previous_mirna_id"]

class Mirna_pre_matureAdmin(admin.ModelAdmin):
    search_fields = ("name",)
    search_fields = ["auto_mirna__mirna_id", "auto_mature__mature_name"]

class Mirna_matureAdmin(admin.ModelAdmin):
    search_fields = ("auto_mature","mature_name","mature_acc")

class Mirna_prefam_idAdmin(admin.ModelAdmin):
    search_fields = ("auto_mirna__mirna_id","auto_prefam__auto_prefam")

# Register your models here.
admin.site.register(Mirna, MirnasetAdmin)
admin.site.register(Mirna_chromosome_build)
admin.site.register(Mirna_context)
admin.site.register(Mirna_mature, Mirna_matureAdmin)
admin.site.register(Mirna_pre_mature, Mirna_pre_matureAdmin)
admin.site.register(Mirna_prefam)
admin.site.register(Mirnaset)
admin.site.register(Mirna_prefam_id,Mirna_prefam_idAdmin)
admin.site.register(Target)