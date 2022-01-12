# Generated by Django 3.2 on 2021-11-09 11:44

import analysis.models
import django.core.files.storage
from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=250)),
                ('geneFile', models.FileField(storage=django.core.files.storage.FileSystemStorage(location='/media/q053pm/96EC-E493/biotools/docker_compose/web/Data'), upload_to=analysis.models.Dataset.get_session_path)),
                ('mirFile', models.FileField(storage=django.core.files.storage.FileSystemStorage(location='/media/q053pm/96EC-E493/biotools/docker_compose/web/Data'), upload_to=analysis.models.Dataset.get_session_path)),
                ('exprFile', models.FileField(blank=True, null=True, storage=django.core.files.storage.FileSystemStorage(location='/media/q053pm/96EC-E493/biotools/docker_compose/web/Data'), upload_to=analysis.models.Dataset.get_session_path)),
                ('metadataFile', models.FileField(storage=django.core.files.storage.FileSystemStorage(location='/media/q053pm/96EC-E493/biotools/docker_compose/web/Data'), upload_to=analysis.models.Dataset.get_session_path)),
                ('public', models.BooleanField(choices=[(False, 'No'), (True, 'Si')], default=True)),
                ('technology', models.CharField(choices=[('sequencing', 'Sequencing Data'), ('microarray', 'Microarray Data')], max_length=50)),
                ('corFile', models.FileField(blank=True, null=True, storage=django.core.files.storage.FileSystemStorage(location='/media/q053pm/96EC-E493/biotools/docker_compose/web/Data'), upload_to=analysis.models.Dataset.get_session_path)),
                ('pearFile', models.FileField(blank=True, null=True, storage=django.core.files.storage.FileSystemStorage(location='/media/q053pm/96EC-E493/biotools/docker_compose/web/Data'), upload_to=analysis.models.Dataset.get_session_path)),
                ('number_gene', models.PositiveIntegerField(blank=True, null=True)),
                ('number_mir', models.PositiveIntegerField(blank=True, null=True)),
                ('number_sample', models.PositiveIntegerField(blank=True, null=True)),
                ('metadata_fields', models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='File',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('data', models.FileField(max_length=250, storage=django.core.files.storage.FileSystemStorage(location='/media/q053pm/96EC-E493/biotools/docker_compose/web/Data'), upload_to=analysis.models.File.get_session_path)),
                ('label', models.CharField(max_length=70)),
                ('type', models.CharField(max_length=70)),
                ('description', models.CharField(max_length=450)),
                ('is_result', models.BooleanField()),
            ],
        ),
        migrations.CreateModel(
            name='Gene',
            fields=[
                ('entrez_id', models.PositiveIntegerField(primary_key=True, serialize=False)),
                ('symbol', models.CharField(max_length=255)),
                ('gene_type', models.CharField(max_length=75)),
            ],
        ),
        migrations.CreateModel(
            name='Geneset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('external_id', models.CharField(max_length=500)),
                ('name', models.CharField(max_length=500, unique=True)),
                ('description', models.TextField(max_length=900000)),
                ('ref_link', models.TextField(max_length=900000)),
                ('public', models.BooleanField(choices=[(False, 'No'), (True, 'Si')], default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Mirna',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('mir_type', models.CharField(max_length=255)),
                ('mature_acc', models.CharField(max_length=255)),
                ('mature_id', models.CharField(max_length=255, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Mirnaset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=500, unique=True)),
                ('description', models.TextField(max_length=900000)),
                ('ref_link', models.TextField(max_length=900000)),
                ('public', models.BooleanField(choices=[(False, 'No'), (True, 'Si')], default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Session',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('identifier', models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ('public', models.BooleanField(choices=[(0, 'No'), (1, 'Si')], default=0)),
                ('name', models.CharField(max_length=50, unique=True)),
            ],
        ),
        migrations.CreateModel(
            name='Workflow',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('label', models.CharField(max_length=50)),
                ('analysis', models.CharField(max_length=200)),
                ('analysis_type', models.CharField(max_length=200)),
                ('feature_type', models.CharField(blank=True, max_length=50, null=True)),
                ('group_data', models.CharField(blank=True, max_length=50, null=True)),
                ('status', models.PositiveIntegerField(default=0)),
                ('custom_metadata', models.FileField(blank=True, max_length=250, null=True, storage=django.core.files.storage.FileSystemStorage(location='/media/q053pm/96EC-E493/biotools/docker_compose/web/Data'), upload_to=analysis.models.Workflow.get_session_path)),
                ('logs', models.CharField(max_length=300)),
                ('job_id', models.UUIDField(blank=True, null=True, unique=True)),
                ('dataset_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='analysis.dataset')),
                ('geneset_id', models.ManyToManyField(blank=True, to='analysis.Geneset')),
                ('mirnaset_id', models.ManyToManyField(blank=True, to='analysis.Mirnaset')),
                ('sesion_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='analysis.session')),
            ],
        ),
        migrations.CreateModel(
            name='Target',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('target', models.CharField(max_length=41)),
                ('number_target', models.PositiveIntegerField()),
                ('gene_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='analysis.gene')),
                ('mirna_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='analysis.mirna')),
            ],
        ),
    ]
