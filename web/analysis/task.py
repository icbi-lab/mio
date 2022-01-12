from mirWeb.settings import BASE_DIR
from scripts.analysis.workflow_task import (run_correlation, 
                                    run_feature, run_feature_ratio,
                                    run_feature_survival, run_feature_survival_ratio,
                                    run_classification)

from scripts.analysis.sql_task import updateSqliteTable, parse_file
import django_rq
from analysis.models import Queue

def getCorrelation(workflow, method = "Correlation", FilterChoice = "NF", normal = True, logfc = 1.2, pval = 0.005, survival = False, group = "event", \
                     filter_sample = False, group_sample = "event", filter_group = "0"):
    
    try:
        workflow.set_status(1)
        table = run_correlation(workflow, method = method , FilterChoice = FilterChoice, normal = normal, logfc = logfc, \
            pval = pval, survival = survival, group = group, filter_sample = filter_sample, group_sample = group_sample, filter_group=filter_group)
        print(table.head())
        workflow.set_status(2)
        
    except Exception as excp:
        print("Error %s"%excp)
        workflow.set_log(excp)
        workflow.set_status(3)


def getFeatures(workflow, topk = 100, k = 2, normal = False, group = "event",feature = "all", filter_sample = False, group_sample = "event", filter_group = "0"):
    try:
        workflow.set_status(1)
        run_feature(workflow, topk = topk, k = k, normal = False, group = group, feature=feature, filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group)
        workflow.set_status(2)
        
    except Exception as excp:
        print("Error %s"%excp)
        workflow.set_log(excp)
        workflow.set_status(3)



def getFeaturesRatio(workflow, topk = 100, k = 2, normal = False, group = "event", filter_sample = False, group_sample = "event", filter_group = "0", filter_pair = False, low_coef = 0.5, min_db = 20):
    try:
        workflow.set_status(1)
        run_feature_ratio(workflow, topk = topk, k = k, normal = False, group = group, filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group,\
            filter_pair = filter_pair, low_coef = low_coef, min_db = min_db)
        workflow.set_status(2)
        
    except Exception as excp:
        print("Error %s"%excp)
        workflow.set_log(excp)
        workflow.set_status(3)


def getSurvivalFeatures(workflow, topk = 100, k = 2, normal = False, group = "event",feature = "all", filter_sample = False, group_sample = "event", filter_group = "0"):
    try:
        workflow.set_status(1)
        run_feature_survival(workflow, topk = topk, k = k, normal = False, group = group, feature=feature, filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group)
        workflow.set_status(2)
        
    except Exception as excp:
        print("Error %s"%excp)
        workflow.set_log(excp)
        workflow.set_status(3)           



def getSurvivalFeaturesRatio(workflow, topk = 100, k = 2, normal = False, group = "event", filter_sample = False, group_sample = "event", filter_group = "0",  filter_pair = False, low_coef = 0.5, min_db = 20):
    try:
        workflow.set_status(1)
        run_feature_survival_ratio(workflow, topk = topk, k = k, normal = False, group = group, filter_sample = filter_sample, group_sample= group_sample, filter_group = filter_group,\
              filter_pair = filter_pair, low_coef = low_coef, min_db = min_db)
        workflow.set_status(2)
        
    except Exception as excp:
        print("Error %s"%excp)
        workflow.set_status(3)           


def getClassification(workflow, model = "rf", k = 10, normal = False, group = "event", feature = "all", use_fit_model = False, pk = 0):
    try:
        workflow.set_status(1)
        run_classification(workflow, model = model, k = k, normal = normal, group = group, feature = feature, use_fit_model = use_fit_model, pk = pk)
        workflow.set_status(2)
        
    except Exception as excp:
        print("Error %s"%excp)
        workflow.set_status(3) 

        
###Queue###
def QueueSqlite(df, table_name):
    queue = django_rq.get_queue('faster')
    queue.enqueue(updateSqliteTable, df, table_name)

def QueueCorrelation(workflow, method = "Correlation", FilterChoice = "NF", normal = True, logfc = 1.2, pval = 0.005, survival = False, group = "event", \
                     filter_sample = False, group_sample = "event", filter_group = "0"):
    queue = django_rq.get_queue('normal')              
    job = queue.enqueue(getCorrelation, workflow=workflow, method = method, FilterChoice = FilterChoice, normal = normal, \
        logfc = logfc, pval = pval, survival = survival, group = group, filter_sample = filter_sample, group_sample = group_sample, filter_group=filter_group)

    Queue(job_id=job.id, workflow_id= workflow).save()


def QueueSurvivalFeature(workflow, topk = 100, k = 2, normal = False, group = "event", feature = "all", filter_sample = False, group_sample = "event", filter_group = "0"):
    queue = django_rq.get_queue('slow')
    job = queue.enqueue(getSurvivalFeatures, workflow=workflow, topk = topk, k = k, normal = False, group = group, feature=feature, filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group)

    Queue(job_id=job.id, workflow_id= workflow).save()

def QueueSurvivalFeatureRatio(workflow, topk = 100, k = 2, normal = False, group = "event", filter_sample = False, group_sample = "event", filter_group = "0", filter_pair = False, low_coef = 0.5, min_db = 20):
    queue = django_rq.get_queue('slow')
    job = queue.enqueue(getSurvivalFeaturesRatio, workflow=workflow, topk = topk, k = k, normal = False, group = group, filter_sample = filter_sample, group_sample=group_sample,filter_group=filter_group, \
        filter_pair = filter_pair, low_coef = low_coef, min_db = min_db)
    Queue(job_id=job.id, workflow_id= workflow).save()



def QueueFeature(workflow, topk = 100, k = 2, normal = False, group = "event",feature = "all", filter_sample = False, group_sample = "event", filter_group = "0"):
    queue = django_rq.get_queue('slow')
    job = queue.enqueue(getFeatures, workflow=workflow, topk = topk, k = k, normal = False, group = group,feature=feature, filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group)
    Queue(job_id=job.id, workflow_id= workflow).save()

def QueueFeatureRatio(workflow, topk = 100, k = 2, normal = False, group = "event", filter_sample = False, group_sample = "event", filter_group = "0",filter_pair = False, low_coef = 0.5, min_db = 20):
    queue = django_rq.get_queue('slow')
    job = queue.enqueue(getFeaturesRatio, workflow=workflow, topk = topk, k = k, normal = normal, group = group, filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group,\
        filter_pair = filter_pair, low_coef = low_coef, min_db = min_db)
    Queue(job_id=job.id, workflow_id= workflow).save()

def QueueClassification(workflow, model = "rf", k = 10, normal = False, group = "event", feature = "all", use_fit_model = False, pk = 0):
    queue = django_rq.get_queue('faster')
    job = queue.enqueue(getClassification, workflow=workflow, model = model, k = k, normal = normal, group = group, feature = feature, use_fit_model = use_fit_model, pk = pk)
    Queue(job_id=job.id, workflow_id= workflow).save()

"""
def QueueCorrelation(workflow, method = "Correlation", FilterChoice = "NF", normal = True, logfc = 1.2, pval = 0.005, survival = False, group = "event", \
                     filter_sample = False, group_sample = "event", filter_group = "0"):
    getCorrelation(workflow=workflow, method = method, FilterChoice = FilterChoice, normal = normal, \
        logfc = logfc, pval = pval, survival = survival, group = group, filter_sample = filter_sample, group_sample = group_sample, filter_group=filter_group)


def QueueSurvivalFeature(workflow, topk = 100, k = 2, normal = False, group = "event", feature = "all"):
    getSurvivalFeatures(workflow=workflow, topk = topk, k = k, normal = False, group = group, feature=feature)


def QueueSurvivalFeatureRatio(workflow, topk = 100, k = 2, normal = False, group = "event"):
    getSurvivalFeaturesRatio(workflow=workflow, topk = topk, k = k, normal = False, group = group)

def QueueFeature(workflow, topk = 100, k = 2, normal = False, group = "event",feature = "all"):
    getFeatures(workflow=workflow, topk = topk, k = k, normal = False, group = group,feature=feature)

def QueueFeatureRatio(workflow, topk = 100, k = 2, normal = False, group = "event"):
    getFeaturesRatio(workflow=workflow, topk = topk, k = k, normal = False, group = group)

def QueueClassification(workflow, model = "rf", k = 10, normal = False, group = "event", feature = "all"):
    getClassification(workflow=workflow, model = model, k = k, normal = normal, group = group, feature = feature)
"""