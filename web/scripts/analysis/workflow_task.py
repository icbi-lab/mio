import pandas as pd
from mirWeb.settings import DATA_DIR, NUM_THREADS
from miopy.correlation import (read_count, concat_matrix, all_methods, tmm_normal, voom_normal, 
                             differential_expression_array, differential_expression_edger, 
                             intersection, gene_set_correlation,ips_correlation,
                             process_matrix_list, header_list, adjust_geneset)
from miopy.feature_selection import feature_selection
from miopy.survival import survival_selection
from miopy.classification import classification_cv, classification_training_model
from analysis.models import File, Session, Workflow, Geneset, Dataset
import numpy as np



def run_correlation(wrkflw, method = "Correlation", FilterChoice = "NF", \
                    normal = True, logfc = 1.2, pval = 0.005, survival = False, group = "event", \
                     filter_sample = False, group_sample = "event", filter_group = "0", background = False):

    exprDf = wrkflw.dataset_id.get_expr(custom_metadata = wrkflw.custom_metadata, normal = normal, survival = survival, filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group)

    lMir, lGene = header_list(exprDf)

    print("Leido los ficheros")

    #Get Genes from GeneSet
    if wrkflw.geneset_id != "":
        print("GS not NONE")
        lGene = wrkflw.get_genes()

    print(lGene)
    
    if FilterChoice != "NF":
        lDEM, lDEG = wrkflw.dataset_id.run_de_analysis(wrkflw=wrkflw, FilterChoice = FilterChoice, custom_metadata = wrkflw.custom_metadata, normal = normal, pval = pval, logfc = logfc, group = group)
        #Intersect Genes
        #lGene = intersection(lDEG,lGene)
        #lMir = intersection(lDEM,lMir)
        lGene = lDEG
        lMir = lDEM


    print("Obtenida Matrix Expression")

    if method == "Correlation":
        ## Obtain correlation
        if bool(wrkflw.dataset_id.corFile):
            table, dfPearson = wrkflw.dataset_id.get_table(lMir, lGene)
        
        else:
            print("Run Correlation")
            table, dfPearson = all_methods(exprDf, lMirUser = lMir, lGeneUser = lGene, n_core = NUM_THREADS, hr = survival, background = background)
            
        table = table.round(4)
        File().set_data(wrkflw, table, "Correlation", True, "GeneCorrelation")
        File().set_data(wrkflw, dfPearson, "Pearson", False, "GeneCorrelation")
    
    elif method == "GeneSetScore":
        lDf = []
        lPval = []
        dfGs = {}
        for gs in wrkflw.geneset_id.all():
            lGenes = list(gs.get_genes())
            print(lGenes)
            CorDf, PvalDf, dfSetScore = gene_set_correlation(exprDf, lGenes, GeneSetName = gs.name, lMirUser = lMir, n_core = NUM_THREADS)   
            print(CorDf)
            lDf.append(CorDf)
            lPval.append(PvalDf)
            dfGs[gs.name] = dfSetScore

        CorDf = pd.concat(lDf, axis=1)
        PvalDf = pd.concat(lPval, axis=1)
        
        lTuple = [(CorDf,"R"),(PvalDf,"Pval")]
        print("Joining")
        table = process_matrix_list(lTuple, add_target=False)
        table = adjust_geneset(table)
        print(table.round(3))

        File().set_data(wrkflw, table, "Correlation", True,"GeneSetCorrelation")
        File().set_data(wrkflw, CorDf, "Pearson", False,"GeneSetCorrelation")
        File().set_pickle(workflow=wrkflw, file = dfGs, ftype = "Pickle", is_result = False, description = "ModuleScore", label = "GeneSetCorrelation_"+wrkflw.label)


    return table



def run_infiltration_correlation(wrkflw,normal = True,  group = "event", lCell = [], \
                     filter_sample = False, group_sample = "event", filter_group = "0"):

    exprDf = wrkflw.dataset_id.get_expr(custom_metadata = wrkflw.custom_metadata, normal = normal, survival = False, \
             filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group)

    lMir, lGene = header_list(exprDf)

    print("Leido los ficheros")

    print("Obtenida Matrix Expression")

    metadata = pd.read_csv(wrkflw.dataset_id.get_metadatapath(),index_col=0)
    metadata = metadata[lCell]
    exprDf = pd.concat((exprDf, metadata), axis = 1).dropna()
    print("Dentro de immuno infiltrate")
    table, dfPearson = all_methods(exprDf, lMirUser = lMir, lGeneUser = lCell, n_core = NUM_THREADS, hr = False, \
        k = 5, background = False, test = False, add_target = False)
    print("cara culo")        
    File().set_data(wrkflw, table, "Correlation", True, "Immunecellinfiltration")
    File().set_data(wrkflw, dfPearson, "Pearson", False, "Immunecellinfiltration")
    
    return table

def run_ips_correlation(wrkflw, method = "Correlation", FilterChoice = "NF", \
                    normal = True, group = "event", \
                     filter_sample = False, group_sample = "event", filter_group = "0"):

    exprDf = wrkflw.dataset_id.get_expr(custom_metadata = wrkflw.custom_metadata, normal = normal, survival = False, filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group)

    lMir, lGene = header_list(exprDf)

    print("Leido los ficheros")


    print("Obtenida Matrix Expression")

    

    table, dfCor, dfIPS = ips_correlation(exprDf, lMirUser = lMir, n_core = NUM_THREADS)   

    File().set_data(wrkflw, table, "Correlation", True,"IpsCorrelation")
    File().set_data(wrkflw, dfCor, "Pearson", False,"IpsCorrelation")
    File().set_pickle(workflow=wrkflw, file = dfIPS, ftype = "Pickle", is_result = False, description = "Immunephenoscore", label = "IpsCorrelation_"+wrkflw.label)


    return table


def run_feature_ratio(wrkflw, topk = 100, k = 10, normal = False, group = "event", filter_sample = False, group_sample = "event", filter_group = "0", filter_pair = False, low_coef = 0.5, min_db = 20):
    exprDf = wrkflw.dataset_id.get_expr(custom_metadata = wrkflw.custom_metadata, normal = normal, \
        survival = True, classifier = True, group = group, ratio = True, lGene = wrkflw.get_genes(), filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group, \
            filter_pair = filter_pair, low_coef = low_coef, min_db = min_db)
 
    print(exprDf)

    print("Obtenida Matrix Expression")

    top, dAll, DictScore = feature_selection(exprDf, k=k, topk=topk, group=group)
    
    File().set_data(wrkflw, top, "Topfeature", True, "FeatureSelection")
    File().set_data(wrkflw, dAll, "Allfeature", False, "FeatureSelection")


    return top, dAll, DictScore 


def run_feature(wrkflw, topk = 100, k = 10, normal = False, group = "event", feature = "all", filter_sample = False, group_sample = "event", filter_group = "0"):
    
    exprDf = wrkflw.dataset_id.get_expr(custom_metadata = wrkflw.custom_metadata, normal = normal, survival = True, classifier = True, group = group, feature = feature, filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group)
 
    print(exprDf)

    print("Obtenida Matrix Expression")

    top, dAll, DictScore = feature_selection(exprDf, k=k, topk=topk, group=group)
    
    File().set_data(wrkflw, top, "Topfeature", True, "FeatureSelection")
    File().set_data(wrkflw, dAll, "Allfeature", False, "FeatureSelection")


    return top, dAll, DictScore 


def run_feature_survival(wrkflw, topk = 100, k = 10, normal = False, group = "event", feature = "all", filter_sample = False, group_sample = "event", filter_group = "0"):
    
    exprDf = wrkflw.dataset_id.get_expr(custom_metadata = wrkflw.custom_metadata, normal = normal, survival = True, feature = feature, group = group, filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group)
 
    print(exprDf)

    print("Obtenida Matrix Expression")

    feature_per, dAll, DictScore, dfTopCoef = survival_selection(exprDf, k=k, topk=topk, event=group, n_core = NUM_THREADS)
    
    File().set_data(wrkflw, feature_per, "Topfeature", True, "FeatureSurvival")
    File().set_data(wrkflw, dAll, "AllfeaturePercentage", False, "FeatureSurvival")
    File().set_data(wrkflw, dfTopCoef, "AllfeatureWeight", False, "FeatureSurvival")

    return feature_per, dAll, DictScore 


def run_classification(wrkflw, model = "rf", k = 10, normal = False, group = "event", feature = "all", use_fit_model = False, pk=0):
    
    exprDf = wrkflw.dataset_id.get_expr(custom_metadata = wrkflw.custom_metadata, normal = normal, survival = False, feature = feature, classifier = True, group = group)
 
    print(exprDf)

    print("Obtenida Matrix Expression")
    print(use_fit_model)
    if use_fit_model is False:
        print("Enter into notr training")
        print(False)
        lFeature = wrkflw.get_genes() + wrkflw.get_mir()
        result = classification_cv(exprDf, k = k, name = model, group = group, lFeature = lFeature)
        result["use_fit_model"] = use_fit_model
        File().set_pickle(workflow=wrkflw, file = result, ftype = "Pickle", is_result = False,description = "Classification", label = "Result_"+wrkflw.label)
        df = result.pop("feature")
        classifier = result.pop("classifier")
        File().set_data(wrkflw, pd.concat([pd.DataFrame(result), df], axis=1), "CSV", True, "Classification")
    
    else:
        print("Enter into classification")
        file = File.objects.get(pk=pk)
        wrkflw.geneset_id.set(file.workflow_id.geneset_id.all())
        wrkflw.mirnaset_id.set(file.workflow_id.mirnaset_id.all())

        model = file.load_pickle()
        print(model)
        print(model.feature_names)
        result = classification_training_model(exprDf, model = model, group = group, lFeature = list(model.feature_names))
        result["use_fit_model"] = use_fit_model
        print(result)
        File().set_pickle(workflow=wrkflw, file = result, ftype = "Pickle", is_result = True, description = "Classification", label = "Result_"+wrkflw.label)
        #df = result.pop("feature")
        #File().set_data(wrkflw, pd.DataFrame(result), "CSV", True, "Classification")

    return result


def run_feature_survival_ratio(wrkflw, topk = 100, k = 10, normal = False, group = "event", filter_sample = False, group_sample = "event", filter_group = "0", filter_pair = False, low_coef = 0.5, min_db = 20):

    exprDf = wrkflw.dataset_id.get_expr(custom_metadata = wrkflw.custom_metadata, normal = normal, survival = True,\
            ratio = True, lGene = wrkflw.get_genes(), filter_sample = filter_sample, group_sample = group_sample, filter_group = filter_group,\
                filter_pair = filter_pair, low_coef = low_coef, min_db = min_db)
 
    print(exprDf)

    print("Obtenida Matrix Expression")

    feature_per, dAll, DictScore, dfTopCoef = survival_selection(exprDf, k=k, topk=topk, event=group, n_core = NUM_THREADS)
    
    File().set_data(wrkflw, feature_per, "Topfeature", True, "FeatureSurvival")
    File().set_data(wrkflw, dAll, "AllfeaturePercentage", False,  "FeatureSurvival")
    File().set_data(wrkflw, dfTopCoef, "AllfeatureWeight", False, "FeatureSurvival")

    return feature_per, dAll, DictScore 


