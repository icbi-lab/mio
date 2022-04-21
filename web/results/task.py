import pandas as pd
from django.shortcuts import redirect
from analysis.models import File, Workflow, Dataset, Gene
from microrna.models import Mirna_mature
from microrna.models import Target
from mirWeb.settings import DATA_DIR, IMG_ROOT, NUM_THREADS
from pathlib import Path
from miopy import borda_table, load_matrix_header, get_target_query, FilterDF, load_table_counts, load_synthetic



def get_expr(wrkfl):
    meta = DATA_DIR+"/"+ wrkfl.dataset_id.metadataFile.url
    exprs = DATA_DIR+"/"+ wrkfl.dataset_id.exprFile.url

    survival = pd.read_csv(meta, index_col=0)
    exprs = pd.read_csv(exprs, index_col=0)

    return pd.concat([survival,exprs], axis = 1)


def predict_target_corr(table = None, matrix = None, lTarget = None, lTools = None, method = "or", min_db = 10, low_coef = -0.5, high_coef = 0.5, pval = 0.05):

    lTools = lTools if lTools != None else load_matrix_header()

    if len(lTarget) > 0: 
        if table is not None:
            print("Holi")
            table, matrix = FilterDF(table = table, matrix = matrix, join = method, lTool = lTools, \
                low_coef = low_coef, high_coef = high_coef, pval = pval)
                    #Obtain Target Table
            target = table[table["Gene"].isin(lTarget)|table["Mir"].isin(lTarget)]
            
        del table
        #Filter by number
        target["Number Prediction Tools"] = target["Prediction Tools"].str.count("1")
        target = target[target["Number Prediction Tools"] >= min_db]

        if not target.empty and matrix is not None:
            gene = target["Gene"].unique().tolist()#Obtain Unique Gene after filter the table
            mir = target["Mir"].unique().tolist()#Obtain Unique mir after filter the table

            try:
                matrix = matrix.loc[mir,gene]#Subset the Correlation matrix to the heatmap
            except:
                matrix = matrix.loc[gene,mir]#Subset the Correlation matrix to the heatmap
        else:
            matrix = None

    else:
        target, matrix = None, None
    return target, matrix

def predict_target_query(lTarget = None, lTools = None, method = "or", min_db = 10):
    from django.db.models import Q

    lTools = lTools if lTools != None else load_matrix_header()

    if len(lTarget) > 0: 
        lGene = list(Gene.objects.filter(symbol__in = lTarget).values_list("pk"))
        lMir = list(Mirna_mature.objects.filter(mature_name__in = lTarget).values_list("pk",))

        target = pd.DataFrame(Target.objects.filter(Q(gene_id__in = lGene) | Q(mirna_id__in = lMir)).values("id" , "gene_id_id__symbol", "mirna_id_id__mature_name", "target", "number_target"))
                
        if not target.empty:
            target.columns =  ["ID" , "Gene", "Mir", "Prediction Tools", "Number Prediction Tools"]

            #Filter by number
            target = target[target["Number Prediction Tools"] >= min_db]

            matrix = None
    else:
        target, matrix = None, None
    return target, matrix


def ora_mir(lGene, mir_name, q):
    from scipy.stats import fisher_exact
    try:
        mir = Mirna_mature.objects.get(mature_name=mir_name)
    except:
        mir = Mirna_mature.objects.filter(mature_name=mir_name)[0]

    print(mir)
    total_number_gene_universe = 20386
    total_number_gene_list = len(set(lGene))
    temp_list = list(Gene.objects.filter(hgnc_id__in = Target.objects.filter(mirna_id=mir.pk, number_target__gte = q).values_list("gene_id")).values_list("symbol"))
    target_gene_list = [x[0] for x in temp_list]
    target_number_universe = len(set(target_gene_list))
    target_number_list = len(set(lGene).intersection(set(target_gene_list)))
    
    in_list, not_list = target_number_list, total_number_gene_list - target_number_list
    in_universe, not_universe = target_number_universe, total_number_gene_universe - target_number_universe
    
    data = {"List":[in_list, not_list], "Universe": [in_universe, not_universe]}
    print(data)
    res = pd.DataFrame.from_dict(data)
    
    odd, pval = fisher_exact(res, alternative='greater')
    
    expected = (in_universe / total_number_gene_universe) * total_number_gene_list
    
    res = pd.Series([mir_name, target_number_list, expected, odd, pval], \
                     index = ["microRNA","Target Number","Expected Number", "Fold Enrichment", "Raw P-value"], name = mir)
    print(res)
    return res


def ora_mir_list(lMir, lGene, minDB):
    df = pd.DataFrame()
    for mir in lMir:
        res = ora_mir(lGene, mir, minDB)
        df = pd.concat([df,res], axis = 1)
    return df


def ora_mir_parallel(lGene, lMir, minDB, n_core = 2):
    import numpy as np
    import functools
    from statsmodels.stats.multitest import multipletests
    from multiprocessing import Pool
    """    ##Split List
    np_list_split = np.array_split(lMir, n_core)
    split_list = [i.tolist() for i in np_list_split]
    #split_list = same_length(split_list)

    #Fix Exprs Variable 
    partial_func = functools.partial(ora_mir_list, lGene = lGene,  minDB=minDB)

    #Generating Pool
    pool = Pool(n_core)
    lres = pool.map(partial_func, split_list)
    res = pd.concat(lres, axis = 1)
    res = res.transpose()
    pool.close() 
    pool.join()
    """
    res = ora_mir_list(lMir, lGene, minDB)
    res = res.transpose()
    res["FDR"] = multipletests(res["Raw P-value"], method = "fdr_bh")[1]
    return res


def predict_lethality2(table = None, matrix = None, lQuery = None, lTools = None, method = "or", min_db = 10, low_coef = -0.5, high_coef = 0.5, pval = 0.05):
    ##Get Methods##
    import pandas as pd

    slDF = load_synthetic()
    
    qA = slDF.loc[slDF["GeneA"].isin(lQuery),:]
    qA.columns = ["Query", "Synthetic Lethal"]
    qB = slDF.loc[slDF["GeneB"].isin(lQuery),:]
    qB.columns = ["Synthetic Lethal","Query"]

    qSl = pd.concat([qA,qB])
    lTarget = qSl["Synthetic Lethal"].tolist()

    if len(lTarget) > 0:
        if table is not None:
            target,matrix = predict_target_corr(table = table, matrix = matrix, lTarget = lTarget, lTools = lTools, method = method, min_db = min_db, low_coef = low_coef, high_coef = high_coef, pval = pval)
        else:
            target, matrix = predict_target_query(lTarget = lTarget, lTools = lTools, method = method, min_db = min_db)
        
        if target is not None:
            target = pd.merge(qSl,target, left_on="Synthetic Lethal", right_on="Gene")
            print("Probando ORA")
            res = ora_mir_parallel(lGene=lTarget, lMir=target["Mir"].unique().tolist(),minDB=min_db)

    else:
        target = None
        res = None
    
    return target, matrix, res
    