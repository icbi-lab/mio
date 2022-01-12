from multiprocessing import context
from re import L
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, Http404
import json
from django.views.generic import (View, CreateView)
from miopy import survival
import miopy
from scipy.sparse.sputils import matrix
from .task import get_expr, predict_target_query,predict_target_corr, predict_lethality2
from scripts.results.graph_scripts import plotly_heat, cytoscape_network, module_score_plot, roc_cruve_plotly, feature_weigth, pca_plotly, df_to_kaplan_meier_plot, survival_weigth_plot
import pandas as pd
from analysis.models import File, Geneset, Mirnaset, Workflow, Session, Dataset
from analysis.forms import KaplanMeierForm
from miopy.correlation import FilterDF, intersection, merge_tables, borda_table
from miopy.survival import hazard_ratio, split_by_exprs, get_exprs_cutoff
from mirWeb.settings import DATA_DIR, BASE_DIR
import networkx as nx
import pickle
import numpy as np
import uuid
import logging

# Create a logger for this file
logger = logging.getLogger(__file__)
#### Feature Results #####

def FeatureListJson(request, identifier):
    """
    Function that transform a DF in JsonResponse
    View list to populate the Html Table.
    The Datatable work with Ajax. We need a Json file.
    """

    #Read DF

    #Tranform DF to Json
    result = request.session[identifier]
    json_dict = {}
    json_dict["data"] = json.loads(result)
    
    return JsonResponse(json_dict, safe = False)


class FeatureDetailView(View):
    """
    View to plot the result of the filter DF
    """
    template_name = 'results/feature_view.html'
    
    def get(self,request, session_slug, pk):
        listCol = ["Feature", "Percentage", "Random Forest", 
            "Logistic Regresion", "Ridge Classfier",
            "Support Vector Machine Classfier", "Ada Classifier"
            ,"Bagging Classifier",]

        # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        #Create Dictionary
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            wrfkl = Workflow.objects.get(pk=pk) #Get Workflow
            
            fileTop = File.objects.get(workflow_id = pk, type = "Topfeature")
            fileAll = File.objects.get(workflow_id = pk, type = "Allfeature")

            context = {'workflow': wrfkl}

        except Exception as error:
            print(error)

        else:
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session

            #Get File


            #Get Df
            top =  pd.read_csv(fileTop.get_path(), index_col=0)
            top.columns = ["Percentage"]
            print(top)
            lTop = top.index.tolist()
            matrix = pd.read_csv(fileAll.get_path(), index_col=0)
            
            table = pd.concat([top, matrix.transpose()], axis = 1)

            #Get Columns
            context["col"] = listCol

            #Filter top Feature
            df = table.loc[lTop,:].round(2)
            df["Feature"] = df.index.tolist()

            #Get Json File
            result = df[listCol].sort_values("Percentage", ascending = False).to_json(orient='values')
            identifier = str(uuid.uuid4())
            context["identifier"] = identifier
            request.session[identifier]=result

            try:
                #Get Heatmap       
                context["plotly_heat"] = plotly_heat(matrix[lTop].fillna(0), colorscale = 'YlGnBu', zmin = 0, zmax=100, zmid = 50)

            except Exception as error:
                print(error) 
                
            try:
                context["plotly_pca"] = pca_plotly(wrfkl.dataset_id.get_expr(classifier=True, group = wrfkl.group_data), df["Feature"], wrfkl.group_data)

            except Exception as error:
                print(error)
            return render(request, self.template_name, context)




class ClassificationView(View):
    """
    View to plot the result of the filter DF
    """
    template_name = 'results/classification_view.html'
    
    def get(self,request, session_slug, pk):

        # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        #Create Dictionary
        try:
            
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            wrfkl = Workflow.objects.get(pk=pk) #Get Workflow
            
            results = File.objects.get(workflow_id = pk, type = "Pickle", description = "Classification")

            context = {'workflow': wrfkl}
        except Exception as error:
            print(error)

        else:
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session

            #Get Df
            results = pickle.load( open(results.get_path(), "rb" ) )            
            context["score"] = tuple(zip(results["train"], results["test"]))
            context["test_mean"] = np.mean(results["test"])
            context["model"] = results["model"]
            context["use_fit_model"] = results["use_fit_model"]

            try:
                df = results["feature"].mean(axis=1)
                df.index.name = "Feature"
                df.name = "Feature Importance"  
            except:
                df = results["feature"]
                pass 
            try:
                #Get plot       
                context["plotly_roc"] = roc_cruve_plotly(results)

            except Exception as error:
                print("#############################")
                print(error) 

            try:
                #Get plot
  
                print(df)  
                context["plotly_feature"] = feature_weigth(df.sort_values(ascending=True))

            except Exception as error:
                print(error)

            try:
                context["plotly_pca"] = pca_plotly(wrfkl.dataset_id.get_expr(classifier=True, group = wrfkl.group_data), df.index.tolist(), wrfkl.group_data)

            except Exception as error:
                print(error)
            print(session)
            return render(request, self.template_name, context)


class SurvivalDetailView(View):
    template_name = 'results/feature_view.html'
    
    def get(self,request, session_slug, pk):
        listCol = ["Percentage", "Gradient Boosted Models", "Support Vector Machine", 
            "Penalized Cox"]

        # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        #Create Dictionary
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            wrfkl = Workflow.objects.get(pk=pk) #Get Workflow
            
            fileTop = File.objects.get(workflow_id = pk, type = "Topfeature")
            fileAll = File.objects.get(workflow_id = pk, type = "AllfeaturePercentage")
            fileWeight = File.objects.get(workflow_id = pk, type = "AllfeatureWeight")

        except Exception as error:
            print(error)

        else:
            context = dict()
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            context['workflow'] = wrfkl


            #Get File


            #Get Df
            top =  pd.read_csv(fileTop.get_path(), index_col=0)
            top.columns = ["Percentage"]
            print(top)
            lTop = top.index.tolist()
            matrixPercentage = pd.read_csv(fileAll.get_path(), index_col=0)
            matrixWeight = pd.read_csv(fileWeight.get_path(), index_col=0)
            print(matrixWeight)
            table = pd.concat([top, matrixWeight], axis = 1)
            print(table)
            table = table.fillna(0)

            #Filter top Feature
            print(lTop)
            print(listCol)

            df = table.loc[lTop,:].round(2)
            df["Feature"] = table.loc[lTop,:].index.tolist()
            df = borda_table(df, lMethod = listCol)
            print(df)

            #Get Columns
            listCol = ["Ranking","Feature"] + listCol
            context["col"] = listCol

            #Get Json File
            result = df[listCol].sort_values("Ranking", ascending = False).to_json(orient='values')
            identifier = str(uuid.uuid4())
            context["identifier"] = identifier
            request.session[identifier]=result

            try:
                #Get Heatmap       
                context["plotly_heat"] = plotly_heat(matrixPercentage[lTop].fillna(0), zmin = 0, zmax = 100, zmid = 50, colorscale = 'YlGnBu')

            except Exception as error:
                print(error) 

            try:
                context["plotly_pca"] = pca_plotly(wrfkl.dataset_id.get_expr(classifier=True, group_sample = wrfkl.group_data), df["Feature"], wrfkl.group_data)

            except Exception as error:
                print(error)

            try:
                #Get Horizontal bar plot
                context["plotly_weigth"] = survival_weigth_plot(matrix = matrixWeight.loc[lTop,:])
            except Exception as error:
                print(error)
            return render(request, self.template_name, context)


#### Kaplain Meier

   
class KaplanMeierFormView(CreateView):
    template_name = 'results/km_formview.html'

    def get(self, request, session_slug):
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            print(session)
            form = KaplanMeierForm(user=request.user)
            context = {'form': form}
            context["title"] = "Kaplan Meier Plot Analysis"

        except:
            raise Http404('Session not found...!')

        else:
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            return render(request, self.template_name, context)

    def post(self, request, session_slug):
        # We get the filter condition
        form = KaplanMeierForm(request.POST, user=request.user)

        if form.is_valid():
            # We add the workflow id and all the filter dict in the session cache
            print(form.cleaned_data)
            request.session["filter_dict"] = form.cleaned_data

            # Redirect the user to results view
            return redirect('survival_km', session_slug=session_slug)

        return render(request, self.template_name, {'form': form})



class KaplanMeierView(View):
    """
    View to plot the result of the filter DF
    """
    template_name = 'results/km_view.html'
    
    def get(self,request, session_slug):
        # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        #Create Dictionary
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            filter_dict = request.session["filter_dict"]
            session = Session.objects.get(identifier=session_slug)
            dts = Dataset.objects.get(pk=filter_dict["dataset"])
            context = {}

        except Exception as error:
            print(error)
            pass

        else:
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session

            #Filter DF
            try:
                print("Getting Ctoff")
                target, q, optimal = filter_dict["target"], filter_dict["q"], filter_dict["get_cutoff"]
                dfExpr = dts.get_expr(survival=True)
                if "/" in target:
                    mir, gene = target.split("/")
                    dfExpr[target] = dfExpr[mir]/dfExpr[gene]
                    
                q, treshold = get_exprs_cutoff(dfExpr, target=target, q=q, optimal = optimal)
                print(q), print(treshold)
                div = df_to_kaplan_meier_plot(split_by_exprs(dfExpr, target=target, treshold = treshold), q=q, treshold = treshold, target=target, dataset=dts.name)
                context["KM_script"] = div
                
            except Exception as error:
                pass
                print(error)


            return render(request, self.template_name, context)


##### Correlation #####
listCol = ["Ranking","Gene", "Mir", "R", "Rho","Tau","RDC","Hoeffding","Lasso", "Ridge","ElasticNet",\
           "nDB","Log(HR)"]

def CytoListJson(request, identifier):
    """
    Function that transform a DF in JsonResponse
    View list to populate the Html Table.
    The Datatable work with Ajax. We need a Json file.
    """

    #Read DF

    #Tranform DF to Json
    result = request.session[identifier]
    json_dict = {}
    json_dict["data"] = json.loads(result)
    
    return JsonResponse(json.loads(result), safe = False)

class ReultsDetailView(View):
    """
    View to plot the result of the filter DF
    """
    template_name = 'results/result_view.html'
    
    def get(self,request, session_slug, pk):
        listCol = [ "R", "Rho","Tau","RDC","Hoeffding","Lasso", "Ridge","ElasticNet",\
                "Pval", "FDR", "Log(HR)"] 
        # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        #Create Dictionary
        try:
            # Obtain the session from the DB with the session_slug (identifier)
            session = Session.objects.get(identifier=session_slug)
            wrfkl = Workflow.objects.get(pk=pk) #Get Workflow

            context = {'workflow': wrfkl}

        except Exception as error:
            print(error)

        else:
            # We pase the session to the template with the Context Dyct
            context['session_detail'] = session
            dictForm = request.session["filter_dict"]
            #If not we get the data without filter
            file_table = File.objects.get(workflow_id = pk, type = "Correlation")
            file_matrix = File.objects.get(workflow_id = pk, type = "Pearson")
                
            print("Obteniendo enlace")
            table_path = file_table.get_path()#Correlation Table
            matrix_path = file_matrix.get_path()#Correlation Matrix

            #Read DF
            print("Leyendo dataframes")
            table = pd.read_csv(table_path, dtype={"Prediction Tools":"str"}, index_col=0)
            matrix = pd.read_csv(matrix_path, index_col=0)
            #Filter DF
            if wrfkl.analysis == "GeneSetScore":
                df, matrix = FilterDF(table = table, matrix = matrix, join = dictForm["join"], lTool = dictForm["nDB"], \
                    low_coef = dictForm["low_coef"], high_coef = dictForm["high_coef"], pval = dictForm["pval"],  analysis = "GeneSetScore")
            else:
                df, matrix = FilterDF(table = table, matrix = matrix, join = dictForm["join"], lTool = dictForm["nDB"], \
                    low_coef = dictForm["low_coef"], high_coef = dictForm["high_coef"], pval = dictForm["pval"],  analysis = "Correlation", min_db = dictForm["min_db"])
            #print(matrix)
            
            #Get Columns
            listCol = intersection(listCol, df.columns.tolist())
            listCol = ["Ranking","Gene","Mir"] + listCol
            context["col"] = listCol

            #Get Json File
            result = df[listCol].sort_values("Ranking").to_json(orient='values')
            identifier = str(uuid.uuid4())
            context["identifier"] = identifier
            request.session[identifier]=result
            filter_dict = request.session["filter_dict"]
            print(listCol)

            if filter_dict["survival"]:
                try:
                    log_hr = hazard_ratio(exprDF=get_expr(wrfkl), lMirUser=df["Mir"].tolist(), lGeneUser=df["Gene"].tolist())
                    log_hr.index = log_hr["target"].tolist()
                    context["hr"] = True
                    print("HR Calculated")
                except Exception as error:
                    print(error)
                    log_hr = None

            else:
                log_hr = None

            try:
                #Get Heatmap       
                context["plotly_heat"] = plotly_heat(matrix.fillna(0))
            except Exception as error:
                print(error) 

            try:
                ##Get Net
                if wrfkl.analysis == "GeneSetScore":
                    print("Hola")
                    dfGS = File.objects.get(workflow_id = wrfkl.pk, type = "Pickle")
                    print(dfGS.get_path())
                    dfDict = pickle.load( open(dfGS.get_path(), "rb" ) )
                    print(dfDict)
                    context["plot_div"] = module_score_plot(dfDict = dfDict, ExprDf = wrfkl.dataset_id.get_expr(), lMir=df["Mir"].unique().tolist())
                    method = None
                else:
                    elements, stylesheet = cytoscape_network(df, log_hr =  log_hr)
                    identifier_data = str(uuid.uuid4())
                    identifier_style = str(uuid.uuid4())
                    request.session[identifier_data] = json.dumps(elements)
                    request.session[identifier_style] = json.dumps(stylesheet)
                    context["identifier_data"] = identifier_data
                    context["identifier_style"] = identifier_style



            except Exception as error:
                
                print(error)
                pass

            return render(request, self.template_name, context)



class CytoscapeView(View):
    """
    View to plot the result of the filter DF
    """
    template_name = 'results/cytoscape.html'
    
    def get(self,request, session_slug, identifier_data, identifier_style):
        try:
            session = Session.objects.get(identifier=session_slug)
        except:
            pass
        else:
            context = {}
            context["session_slug"] = session
            context["identifier_data"] = identifier_data
            context["identifier_style"] = identifier_style

        return render(request, self.template_name, context)



### Synthetic Lethal ###
class ReultsSyntheticView(View):
    """
    View to plot the result of the filter DF
    """
    template_name = 'results/result_view.html'

    def get(self,request, session_slug): 
        listCol = ["Rho","R","RDC","Hoeffding","Tau","Lasso", "Ridge","ElasticNet",\
                "nDB", "Log(HR)"]


        try:
            #Col to the table
            context = {}
            context["session_detail"] = Session.objects.get(identifier=session_slug)
            #Obtain the filter file to work

            dictFilter = request.session["filter_dict"]
            print("Leyendo diccionariio")
            pk, tQuery, use_set, use_correlation,  publicGeneset,  = dictFilter["table"], dictFilter["tQuery"], dictFilter["use_set"], dictFilter["use_correlation"], dictFilter["publicGeneset"]
            pval, low_coef, high_coef, join, nDB, min_db = dictFilter["pval"], dictFilter["low_coef"], dictFilter["high_coef"], dictFilter["join"], dictFilter["nDB"], dictFilter["min_db"]
            print("Leido diccionariio")

            lQuery = []
            if use_set:
                if publicGeneset != []:
                    lGeneset = Geneset.objects.filter(pk__in=publicGeneset)
                    for gs in lGeneset:
                        lQuery += list(gs.get_genes()) 
            else:
                lQuery.append(tQuery)
            
            print("Obteina lista genes")
            if use_correlation:
                #If not we get the data without filter
                file_table = File.objects.get(workflow_id = pk, type = "Correlation")
                file_matrix = File.objects.get(workflow_id = pk, type = "Pearson")
                
                print("Obteniendo enlace")
                table_path = file_table.get_path()#Correlation Table
                matrix_path = file_matrix.get_path()#Correlation Matrix

                #Read DF
                print("Leyendo dataframes")
                table = pd.read_csv(table_path, dtype={"Prediction Tools":"str"}, index_col=0)
                matrix = pd.read_csv(matrix_path, index_col=0)
            else:
                table = None
                matrix = None
                context["min_db"] = True

            
            print("Lanzando prediccion")
            target, matrix, res = predict_lethality2(table = table, matrix = matrix, lQuery = lQuery, lTools = nDB, method = join, min_db = min_db, low_coef = low_coef, high_coef = high_coef, pval = pval)
            print("##########################################")

        except Exception as error:
            print(error)

        else:
            #Read DF
        
            #Filter by number

            #Get Columns
            listCol = intersection(listCol, target.columns.tolist())
            if dictFilter["use_correlation"]:
                listCol = ["Ranking", "Query","Synthetic Lethal","Mir"] + listCol
            else:
                listCol = ["Number Prediction Tools", "Query","Synthetic Lethal","Mir"] + listCol

            context["col"] = listCol
            print(target[listCol])
            #Tranform DF to Json
            result = target[listCol].to_json(orient='values')
            identifier = str(uuid.uuid4())
            context["identifier"] = identifier
            request.session[identifier]=result

            ## ORA RES
            listCol2 =["microRNA","Target Number","Expected Number", "Fold Enrichment", "Raw P-value", "FDR"]

            context["col2"] = listCol2
            #Tranform DF to Json
            print(res.round(3))
            res = res.round(3)
            result = res[listCol2].to_json(orient='values')
            identifier2 = str(uuid.uuid4())
            context["identifier2"] = identifier2
            request.session[identifier2]=result


            if use_correlation:
                try:
                    np.seterr(divide='ignore', invalid='ignore')
                    log_hr = hazard_ratio(exprDF=get_expr(Workflow.objects.get(pk=pk)), lMirUser=target["Mir"].tolist(), lGeneUser=target["Gene"].tolist())
                    log_hr.index = log_hr["target"].tolist()
                    lmethod = ["R",]
                    context["hr"] = True
                    print("HR Calculated")
                except Exception as error:
                    log_hr = None
                    print(error)
            else:
                lmethod = ["Number Prediction Tools",]
                log_hr = None

            try:
                #Get Heatmap       
                context["plotly_heat"] = plotly_heat(matrix.fillna(0))
            except Exception as error:
                print(error) 

            try:
                target["Number Prediction Tools"] = (target["Number Prediction Tools"]) / (40)
                elements, stylesheet = cytoscape_network(target, lmethod=lmethod, log_hr =  log_hr)
                #print(elements)
                identifier_data = str(uuid.uuid4())
                identifier_style = str(uuid.uuid4())
                request.session[identifier_data] = json.dumps(elements)
                request.session[identifier_style] = json.dumps(stylesheet)
                context["identifier_data"] = identifier_data
                context["identifier_style"] = identifier_style

            except Exception as error:
                
                print(error)
                pass
                
            return render(request, self.template_name, context)



#### Target Prediction ####

class TargetView(View):
    """
    View to plot the result of the filter DF
    """
    template_name = 'results/result_view.html'


    def get(self,request, session_slug): # loads the session_detail template with the selected session object loaded as 'instance' and upload associated with that instance loaded from 'form'
        listCol = [ "R", "Rho","Tau","RDC","Hoeffding","Lasso", "Ridge","ElasticNet",\
                "Pval", "Log(HR)", "Prediction Tools"] 
        try:
            #Col to the table
            context = {}
            context["session_detail"] = Session.objects.get(identifier=session_slug)
            #Obtain the filter file to work

            dictFilter = request.session["filter_dict"]
            pk, tQuery, use_set, use_correlation,  publicGeneset, publicMirnaset = dictFilter["table"], dictFilter["tQuery"], dictFilter["use_set"], dictFilter["use_correlation"], dictFilter["publicGeneset"], dictFilter["publicMirnaset"]
            pval, low_coef, high_coef, join, nDB, min_db = dictFilter["pval"], dictFilter["low_coef"], dictFilter["high_coef"], dictFilter["join"], dictFilter["nDB"], dictFilter["min_db"]

            lTarget = []
            if use_set:
                if publicGeneset != []:
                    lGeneset = Geneset.objects.filter(pk__in=publicGeneset)
                    for gs in lGeneset:
                        lTarget += list(gs.get_genes()) 

                if publicMirnaset != []:
                    lMirset = Mirnaset.objects.filter(pk__in=publicGeneset)
                    for gs in lMirset:
                        lTarget += list(gs.get_mir()) 
            else:
                lTarget.append(tQuery)
            
            if use_correlation:
                #If not we get the data without filter
                file_table = File.objects.get(workflow_id = pk, type = "Correlation")
                file_matrix = File.objects.get(workflow_id = pk, type = "Pearson")
                
                print("Obteniendo enlace")
                table_path = file_table.get_path()#Correlation Table
                matrix_path = file_matrix.get_path()#Correlation Matrix

                #Read DF
                print("Leyendo dataframes")
                table = pd.read_csv(table_path, dtype={"Prediction Tools":"str"}, index_col=0)
                matrix = pd.read_csv(matrix_path, index_col=0)
                target, matrix = predict_target_corr(table = table, matrix = matrix, lTarget = lTarget, lTools = nDB, method = join, min_db = min_db, low_coef = low_coef, high_coef = high_coef, pval = pval)
            else:
                target, matrix = predict_target_query(lTarget = lTarget, lTools = nDB, method = join, min_db = min_db)
                context["min_db"] = True
            print("##########################################")
            print(target)

        except Exception as error:
            print(error)

        else:
            #Read DF
        
            #Filter by number

            #Get Columns
            listCol = intersection(listCol, target.columns.tolist())
            if dictFilter["use_correlation"]:
                listCol = ["Ranking","Gene","Mir"] + listCol
            else:
                listCol = ["Number Prediction Tools","Gene","Mir"] + listCol

            context["col"] = listCol
            print(target[listCol])
            #Tranform DF to Json
            result = target[listCol].to_json(orient='values')
            identifier = str(uuid.uuid4())
            context["identifier"] = identifier
            request.session[identifier]=result

            if use_correlation:
                try:
                    np.seterr(divide='ignore', invalid='ignore')
                    log_hr = hazard_ratio(exprDF=get_expr(Workflow.objects.get(pk=pk)), lMirUser=target["Mir"].tolist(), lGeneUser=target["Gene"].tolist())
                    context["hr"] = True
                    log_hr.index = log_hr["target"].tolist()
                    lmethod = ["R",]
                    print("HR Calculated")
                except Exception as error:
                    log_hr = None
                    print(error)
            else:
                lmethod = ["Number Prediction Tools",]
                log_hr = None

            try:
                #Get Heatmap       
                context["plotly_heat"] = plotly_heat(matrix.fillna(0))
            except Exception as error:
                print(error) 

            try:
                target["Number Prediction Tools"] = (target["Number Prediction Tools"]) / (40)
                elements, stylesheet = cytoscape_network(target, lmethod=lmethod, log_hr =  log_hr)
                identifier_data = str(uuid.uuid4())
                identifier_style = str(uuid.uuid4())
                request.session[identifier_data] = json.dumps(elements)
                request.session[identifier_style] = json.dumps(stylesheet)
                context["identifier_data"] = identifier_data
                context["identifier_style"] = identifier_style

            except Exception as error:
                
                print(error)
                pass

        return render(request, self.template_name, context)
