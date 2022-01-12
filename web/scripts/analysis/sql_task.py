import os
from pandas.io import parsers
import psycopg2
import sqlite3
import numpy as np
import psycopg2.extras as extras
from io import StringIO
import pandas as pd
import sys
import io
from django.core.management import call_command
from django.db import connection
from mirWeb.settings import BASE_DIR, DATA_DIR


param_dic = {
        "host"      : os.environ.get("SQL_HOST", "localhost"),
        "database"  : os.environ.get("SQL_DATABASE", os.path.join(BASE_DIR, "db.sqlite3")),
        "user"      : os.environ.get("SQL_USER", "user"),
        "password"  : os.environ.get("SQL_PASSWORD", "password"),
        'port': os.environ.get("SQL_PORT", "5432")
    }


def update_pk():
    app_name = 'analysis'

    # Get SQL commands from sqlsequencereset
    output = io.StringIO()
    call_command('sqlsequencereset', app_name, stdout=output, no_color=True)
    sql = output.getvalue()
        
    with connection.cursor() as cursor:
        cursor.execute(sql)
    output.close()


def connect(params_dic):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        if "SQL_HOST" in os.environ:
            # connect to the PostgreSQL server
            print('Connecting to the PostgreSQL database...')
            conn = psycopg2.connect(**params_dic)
        else:            
            # connect to the SQLite server
            print('Connecting to the SQLite database...')
            conn = sqlite3.connect(params_dic["database"])

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1) 
    print("Connection successful")
    return conn


def copy_from_stringio(conn, df, table):
    """
    Here we are going save the dataframe in memory 
    and use copy_from() to copy it to the table
    """
    # save dataframe to an in memory buffer
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)
    
    cursor = conn.cursor()
    try:
        cursor.copy_from(buffer, table, sep="\t")
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("copy_from_stringio() done")
    cursor.close()

def parse_file(file_upload):
    print(file_upload)
    try:
        file = file_upload.read().decode('utf-8')
    except:
        file = file_upload.read().decode('ISO-8859-1')
        
    try:
        df = pd.read_table(StringIO(file), delimiter='\t',  dtype={"Prediction Tools":"str"})
    except:
        df = pd.read_table(StringIO(file), delimiter='\t')

    print(df.head())
    return df


####SQL####

def updateSqliteTable(df, table_name):
    try:   
        conn = connect(params_dic=param_dic)
        print("Connected to DB")
    except Exception as error:
        print("Failed to update sqlite table", error)
    else:
        if table_name == "analysis_gene":
            df = df.drop("GENETIC_ENTITY_ID", axis = 1)
            df.columns = ["entrez_id","symbol","gene_type"]
        elif table_name == "analysis_geneset":
            df.columns = ["id","external_id","name","description","ref_link", "public", "user_id_id"]
        elif table_name == "analysis_geneset_genes_id":
            df.columns = ["id","geneset_id", "gene_id"]
        elif table_name == "analysis_gene_synthetic_lethal":
            df.columns = ["from_gene_id", "to_gene_id"]
        elif table_name == "analysis_mirna":
            df.columns = ["id", "mir_type", "mature_acc", "mature_id"]
        elif table_name == "analysis_mirnaset_mirna_id":
            df.columns = ["id","mirnaset_id", "mirna_id"]
        elif table_name == "analysis_target":
            df.columns = ["id","target","number_target", "gene_id_id",  "mirna_id_id", ]
        elif table_name == "reference_prediction_tool":
            df.columns = ["id","name","url", "doi",  "pmid", "reference"]
        try:
            if "SQL_HOST" in os.environ:
                print(df.head())
                try:
                    copy_from_stringio(conn, df, table_name)
                except Exception as error:
                    print(error)
  
            else:
                df.to_sql(name=table_name,if_exists='append', con=conn, index=False)
                conn.close()

            update_pk()
        except Exception as error:
            print(error)
            pass #or any other action

 

    
