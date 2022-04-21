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


def update_pk(table):
    app_name = list(table.split("_"))[0]

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


def copy_from_stringio(conn, df, table, columns):
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
        cursor.copy_from(buffer, table, sep="\t", columns = columns)
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
        df = pd.read_table(StringIO(file), delimiter='\t',  dtype={"Prediction Tools":"str"},  header = None, index_col=None)
    except:
        df = pd.read_table(StringIO(file), delimiter='\t', header = None, index_col=None)

    print(df.head())
    return df


####SQL####
def try_int(x):
    try:
        x = pd.Int64Dtype(x)
    except:
        x = x

    return x

def updateSqliteTable(df, table_name):
    try:   
        conn = connect(params_dic=param_dic)
        print("Connected to DB")
        #df.loc[int(len(df))+1,:] = df.columns.tolist()
    except Exception as error:
        print("Failed to update sqlite table", error)
    else:
        ##Gene
        if table_name == "gene_gene":
            df.columns = ["hgnc_id","symbol","approved_name","status","previus_symbol",\
                 "alias_symbols", "chromosome","locus_type","ncbi_gene_id","ensembl_gene_id"]
            df["hgnc_id"] = df["hgnc_id"].apply(int)
            df = df.astype({"ncbi_gene_id": pd.Int64Dtype()}, errors='ignore')
            print(df["ncbi_gene_id"].apply(lambda x: try_int(x)))
        elif table_name == "gene_geneset":
            df.columns = ["id","external_id","name","description","ref_link", "public", "user_id_id"]
            df["id"] = df["id"].apply(int)
            df["user_id_id"] = df["user_id_id"].apply(int)

        elif table_name == "gene_geneset_genes_id":
            df.columns = ["id","geneset_id", "gene_id"]
            df["id"] = df["id"].apply(int)
            df["geneset_id"] = df["geneset_id"].apply(int)
            df["gene_id"] = df["gene_id"].apply(int)

        elif table_name == "genegene_synthetic_lethal":
            df.columns = ["from_gene_id", "to_gene_id"]
        ##microRNA
        elif table_name == "microrna_mirna":
            df.columns =  ["auto_mirna","mirna_acc", "mirna_id", \
                            "previous_mirna_id", "description", \
                            "sequence", "comment", "auto_species", "dead_flag"]
            df["auto_mirna"] = df["auto_mirna"].apply(int)
            df["auto_species"] = df["auto_species"].apply(int)
            df["dead_flag"] = df["dead_flag"].apply(int)

        elif table_name == "microrna_mirna_mature":
            df.columns =  ["auto_mature","mature_name", "previous_mature_id", \
                            "mature_acc", "evidence", \
                            "experiment", "similarity", "dead_flag"]
            df["auto_mature"] = df["auto_mature"].apply(int)
            df["dead_flag"] = df["dead_flag"].apply(int)

        elif table_name == "microrna_mirna_prefam":
            df.columns =  ["auto_prefam","prefam_acc", "prefam_id", \
                            "description"]
            df["auto_prefam"] = df["auto_prefam"].apply(int)      

        elif table_name == "microrna_mirna_chromosome_build":
            df.columns =  ["auto_mirna_id","xsome", "contig_start", \
                            "contig_end", "strand"]
            df["contig_start"] = df["contig_start"].apply(int)
            df["contig_end"] = df["contig_end"].apply(int)
            df["auto_mirna_id"] = df["auto_mirna_id"].apply(int)

        elif table_name == "microrna_mirna_context":
            df.columns =  ["auto_mirna_id", "transcript_id", \
                            "overlap_sense", "overlap_type", "number", \
                            "transcript_source", "transcript_name"]
            #df["id"] = df["id"].apply(int)
            df["auto_mirna_id"] = df["auto_mirna_id"].apply(int)

        elif table_name == "microrna_mirna_pre_mature":
            df.columns = ["auto_mirna_id","auto_mature_id", \
                           "mature_from","mature_to"]
            df["auto_mirna_id"] = df["auto_mirna_id"].apply(int)
            df["auto_mature_id"] = df["auto_mature_id"].apply(int)
            df["mature_from"] = df["mature_from"].apply(int)
            df["mature_to"] = df["mature_to"].apply(int)

        elif table_name == "microrna_mirna_prefam_id":
            df.columns = ["auto_mirna_id", "auto_prefam_id"]
            df["auto_mirna_id"] = df["auto_mirna_id"].apply(int)
            df["auto_prefam_id"] = df["auto_prefam_id"].apply(int)

        elif table_name == "microrna_mirnaset_mirna_id":
            df.columns = ["id","mirnaset_id", "mirna_mature_id",]
            df["id"] = df["id"].apply(int)
            df["mirnaset_id"] = df["mirnaset_id"].apply(int)
            df["mirna_mature_id"] = df["mirna_mature_id"].apply(int)

        elif table_name == "microrna_mirnaset":
            df.columns = ["id","name", "description","ref_link", "public", "user_id_id"]
            #print(df.columns.tolist())
            df["id"] = df["id"].apply(int)
            df["user_id_id"] = df["user_id_id"].apply(int)
            #print(df)

        elif table_name == "microrna_target":
            df.columns = ["target","number_target", "gene_id_id",  "mirna_id_id", ]
            df["number_target"] = df["number_target"].apply(int)
            df["gene_id_id"] = df["gene_id_id"].apply(int)
            df["mirna_id_id"] = df["mirna_id_id"].apply(int)

        elif table_name == "reference_prediction_tool":
            df.columns = ["id","name","url", "doi",  "pmid", "reference"]
            df["id"] = df["id"].apply(int)

        try:
            if "SQL_HOST" in os.environ:
                try:
                    print(df.head())
                    copy_from_stringio(conn, df, table_name, df.columns.tolist())
                except Exception as error:
                    print(error)
  
            else:
                df.to_sql(name=table_name,if_exists='append', con=conn, index=False)
                conn.close()

            update_pk(table_name)
        except Exception as error:
            print(error)
            pass #or any other action

 

    
