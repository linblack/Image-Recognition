import pyodbc
import pandas as pd
import SystemConfig.ServerConfig as ServerConfig  # Reaver

def DB_Connection(server_name="10.97.36.137", db_name="APC"):
    user = ServerConfig.UID
    password = ServerConfig.PWD
    cnxn1 = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server_name + ';DATABASE=' + db_name + ';UID=' + user + ';PWD=' + password)
    return cnxn1

def select_projectx_10run_by_itime(project_id, last_scantime, max_time, server_name1=ServerConfig.SmartPrediction_DBServer_IP, db_name1=ServerConfig.SmartPrediction_Config_DB):
    cnxn1 = DB_Connection(server_name=server_name1, db_name=db_name1)
    sql = r"select top 10 * from " + str(project_id) + "_RUN where ITIME > '" + last_scantime + r"' AND ITIME < '" + max_time + r"' order by ITIME"
    df_project_model = pd.read_sql(sql, cnxn1)
    return df_project_model

def update_projectx_predict_data_paramvalue_isretrainpredict_by_runindex_parameter_modelid(projectx_predict_data,param_value,is_retrain_predict, runindex,parameter, model_id,server_name1="10.97.36.137",db_name1="APC"):
    cnxn1 = DB_Connection(server_name=server_name1, db_name=db_name1)
    cursor1 = cnxn1.cursor()
    sql = 'UPDATE {} SET PARAM_VALUE = {}, IS_RETRAIN_PREDICT = {}, ITIME = getdate() WHERE RUNINDEX = {} AND PARAMETER = \'{}\' AND MODEL_ID = {}'.format(
        projectx_predict_data + "_PREDICT_DATA", param_value, is_retrain_predict, int(runindex), str(parameter), int(model_id))
    cursor1.execute(sql)
    cnxn1.commit()


def insert_projectx_predict_data_runindex_parameter_paramvalue_modelid_isretrainpredict(projectx_predict_data, runindex,parameter, param_value,model_id, is_retrain_predict,server_name1=ServerConfig.SmartPrediction_DBServer_IP,db_name1=ServerConfig.SmartPrediction_Config_DB):
    cnxn1 = DB_Connection(server_name=server_name1, db_name=db_name1)
    cursor1 = cnxn1.cursor()
    sql = 'INSERT INTO {} (RUNINDEX,PARAMETER,PARAM_VALUE,MODEL_ID,IS_RETRAIN_PREDICT,ITIME) VALUES ({},\'{}\',{},{},{},getdate())'.format(
        projectx_predict_data + "_PREDICT_DATA", int(runindex), str(parameter), param_value, int(model_id),is_retrain_predict)
    cursor1.execute(sql)
    cnxn1.commit()