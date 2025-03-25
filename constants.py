################################################################################
############################# Path Variables ###################################
################################################################################

model_output = "model_output"  # model output path

################################################################################
############################# Mlflow Variables #################################
################################################################################

mlflow_artifacts_data = "./mlruns/preprocessing"
mlflow_models_data = "./mlruns/models"
mlflow_models_copy = "./mlruns/models_copy"

artifact_data = "artifacts/"  # path to store mlflow artifacts
profile_data = "profile_data"  # path to store pandas profiles in
sql_path = "./preprocessing/sql_queries/"  # path to store sql queries in
data_path = "data/processed/"

################################################################################
################### Microsoft SQL Server Database Variables ####################
################################################################################

server_var = "UCLAEDRVAL"
database_var = "OHIA_Workspace_FPG_OPHAC"
driver_var = "ODBC Driver 17 for SQL Server"

connection_string_var = (
    f'mssql+pyodbc://@{server_var}/{database_var}?driver={driver_var.replace(" ", "+")}'
)
# SQL file to pass into the query
sql_file = sql_path + "SupportiveCare_CompleteOutput_v5_3.sql"

# Convert the date column to datetime format to avoid
# ValueError: Can't infer object conversion type: 0
run_date_var = "RunDate"  # run date variable
tsh_date_var = "TSHMinDate"  # thyroid stimulating hormone minimum date

# One Hot Encoded Vars to Be Omitted
# Already one-hot encoded by data engineering team
cat_vars = [
    "Sex",
    "SexAssignedAtBirth",
    "GenderIdentity",
    "SexualOrientation",
    "MaritalStatus",
    "EthnicityCategory",
    "RaceCategory",
    "TSHMinDate",
    "PatientLivingStatus",
]

patient_living_status = "PatientLivingStatus"  # --> refers to patient living status
# should never be used as feature b/c leak"

################################################################################
########################## Variable/DataFrame Constants ########################
################################################################################

var_index = "PatientID"  # patient id index
age = "Age"  # age of patient
age_bin = "age_bin"  # bin of ages for stratification only
main_df = "df.parquet"  # main dataframe file name
cols_w_thou_sep = ["VitB12MaxDateValue", "VitB12MinValue"]
string_cols_filename = "column_list.pkl"  # string columns filename for columns
# identified as strings from string_cols.py
bmi_var = "BMI"  # original BMI variable
weight_var = "Weight"  # original weight variable
weight_pounds = "Weight_lbs"  # weight in punds
weight_kilos = "Weight_kg"  # weight in kilograms
height_feet = "Height_ft"  # height in feet

################################################################################
############################ Language ##########################################

preferred_lang = "PreferredLanguage"  # preferred language col
other_lang = "Other"

################################################################################

target_death = [
    "ISDEATHDATElead6mo",
    "ISDEATHDATElead1yr",
]

other_target_death = [
    "ISEPICDEATHlead6mo",
    "ISEPICDEATHlead1yr",
    "ISREGDEATHlead6mo",
    "ISREGDEATHlead1yr",
]

six_mo_outcome = "ISDEATHDATElead6mo"
one_year_outcome = "ISDEATHDATElead1yr"
epic_death = "IsEPICDeath"

death_date_lag = "ISDEATHDATElag"  # lag column search term for death date
death_date_presence = "ISDEATHDATEPRESENT"  # col for if eath date is present

################################################################################
########################## Missingness Percentages #############################

miss_col_thresh = 60  # missingness threshold tolerated for zero-var cols
miss_row_thresh = 0.5  # missingness threshold (rows) tolerated based on dev. set
percent_miss = "percentage_missing"  # new col for percentage missing in rows
miss_indicator = "missing_indicator"  # indicator for percentage missing (0,1)

################################################################################

# The below artificat name is used for preprocessing alone
exp_artifact_name = "preprocessing"
preproc_run_name = "preprocessing"
artifact_run_id = "preprocessing"
artifact_name = "preprocessing"


################################################################################
############################## SHAP Constants ##################################

shap_artifact_name = "explainer"
shap_run_name = "explainer"
shap_artifacts_data = "./mlruns/explainer"
