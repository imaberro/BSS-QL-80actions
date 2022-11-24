# Utils
# from envs import env
import numpy as np
import configparser

# SQL
import sqlalchemy as db
import json


#Credenciales
config = configparser.ConfigParser()
config.read('db_config.ini')
host = config['postgres']['host']
db_name = config['postgres']['db_name']
port = config['postgres']['port']
user = config['postgres']['user']
pwd = config['postgres']['pass']


# Conexión a la DB de resultados

engine = db.create_engine(f'postgresql://{user}:{pwd}@{host}:{port}/{db_name}')
metadata = db.MetaData()


try: 
    connection = engine.connect()

except db.exc.SQLAlchemyError as e:
    exit(str(e.__dict__['orig']))



datosEjecucion = db.Table('datos_ejecucion', metadata, autoload=True, autoload_with=engine)
insertDatosEjecucion = datosEjecucion.insert().returning(datosEjecucion.c.id)


algorithms = [
'GWO_SCP_80aQL1','WOA_SCP_80aQL1','SCA_SCP_80aQL1'
]


instances = [
'mscp41','mscp42','mscp43','mscp44','mscp45','mscp46','mscp47','mscp48','mscp49','mscp410',
'mscp51','mscp52','mscp53','mscp54','mscp55','mscp56','mscp57','mscp58','mscp59','mscp510',
'mscp61','mscp62','mscp63','mscp64','mscp65',
'mscpa1','mscpa2','mscpa3','mscpa4','mscpa5',
'mscpb1','mscpb2','mscpb3','mscpb4','mscpb5',
'mscpc1','mscpc2','mscpc3','mscpc4','mscpc5',
'mscpd1','mscpd2','mscpd3','mscpd4','mscpd5'
]
runs = 31
population  = 40
maxIter = 1000
beta_Dis = 0.8 #Parámetro de la discretización de RW
ql_alpha = 0.1
ql_gamma =  0.4
policy = "softMax-rulette-elitist" #puede ser 'e-greedy', 'greedy', 'e-soft', 'softMax-rulette', 'softMax-rulette-elitist'
qlAlphaType = "static" # Puede ser 'static', 'iteration', 'visits'
repair = 2 # 1:Simple; 2:Compleja
instance_dir = "MSCP/"
dir_DS_actions = {
                "40aQL1":
                ['S1,Standard', 'S1,Complement', 'S1,Elitist', 'S1,Static', 'S1,ElitistRoulette', 
                'S2,Standard', 'S2,Complement', 'S2,Elitist', 'S2,Static', 'S2,ElitistRoulette', 
                'S3,Standard', 'S3,Complement', 'S3,Elitist', 'S3,Static', 'S3,ElitistRoulette', 
                'S4,Standard', 'S4,Complement', 'S4,Elitist', 'S4,Static', 'S4,ElitistRoulette',
                'V1,Standard','V1,Complement','V1,Static','V1,Elitist','V1,ElitistRoulette',
                'V2,Standard','V2,Complement','V2,Static','V2,Elitist','V2,ElitistRoulette',
                'V3,Standard','V3,Complement','V3,Static','V3,Elitist','V3,ElitistRoulette',
                'V4,Standard','V4,Complement','V4,Static','V4,Elitist','V4,ElitistRoulette'],
                "80aQL1":
                ['S1,Standard', 'S1,Complement', 'S1,Elitist', 'S1,Static', 'S1,ElitistRoulette', 
                'S2,Standard', 'S2,Complement', 'S2,Elitist', 'S2,Static', 'S2,ElitistRoulette', 
                'S3,Standard', 'S3,Complement', 'S3,Elitist', 'S3,Static', 'S3,ElitistRoulette', 
                'S4,Standard', 'S4,Complement', 'S4,Elitist', 'S4,Static', 'S4,ElitistRoulette',
                'V1,Standard','V1,Complement','V1,Static','V1,Elitist','V1,ElitistRoulette',
                'V2,Standard','V2,Complement','V2,Static','V2,Elitist','V2,ElitistRoulette',
                'V3,Standard','V3,Complement','V3,Static','V3,Elitist','V3,ElitistRoulette',
                'V4,Standard','V4,Complement','V4,Static','V4,Elitist','V4,ElitistRoulette',
                'X1,Standard','X1,Complement','X1,Static','X1,Elitist','X1,ElitistRoulette',
                'X2,Standard','X2,Complement','X2,Static','X2,Elitist','X2,ElitistRoulette',
                'X3,Standard','X3,Complement','X3,Static','X3,Elitist','X3,ElitistRoulette',
                'X4,Standard','X4,Complement','X4,Static','X4,Elitist','X4,ElitistRoulette',
                'Z1,Standard','Z1,Complement','Z1,Static','Z1,Elitist','Z1,ElitistRoulette',
                'Z2,Standard','Z2,Complement','Z2,Static','Z2,Elitist','Z2,ElitistRoulette',
                'Z3,Standard','Z3,Complement','Z3,Static','Z3,Elitist','Z3,ElitistRoulette',
                'Z4,Standard','Z4,Complement','Z4,Static','Z4,Elitist','Z4,ElitistRoulette'],
                "V1":
                ['S1,Standard', 'S1,Complement', 'S1,Elitist', 'S1,Static', 'S1,ElitistRoulette', 
                'S2,Standard', 'S2,Complement', 'S2,Elitist', 'S2,Static', 'S2,ElitistRoulette', 
                'S3,Standard', 'S3,Complement', 'S3,Elitist', 'S3,Static', 'S3,ElitistRoulette', 
                'S4,Standard', 'S4,Complement', 'S4,Elitist', 'S4,Static', 'S4,ElitistRoulette'],
                "V2":
                ['V1,Standard','V1,Complement','V1,Static','V1,Elitist','V1,ElitistRoulette',
                'V2,Standard','V2,Complement','V2,Static','V2,Elitist','V2,ElitistRoulette',
                'V3,Standard','V3,Complement','V3,Static','V3,Elitist','V3,ElitistRoulette',
                'V4,Standard','V4,Complement','V4,Static','V4,Elitist','V4,ElitistRoulette'],
                "V3":
                ['X1,Standard','X1,Complement','X1,Static','X1,Elitist','X1,ElitistRoulette',
                'X2,Standard','X2,Complement','X2,Static','X2,Elitist','X2,ElitistRoulette',
                'X3,Standard','X3,Complement','X3,Static','X3,Elitist','X3,ElitistRoulette',
                'X4,Standard','X4,Complement','X4,Static','X4,Elitist','X4,ElitistRoulette'],
                "V4":
                ['Z1,Standard','Z1,Complement','Z1,Static','Z1,Elitist','Z1,ElitistRoulette',
                'Z2,Standard','Z2,Complement','Z2,Static','Z2,Elitist','Z2,ElitistRoulette',
                'Z3,Standard','Z3,Complement','Z3,Static','Z3,Elitist','Z3,ElitistRoulette',
                'Z4,Standard','Z4,Complement','Z4,Static','Z4,Elitist','Z4,ElitistRoulette'],
                "V5":
                ['S1,Standard','S2,Standard','S3,Standard','S4,Standard','V1,Standard','V2,Standard','V3,Standard','V4,Standard'],
                "V6":
                ['S1,Complement','S2,Complement','S3,Complement','S4,Complement','V1,Complement','V2,Complement','V3,Complement','V4,Complement'],
                "V7":
                ['S1,Static','S2,Static','S3,Static','S4,Static','V1,Static','V2,Static','V3,Static','V4,Static'],
                "V8":
                ['S1,Elitist','S2,Elitist','S3,Elitist','S4,Elitist','V1,Elitist','V2,Elitist','V3,Elitist','V4,Elitist'],
                "V9":
                ['S1,ElitistRoulette','S2,ElitistRoulette','S3,ElitistRoulette','S4,ElitistRoulette','V1,ElitistRoulette','V2,ElitistRoulette','V3,ElitistRoulette','V4,ElitistRoulette'],
                "V10":
                ['S1,Standard','S2,Standard','S3,Standard','S4,Standard','X1,Standard','X2,Standard','X3,Standard','X4,Standard'],
                "V11":
                ['S1,Complement','S2,Complement','S3,Complement','S4,Complement','X1,Complement','X2,Complement','X3,Complement','X4,Complement'],
                "V12":
                ['S1,Static','S2,Static','S3,Static','S4,Static','X1,Static','X2,Static','X3,Static','X4,Static'],
                "V13":
                ['S1,Elitist','S2,Elitist','S3,Elitist','S4,Elitist','X1,Elitist','X2,Elitist','X3,Elitist','X4,Elitist'],
                "V14":
                ['S1,ElitistRoulette','S2,ElitistRoulette','S3,ElitistRoulette','S4,ElitistRoulette','X1,ElitistRoulette','X2,ElitistRoulette','X3,ElitistRoulette','X4,ElitistRoulette'],
                "V15":
                ['S1,Standard','S2,Standard','S3,Standard','S4,Standard','Z1,Standard','Z2,Standard','Z3,Standard','Z4,Standard'],
                "V16":
                ['S1,Complement','S2,Complement','S3,Complement','S4,Complement','Z1,Complement','Z2,Complement','Z3,Complement','Z4,Complement'],
                "V17":
                ['S1,Static','S2,Static','S3,Static','S4,Static','Z1,Static','Z2,Static','Z3,Static','Z4,Static'],
                "V18":
                ['S1,Elitist','S2,Elitist','S3,Elitist','S4,Elitist','Z1,Elitist','Z2,Elitist','Z3,Elitist','Z4,Elitist'],
                "V19":
                ['S1,ElitistRoulette','S2,ElitistRoulette','S3,ElitistRoulette','S4,ElitistRoulette','Z1,ElitistRoulette','Z2,ElitistRoulette','Z3,ElitistRoulette','Z4,ElitistRoulette'],
                "V20":
                ['V1,Standard','V2,Standard','V3,Standard','V4,Standard','X1,Standard','X2,Standard','X3,Standard','X4,Standard'],
                "V21":
                ['V1,Complement','V2,Complement','V3,Complement','V4,Complement','X1,Complement','X2,Complement','X3,Complement','X4,Complement'],
                "V22":
                ['V1,Static','V2,Static','V3,Static','V4,Static','X1,Static','X2,Static','X3,Static','X4,Static'],
                "V23":
                ['V1,Elitist','V2,Elitist','V3,Elitist','V4,Elitist','X1,Elitist','X2,Elitist','X3,Elitist','X4,Elitist'],
                "V24":
                ['V1,ElitistRoulette','V2,ElitistRoulette','V3,ElitistRoulette','V4,ElitistRoulette','X1,ElitistRoulette','X2,ElitistRoulette','X3,ElitistRoulette','X4,ElitistRoulette'],
                "V25":
                ['X1,Standard','X2,Standard','X3,Standard','X4,Standard','Z1,Standard','Z2,Standard','Z3,Standard','Z4,Standard'],
                "V26":
                ['X1,Complement','X2,Complement','X3,Complement','X4,Complement','Z1,Complement','Z2,Complement','Z3,Complement','Z4,Complement'],
                "V27":
                ['X1,Static','X2,Static','X3,Static','X4,Static','Z1,Static','Z2,Static','Z3,Static','Z4,Static'],
                "V28":
                ['X1,Elitist','X2,Elitist','X3,Elitist','X4,Elitist','Z1,Elitist','Z2,Elitist','Z3,Elitist','Z4,Elitist'],
                "V29":
                ['X1,ElitistRoulette','X2,ElitistRoulette','X3,ElitistRoulette','X4,ElitistRoulette','Z1,ElitistRoulette','Z2,ElitistRoulette','Z3,ElitistRoulette','Z4,ElitistRoulette'],
                "V30":
                ['S1,Standard','S2,Standard','S3,Standard','S4,Standard','V1,Standard','V2,Standard','V3,Standard','V4,Standard','X1,Standard','X2,Standard','X3,Standard','X4,Standard'],
                "V31":
                ['S1,Complement','S2,Complement','S3,Complement','S4,Complement','V1,Complement','V2,Complement','V3,Complement','V4,Complement','X1,Complement','X2,Complement','X3,Complement','X4,Complement'],
                "V32":
                ['S1,Static','S2,Static','S3,Static','S4,Static','V1,Static','V2,Static','V3,Static','V4,Static','X1,Static','X2,Static','X3,Static','X4,Static'],
                "V33":
                ['S1,Elitist','S2,Elitist','S3,Elitist','S4,Elitist','V1,Elitist','V2,Elitist','V3,Elitist','V4,Elitist','X1,Elitist','X2,Elitist','X3,Elitist','X4,Elitist'],
                "V34":
                ['S1,ElitistRoulette','S2,ElitistRoulette','S3,ElitistRoulette','S4,ElitistRoulette','V1,ElitistRoulette','V2,ElitistRoulette','V3,ElitistRoulette','V4,ElitistRoulette','X1,ElitistRoulette','X2,ElitistRoulette','X3,ElitistRoulette','X4,ElitistRoulette'],
                "V35":
                ['S1,Standard','S2,Standard','S3,Standard','S4,Standard','V1,Standard','V2,Standard','V3,Standard','V4,Standard','Z1,Standard','Z2,Standard','Z3,Standard','Z4,Standard'],
                "V36":
                ['S1,Complement','S2,Complement','S3,Complement','S4,Complement','V1,Complement','V2,Complement','V3,Complement','V4,Complement','Z1,Complement','Z2,Complement','Z3,Complement','Z4,Complement'],
                "V37":
                ['S1,Static','S2,Static','S3,Static','S4,Static','V1,Static','V2,Static','V3,Static','V4,Static','Z1,Static','Z2,Static','Z3,Static','Z4,Static'],
                "V38":
                ['S1,Elitist','S2,Elitist','S3,Elitist','S4,Elitist','V1,Elitist','V2,Elitist','V3,Elitist','V4,Elitist','Z1,Elitist','Z2,Elitist','Z3,Elitist','Z4,Elitist'],
                "V39":
                ['S1,ElitistRoulette','S2,ElitistRoulette','S3,ElitistRoulette','S4,ElitistRoulette','V1,ElitistRoulette','V2,ElitistRoulette','V3,ElitistRoulette','V4,ElitistRoulette','Z1,ElitistRoulette','Z2,ElitistRoulette','Z3,ElitistRoulette','Z4,ElitistRoulette'],
                "V40":
                ['S1,Standard','S2,Standard','S3,Standard','S4,Standard','X1,Standard','X2,Standard','X3,Standard','X4,Standard','Z1,Standard','Z2,Standard','Z3,Standard','Z4,Standard'],
                "V41":
                ['S1,Complement','S2,Complement','S3,Complement','S4,Complement','X1,Complement','X2,Complement','X3,Complement','X4,Complement','Z1,Complement','Z2,Complement','Z3,Complement','Z4,Complement'],
                "V42":
                ['S1,Static','S2,Static','S3,Static','S4,Static','X1,Static','X2,Static','X3,Static','X4,Static','Z1,Static','Z2,Static','Z3,Static','Z4,Static'],
                "V43":
                ['S1,Elitist','S2,Elitist','S3,Elitist','S4,Elitist','X1,Elitist','X2,Elitist','X3,Elitist','X4,Elitist','Z1,Elitist','Z2,Elitist','Z3,Elitist','Z4,Elitist'],
                "V44":
                ['S1,ElitistRoulette','S2,ElitistRoulette','S3,ElitistRoulette','S4,ElitistRoulette','X1,ElitistRoulette','X2,ElitistRoulette','X3,ElitistRoulette','X4,ElitistRoulette','Z1,ElitistRoulette','Z2,ElitistRoulette','Z3,ElitistRoulette','Z4,ElitistRoulette'],
                "V45":
                ['V1,Standard','V2,Standard','V3,Standard','V4,Standard','X1,Standard','X2,Standard','X3,Standard','X4,Standard','Z1,Standard','Z2,Standard','Z3,Standard','Z4,Standard'],
                "V46":
                ['V1,Complement','V2,Complement','V3,Complement','V4,Complement','X1,Complement','X2,Complement','X3,Complement','X4,Complement','Z1,Complement','Z2,Complement','Z3,Complement','Z4,Complement'],
                "V47":
                ['V1,Static','V2,Static','V3,Static','V4,Static','X1,Static','X2,Static','X3,Static','X4,Static','Z1,Static','Z2,Static','Z3,Static','Z4,Static'],
                "V48":
                ['V1,Elitist','V2,Elitist','V3,Elitist','V4,Elitist','X1,Elitist','X2,Elitist','X3,Elitist','X4,Elitist','Z1,Elitist','Z2,Elitist','Z3,Elitist','Z4,Elitist'],
                "V49":
                ['V1,ElitistRoulette','V2,ElitistRoulette','V3,ElitistRoulette','V4,ElitistRoulette','X1,ElitistRoulette','X2,ElitistRoulette','X3,ElitistRoulette','X4,ElitistRoulette','Z1,ElitistRoulette','Z2,ElitistRoulette','Z3,ElitistRoulette','Z4,ElitistRoulette'],
                "V50":
                ['S1,Standard','S2,Standard','S3,Standard','S4,Standard','V1,Standard','V2,Standard','V3,Standard','V4,Standard','X1,Standard','X2,Standard','X3,Standard','X4,Standard','Z1,Standard','Z2,Standard','Z3,Standard','Z4,Standard'],
                "V51":
                ['S1,Complement','S2,Complement','S3,Complement','S4,Complement','V1,Complement','V2,Complement','V3,Complement','V4,Complement','X1,Complement','X2,Complement','X3,Complement','X4,Complement','Z1,Complement','Z2,Complement','Z3,Complement','Z4,Complement'],
                "V52":
                ['S1,Static','S2,Static','S3,Static','S4,Static','V1,Static','V2,Static','V3,Static','V4,Static','X1,Static','X2,Static','X3,Static','X4,Static','Z1,Static','Z2,Static','Z3,Static','Z4,Static'],
                "V53":
                ['S1,Elitist','S2,Elitist','S3,Elitist','S4,Elitist','V1,Elitist','V2,Elitist','V3,Elitist','V4,Elitist','X1,Elitist','X2,Elitist','X3,Elitist','X4,Elitist','Z1,Elitist','Z2,Elitist','Z3,Elitist','Z4,Elitist'],
                "V54":
                ['S1,ElitistRoulette','S2,ElitistRoulette','S3,ElitistRoulette','S4,ElitistRoulette','V1,ElitistRoulette','V2,ElitistRoulette','V3,ElitistRoulette','V4,ElitistRoulette','X1,ElitistRoulette','X2,ElitistRoulette','X3,ElitistRoulette','X4,ElitistRoulette','Z1,ElitistRoulette','Z2,ElitistRoulette','Z3,ElitistRoulette','Z4,ElitistRoulette'],
                "V55": 
                ['S1,Standard', 'S1,Complement', 'S1,StaticProbability', 'S1,Elitist', 'S1,ElitistRoulette'],
                "V56":
                ['S2,Standard', 'S2,Complement', 'S2,StaticProbability', 'S2,Elitist', 'S2,ElitistRoulette'],
                "V57":
                ['S3,Standard', 'S3,Complement', 'S3,StaticProbability', 'S3,Elitist', 'S3,ElitistRoulette'],
                "V58":
                ['S4,Standard', 'S4,Complement', 'S4,StaticProbability', 'S4,Elitist', 'S4,ElitistRoulette'],
                "V59":
                ['V1,Standard', 'V1,Complement', 'V1,StaticProbability', 'V1,Elitist', 'V1,ElitistRoulette'],
                "V60":
                ['V2,Standard', 'V2,Complement', 'V2,StaticProbability', 'V2,Elitist', 'V2,ElitistRoulette'],
                "V61":
                ['V3,Standard', 'V3,Complement', 'V3,StaticProbability', 'V3,Elitist', 'V3,ElitistRoulette'],
                "V62":
                ['V4,Standard', 'V4,Complement', 'V4,StaticProbability', 'V4,Elitist', 'V4,ElitistRoulette'],
                "V63":
                ['X1,Standard', 'X1,Complement', 'X1,StaticProbability', 'X1,Elitist', 'X1,ElitistRoulette'],
                "V64":
                ['X2,Standard', 'X2,Complement', 'X2,StaticProbability', 'X2,Elitist', 'X2,ElitistRoulette'],
                "V65":
                ['X3,Standard', 'X3,Complement', 'X3,StaticProbability', 'X3,Elitist', 'X3,ElitistRoulette'],
                "V66":
                ['X4,Standard', 'X4,Complement', 'X4,StaticProbability', 'X4,Elitist', 'X4,ElitistRoulette'],
                "V67":
                ['Z1,Standard', 'Z1,Complement', 'Z1,StaticProbability', 'Z1,Elitist', 'Z1,ElitistRoulette'],
                "V68":
                ['Z2,Standard', 'Z2,Complement', 'Z2,StaticProbability', 'Z2,Elitist', 'Z2,ElitistRoulette'],
                "V69":
                ['Z3,Standard', 'Z3,Complement', 'Z3,StaticProbability', 'Z3,Elitist', 'Z3,ElitistRoulette'],
                "V70":
                ['Z4,Standard', 'Z4,Complement', 'Z4,StaticProbability', 'Z4,Elitist', 'Z4,ElitistRoulette'],
                "V71":
                ['S1,Standard', 'S2,Standard', 'S3,Standard', 'S4,Standard'],
                "V72":
                ['S1,Complement', 'S2,Complement', 'S3,Complement', 'S4,Complement'],
                "V73":
                ['S1,Static Probability', 'S2,Static Probability', 'S3,Static Probability', 'S4,Static Probability'],
                "V74":
                ['S1,Elitist', 'S2,Elitist', 'S3,Elitist', 'S4,Elitist'],
                "V75":
                ['S1,ElitistRoulette', 'S2,ElitistRoulette', 'S3,ElitistRoulette', 'S4,ElitistRoulette'],
                "V76":
                ['V1,Standard', 'V2,Standard', 'V3,Standard', 'V4,Standard'],
                "V77":
                ['V1,Complement', 'V2,Complement', 'V3,Complement', 'V4,Complement'],
                "V78":
                ['V1,Static Probability', 'V2,Static Probability', 'V3,Static Probability', 'V4,Static Probability'],
                "V79":
                ['V1,Elitist', 'V2,Elitist', 'V3,Elitist', 'V4,Elitist'],
                "V80":
                ['V1,ElitistRoulette', 'V2,ElitistRoulette', 'V3,ElitistRoulette', 'V4,ElitistRoulette'],
                "V81":
                ['X1,Standard', 'X2,Standard', 'X3,Standard', 'X4,Standard'],
                "V82":
                ['X1,Complement', 'X2,Complement', 'X3,Complement', 'X4,Complement'],
                "V83":
                ['X1,Static Probability', 'X2,Static Probability', 'X3,Static Probability', 'X4,Static Probability'],
                "V84":
                ['X1,Elitist', 'X2,Elitist', 'X3,Elitist', 'X4,Elitist'],
                "V85":
                ['X1,ElitistRoulette', 'X2,ElitistRoulette', 'X3,ElitistRoulette', 'X4,ElitistRoulette'],
                "V86":
                ['Z1,Standard', 'Z3,Standard', 'Z3,Standard', 'Z4,Standard'],
                "V87":
                ['Z1,Complement', 'Z3,Complement', 'Z3,Complement', 'Z4,Complement'],
                "V88":
                ['Z1,Static Probability', 'Z3,Static Probability', 'Z3,Static Probability', 'Z4,Static Probability'],
                "V89":
                ['Z1,Elitist', 'Z3,Elitist', 'Z3,Elitist', 'Z4,Elitist'],
                "V90":
                ['Z1,ElitistRoulette', 'Z3,ElitistRoulette', 'Z3,ElitistRoulette', 'Z4,ElitistRoulette']
                }

for run in range(runs):
    for instance in instances:
        for algorithm in algorithms:
            FO = algorithm.split("_")[1].replace("RW","")
            MH = algorithm.split("_")[0]
            ML = algorithm.split("_")[2][:2]
            VDS = algorithm.split("_")[2][3:] 
            numRewardType = int(algorithm.split("_")[2][2])
            DS_actions = dir_DS_actions[VDS]
            if ML == "40aQL" or ML == "40aSA" or ML == "40aBQSA" or ML == "QL":
                discretizationScheme = DS_actions[np.random.randint(low=0, high=len(DS_actions))]
                if numRewardType == 1:
                    rewardType = "withPenalty1"
                if numRewardType == 2:
                    rewardType = "withoutPenalty1"
                if numRewardType == 3:
                    rewardType = "globalBest"
                if numRewardType == 4:
                    rewardType = "rootAdaptation"
                if numRewardType == 5:
                    rewardType = "escalatingMultiplicativeAdaptation"

            if algorithm.split("_")[2] == "BCL":
                discretizationScheme = 'V4,Elitist'
            if algorithm.split("_")[2] == "MIR":
                discretizationScheme = 'V4,Complement'

            paramsML = {'cond_backward': '10', 'MinMax': 'min', 'DS_actions': DS_actions} 


            data = {
                'nombre_algoritmo' : algorithm,
                'parametros': json.dumps({
                    'instance_name' : instance,
                    'instance_file': instance+'.txt',
                    'instance_dir': instance_dir,
                    'population': population,
                    'maxIter':maxIter,
                    'discretizationScheme':discretizationScheme,
                    'ql_alpha': ql_alpha,
                    'ql_gamma': ql_gamma,
                    'repair': repair,
                    'policy': policy,
                    'rewardType': rewardType,
                    'qlAlphaType': qlAlphaType,
                    'beta_Dis': beta_Dis,
                    'FO': FO,
                    'MH': MH,
                    'ML': ML,
                    'paramsML': paramsML
            }),
                'estado' : 'pendiente'
            }
            result = connection.execute(insertDatosEjecucion,data)
            idEjecucion = result.fetchone()[0]
            print(f'Poblado ID #:{idEjecucion}')

print("Todo poblado")
