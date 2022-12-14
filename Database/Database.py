
# Utils
import sys
import os
import settings
# from envs import env
import configparser
from datetime import datetime

# SQL
import sqlalchemy as db
import psycopg2
import json
import pickle
import zlib

class Database:

    def __init__(self):

        # # Definicion Environments Vars
        # db_motor = env('DB_MOTOR')
        # db_user = env('DB_USER')
        # db_pass = env('DB_PASS')
        # db_server = env('DB_SERVER')
        # db_port = env('DB_PORT')
        # db_base   = env('DB_BASE')

        # # Conexión a la DB de resultados
        # self.engine = db.create_engine(f'{db_motor}://{db_user}:{db_pass}@{db_server}:{db_port}/{db_base}')
        # self.metadata = db.MetaData()

        #Credenciales
        config = configparser.ConfigParser()
        config.read('db_config.ini')
        host = config['postgres']['host']
        db_name = config['postgres']['db_name']
        port = config['postgres']['port']
        user = config['postgres']['user']
        pwd = config['postgres']['pass']


        # Conexión a la DB de resultados

        self.engine = db.create_engine(f'postgresql://{user}:{pwd}@{host}:{port}/{db_name}')
        self.metadata = db.MetaData()



    def endEjecucion(self,id,hora,estado):

        try: 
            connection = self.engine.connect()
            datosEjecucion = db.Table('datos_ejecucion', self.metadata, autoload=True, autoload_with=self.engine)
            updateDatosEjecucion = datosEjecucion.update().where(datosEjecucion.c.id == id)
            connection.execute(updateDatosEjecucion, {'fin':  hora, 'estado' : estado})

            return True

        except db.exc.SQLAlchemyError as e:
            return False

    def insertMemory(self,memory):

        try: 
            connection = self.engine.connect()
            datosIteracion = db.Table('datos_iteracion', self.metadata, autoload=True, autoload_with=self.engine)

            insertDatosIteracion = datosIteracion.insert().returning(datosIteracion.c.id)
            connection.execute(insertDatosIteracion,memory)
            memory = []

            return memory

        except db.exc.SQLAlchemyError as e:

            return memory

    def insertMemoryBest(self,memory):

        try: 
            connection = self.engine.connect()
            datosResultadoEjecucion = db.Table('resultado_ejecucion', self.metadata, autoload=True, autoload_with=self.engine)

            insertDatosIteracion = datosResultadoEjecucion.insert().returning(datosResultadoEjecucion.c.id)
            connection.execute(insertDatosIteracion,memory)
            memory = []

            return memory

        except db.exc.SQLAlchemyError as e:

            return memory
        

    
    def getLastPendingAlgorithm(self,estado):

        params = {}
        id = 0
        algorithm = ''
        try: 
            connection = self.engine.connect()
            datosEjecucion = db.Table('datos_ejecucion', self.metadata, autoload=True, autoload_with=self.engine)

            sql = db.text("""update datos_ejecucion set estado = 'ejecutando', inicio=:inicio
                            where id =
                            (select  id from datos_ejecucion
                            where estado = 'pendiente'
                            order by id asc
                            limit 1 for update)
                            returning id, nombre_algoritmo, parametros;""")

            inicio = datetime.now()
            result = connection.execute(sql,**{"inicio":inicio}).fetchall()


        except db.exc.SQLAlchemyError as e:

            return id,algorithm,params

        if(result):

            for row in result:

                params = json.loads(row[datosEjecucion.c.parametros])
                id = row[datosEjecucion.c.id]
                algorithm = row[datosEjecucion.c.nombre_algoritmo]

            return id, algorithm,  params

        else:
            return id,algorithm,params

        
        
