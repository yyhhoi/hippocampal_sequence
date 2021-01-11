import numpy as np
import sqlite3
import pandas as pd
import os
from sqlalchemy import create_engine
class SQLRecorder:
    def __init__(self, num_neurons, dbpth, overwrite=False):
        self.num_neurons = num_neurons
        self.dbpth = dbpth
        
        # Connect database
        if overwrite:
            if os.path.exists(self.dbpth):
                os.remove(self.dbpth)
        self.sqlcon = sqlite3.connect(self.dbpth)
        self.cursor = self.sqlcon.cursor()

        # initialize insert's queries
        self.activity_insert_query = self._get_activity_insert_query()
        self.indata_insert_query = self._get_indata_insert_query()
        self.spikedata_insert_query = self._get_spikedata_insert_query()


    def create_table(self, neuronpos):

        # Activity table
        # id, tid, t, neuronidx, m
        activity_table_query = '''CREATE TABLE Activity (
            id INTEGER PRIMARY KEY,
            tid INTEGER NOT NULL,
            t REAL NOT NULL,
            neuronidx INTEGER NOT NULL,
            m REAL NOT NULL
        );'''
        
        self.cursor.execute(activity_table_query)

        # Indata table
        # id, tid, t, x, y, phase
        self.cursor.execute('''CREATE TABLE Indata ( 
            id INTEGER PRIMARY KEY,
            tid INTEGER UNIQUE NOT NULL, t REAL UNIQUE NOT NULL, 
            x REAL NOT NULL, y REAL NOT NULL, phase REAL NOT NULL);
        ''')

        # SpikeData table
        # id, neuronidx	neuronx	neurony	tidxsp	tsp	xsp	ysp	phasesp
        self.cursor.execute('''CREATE TABLE SpikeData ( 
            id INTEGER PRIMARY KEY,
            neuronidx INTEGER NOT NULL, neuronx REAL NOT NULL, neurony REAL NOT NULL,
            tidxsp REAL NOT NULL, tsp REAL NOT NULL, 
            xsp REAL NOT NULL, ysp REAL NOT NULL, phasesp REAL NOT NULL);
        ''')
        
        
        # NeuronPos Table
        # id, neuronx, neurony
        self.cursor.execute(
        '''CREATE TABLE NeuronPos (
            id INTEGER PRIMARY KEY,
            neuronx REAL NOT NULL,
            neurony REAL NOT NULL
        );''')
        
        neuronpos_insert_query = """INSERT INTO NeuronPos
            (id, neuronx, neurony) VALUES (?, ?, ?);""" 
        data = np.hstack([np.arange(neuronpos.shape[0]).reshape(-1, 1), neuronpos])
        self.conn.executemany(neuronpos_insert_query, data)
        


    def insert_activity(self, data):
        # data = (tid, t, neuronidx, m)
        # shape = (num_neurons, 4)
        self.cursor.executemany(self.activity_insert_query, data)


    def insert_indata(self, data):
        # data = (tid, t, x, y, phase)
        # shape = (5, )
        self.cursor.execute(self.indata_insert_query, data)


    def insert_spikedata(self, data):
        # data = (neuronidx, neuronx, neurony, tidxsp, tsp, xsp, ysp, phasesp)
        # shape = (arbitary, 8)
        self.cursor.executemany(self.spikedata_insert_query, data)

            
    def close(self):
        self.cursor.close()
        self.sqlcon.close()
        
    
    def commit(self):
        self.sqlcon.commit()
        
    
    def _get_activity_insert_query(self):
        # id, tid, t, neuronidx, m
        activity_insert_query = """INSERT INTO Activity
            (id, tid, t, neuronidx, m) VALUES (?, ?, ?, ?, ?);""" 
        return activity_insert_query


    def _get_indata_insert_query(self):
        # id, tid, t, x, y, phase
        indata_insert_query = """INSERT INTO Indata
            (id, tid, t, x, y, phase) VALUES (?, ?, ?, ?, ? ,?);""" 
        return indata_insert_query


    def _get_spikedata_insert_query(self):
        # id, neuronidx, neuronx, neurony, tidxsp, tsp, xsp, ysp, phasesp
        spikedata_insert_query = """INSERT INTO SpikeData
            (id, neuronidx, neuronx, neurony, tidxsp, tsp, xsp, ysp, phasesp) VALUES (?, ?, ?, ?, ? ,?, ?, ?, ?);""" 
        return spikedata_insert_query

    
    
def read_sql_simtables(db_pth):
    sqlcon = sqlite3.connect(db_pth)
    NeuronPos = pd.read_sql_query("SELECT neuronx, neurony FROM NeuronPos", sqlcon)
    NeuronPos = NeuronPos.to_numpy()
    Indata = pd.read_sql_query("SELECT * FROM InData", sqlcon)
    SpikeData = pd.read_sql_query("SELECT * FROM SpikeData", sqlcon)
    sqlcon.close()
    return Indata, SpikeData, NeuronPos
    
    