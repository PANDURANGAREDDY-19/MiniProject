import sqlite3
import pandas as pd
from datetime import datetime
import sys

from src.exception import CustomException
from src.logger import logging

class DatabaseHandler:
    def __init__(self, db_path='predictions.db'):
        """Initialize database connection"""
        try:
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.create_tables()
            logging.info(f"Database connection established at {db_path}")
        except Exception as e:
            logging.error(f"Error connecting to database: {e}")
            raise CustomException(e, sys)
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            cursor = self.conn.cursor()
            
            # Create table for storing prediction data
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                age INTEGER,
                sex TEXT,
                cholesterol INTEGER,
                chest_pain_type TEXT,
                resting_bp INTEGER,
                resting_ecg TEXT,
                fasting_bs INTEGER,
                max_hr INTEGER,
                st_slope TEXT,
                oldpeak REAL,
                exercise_angina TEXT,
                prediction INTEGER
            )
            ''')
            
            self.conn.commit()
            logging.info("Database tables created successfully")
        except Exception as e:
            logging.error(f"Error creating tables: {e}")
            raise CustomException(e, sys)
    
    def save_prediction(self, input_data, prediction):
        """Save prediction data to database"""
        try:
            cursor = self.conn.cursor()
            
            data = input_data.iloc[0].to_dict()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute('''
            INSERT INTO predictions (
                timestamp, age, sex, cholesterol, chest_pain_type, 
                resting_bp, resting_ecg, fasting_bs, max_hr, 
                st_slope, oldpeak, exercise_angina, prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, 
                data['Age'], 
                data['Sex'], 
                data['Cholesterol'],
                data['ChestPainType'],
                data['RestingBP'],
                data['RestingECG'],
                data['FastingBS'],
                data['MaxHR'],
                data['ST_Slope'],
                data['Oldpeak'],
                data['ExerciseAngina'],
                int(prediction)
            ))
            
            self.conn.commit()
            logging.info(f"Prediction saved to database with timestamp {timestamp}")
            return True
        except Exception as e:
            logging.error(f"Error saving prediction to database: {e}")
            raise CustomException(e, sys)
    
    def get_all_predictions(self):
        """Retrieve all predictions from database"""
        try:
            query = "SELECT * FROM predictions ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            logging.error(f"Error retrieving predictions from database: {e}")
            raise CustomException(e, sys)
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()