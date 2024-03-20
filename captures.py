"""
Captures database model
"""
import sqlite3
import datetime
import io
import numpy as np

# class logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('models_captures')


class Captures:
    """
    Database model for captures of world data
    This is more for studying and metrics as slow to use for now
    Will be expanded for other types of captures than just video
    Will be moved off of sqlite
    """
    def __init__(self, db_path: str=None):
        self.table_name = "captures"
        self.sqlconn = None
        self.frame = None

        if db_path:
            self.db_path = db_path
        else:
            dbn_dt = datetime.datetime.now().strftime("%m%d%Y")
            db_name = f"vc_{dbn_dt}.sql"
            self.db_path = f"db/{db_name}"

        self.table_schema = """
        CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame BLOB NOT NULL,
            text TEXT
        )
        """
        
        # check if table exists for model
        # if not and cannot create it, raise exception
        try:
            self.check_table()
        except Exception:
            raise

    def check_table(self):
        """
        This function checks for a captures table. 
        If doesn't exist, create it
        """
        try:
            self.sqlconn = sqlite3.connect(f"{self.db_path}")
            cursor = self.sqlconn.cursor()

            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.table_name,))
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                logger.info(f"Table {self.table_name} not found, creating table with schema\n{self.table_schema}")
                # Create table if it doesn't exist
                cursor.execute(self.table_schema)
                self.sqlconn.commit()

            self.sqlconn.close()
        except Exception as err:
            logger.error(f"check_tables error\n{err}")
            raise

    def insert_pframe_tensor(self, pframe_tensor: bytes, frame_text: str=None) -> bool:
        """
        Inserts a pickled frame tensor into the database with the text from the frame

        Args:
            bytes pframe_tensor: pickled frame from capture
            str pframe_text: text associated with pickled frame
        """
        try:
            self.sqlconn = sqlite3.connect(self.db_path)
            cursor = self.sqlconn.cursor()

            # convert pframe_tensor
            pt_binary = self.adapt_array(pframe_tensor)

            if not frame_text:
                cursor.execute(f"INSERT INTO {self.table_name} (frame, text) VALUES (?, ?)", (pt_binary,""))
            else:
                cursor.execute(f"INSERT INTO {self.table_name} (frame, text) VALUES (?, ?)", (pt_binary,frame_text))

            self.sqlconn.commit()

            # logger.info(f"Inserted pickled frame (len {len(pframe_tensor)}) into '{self.table_name}'")

            self.sqlconn.close()

            return True
        except Exception as err:
            logger.error(f"insert_pframe_tensor error\n{err}")
        
        return False

    def get_all_pframes(self, limit: int=0) -> list:
        """
        Return all pickled frames from db in table
        
        Args:
            int limit - Amount of frames to be returned. If 0, return all.
        """

        db_pframes = []
        try:
            self.sqlconn = sqlite3.connect(self.db_path)
            cursor = self.sqlconn.cursor()

            if limit > 0:
                cursor.execute(f"SELECT * FROM {self.table_name} LIMIT {limit}")
            else:
                cursor.execute(f"SELECT * FROM {self.table_name}")

            db_pframes = cursor.fetchall()
            self.sqlconn.close()
        except Exception as err:
            logger.error(f"get_all_pframes error\n{err}")
        
        return db_pframes

    def count_pframes(self) -> int:
        """
        Return amount of pframes in table captures
        """ 
        pfcount = 0

        try:
            self.sqlconn = sqlite3.connect(self.db_path)
            cursor = self.sqlconn.cursor()

            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")

            resp = cursor.fetchall()
            pfcount = resp[0]

            self.sqlconn.close()
        except Exception as err:
            logger.error(f"count_pframes error\n{err}")
        
        return pfcount
    
    def adapt_array(self, arr):
        """
        convert arr to binary
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
        """
        
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(self, text):
        """
        convert binary text to arr
        """
        out = io.BytesIO(text)
        out.seek(0)
        out = io.BytesIO(out.read())
        return np.load(out)



