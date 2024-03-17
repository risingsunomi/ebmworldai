"""
Captures database model
"""
import sqlite3

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
    def __init__(self):
        self.table_name = "captures"
        self.sqlconn = None
        self.frame = None

        self.table_schema = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame BLOB NOT NULL
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
            self.sqlconn = sqlite3.connect("db/video_captures.sql")
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

    def insert_pframe_tensor(self, pframe_tensor: bytes) -> bool:
        """
        Inserts a pickled frame tensor into the database

        Args:
            bytes pframe_tensor: pickled frame from capture
        """
        try:
            self.sqlconn = sqlite3.connect("db/video_captures.sql")
            cursor = self.sqlconn.cursor()

            cursor.execute(f"INSERT INTO {self.table_name} (frame) VALUES (?)", (pframe_tensor,))
            self.sqlconn.commit()

            logger.info(f"Inserted pickled frame (len {len(pframe_tensor)}) into '{self.table_name}'")

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
            self.sqlconn = sqlite3.connect("db/video_captures.sql")
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


