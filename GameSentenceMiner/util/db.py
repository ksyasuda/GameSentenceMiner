

import copy
from datetime import datetime
import json
import os
import random
import shutil
import sqlite3
from sys import platform
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Type, TypeVar
import threading
import uuid

import pytz
from datetime import timedelta

import regex

from GameSentenceMiner.util.text_log import GameLine
from GameSentenceMiner.util.configuration import get_config, get_stats_config, logger, is_dev, sanitize_and_resolve_path
import gzip

# Matches any Unicode punctuation (\p{P}), symbol (\p{S}), or separator (\p{Z}); \p{Z} includes whitespace/separator chars
punctuation_regex = regex.compile(r'[\p{P}\p{S}\p{Z}]')

# Matches repeating characters that are 3 repeats or more and limits them to three repeats.
# For example: あああああ -> あああ or 黙れ黙れ黙れ黙れ -> 黙れ黙れ黙れ
repeating_chars_regex = regex.compile(r'(.+?)\1{2,}')

class SQLiteDB:
    """
    Multi-purpose SQLite database utility class for general use.
    Thread-safe for basic operations.
    Supports optional read-only mode.
    """

    def __init__(self, db_path: str, read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        self._lock = threading.Lock()

    def _get_connection(self):
        if self.read_only:
            # Use URI mode for read-only
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Enable foreign key enforcement
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def backup(self, backup_path: str):
        """ Create a backup of the database using built in SQLite backup API. """
        if self.read_only:
            raise RuntimeError("Cannot backup a database opened in read-only mode.")
        with self._lock, self._get_connection() as conn:
            with sqlite3.connect(backup_path, check_same_thread=False) as backup_conn:
                conn.backup(backup_conn)
                if is_dev:
                    logger.debug(f"Database backed up to {backup_path}")

    def execute(self, query: str, params: Union[Tuple, Dict] = (), commit: bool = False) -> sqlite3.Cursor:
        if self.read_only and commit:
            raise RuntimeError("Cannot commit changes in read-only mode.")
        with self._lock, self._get_connection() as conn:
            if is_dev:
                truncated_params = tuple(str(p)[:50] if p is not None else None for p in params)
                logger.debug(f"Executed query: {query} with params: {truncated_params}")
            cur = conn.cursor()
            cur.execute(query, params)
            if commit:
                conn.commit()
            return cur

    def executemany(self, query: str, seq_of_params: List[Union[Tuple, Dict]], commit: bool = False) -> sqlite3.Cursor:
        if self.read_only and commit:
            raise RuntimeError("Cannot commit changes in read-only mode.")
        with self._lock, self._get_connection() as conn:
            if is_dev:
                logger.debug(f"Executed query: {query} with {len(seq_of_params)} params")
            cur = conn.cursor()
            cur.executemany(query, seq_of_params)
            if commit:
                conn.commit()
            return cur

    def fetchall(self, query: str, params: Union[Tuple, Dict] = ()) -> List[Tuple]:
        cur = self.execute(query, params)
        return cur.fetchall()

    def fetchone(self, query: str, params: Union[Tuple, Dict] = ()) -> Optional[Tuple]:
        cur = self.execute(query, params)
        return cur.fetchone()

    def create_table(self, table_sql: str):
        if self.read_only:
            raise RuntimeError("Cannot create tables in read-only mode.")
        self.execute(table_sql, commit=True)

    def table_exists(self, table: str) -> bool:
        result = self.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return result is not None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def execute_in_transaction(self, operations: List[Tuple[str, List[Tuple]]]) -> None:
        """
        Execute multiple operations in a single transaction.
        Rolls back all changes if any operation fails.

        Args:
            operations: List of (query, list_of_params) tuples.
                       Each operation will be executed with executemany.
        """
        if self.read_only:
            raise RuntimeError("Cannot execute transaction in read-only mode.")

        with self._lock, self._get_connection() as conn:
            try:
                cur = conn.cursor()
                for query, params_list in operations:
                    if params_list:
                        cur.executemany(query, params_list)
                        if is_dev:
                            logger.debug(f"Transaction: executed {query[:50]}... with {len(params_list)} rows")
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e



# Abstract base for table-mapped classes
T = TypeVar('T', bound='SQLiteDBTable')


class SQLiteDBTable:
    _db: SQLiteDB = None
    _table: str = ''
    _fields: List[str] = []
    _types: List[type] = []
    _pk: str = 'id'
    _auto_increment: bool = True
    _column_order_cache: Optional[List[str]] = None  # Cache for actual column order
    _foreign_keys: List[Tuple[str, str, str]] = []  # List of (column, ref_table, ref_column)
    _defaults: Dict[str, Any] = {}  # Default values for columns
    _indexes: List[Tuple] = []  # Index specs: (columns, is_unique, name) where columns is str or tuple

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_table') or not cls._table:
            cls._table = cls.__name__.lower()
        if not hasattr(cls, '_fields') or not cls._fields:
            raise NotImplementedError(f"{cls.__name__} must define _fields")

    @classmethod
    def _python_type_to_sqlite(cls, py_type: type) -> str:
        """Convert Python type to SQLite type."""
        type_map = {
            str: 'TEXT',
            int: 'INTEGER',
            float: 'REAL',
            bool: 'INTEGER',  # SQLite doesn't have bool, uses 0/1
            list: 'TEXT',     # Stored as JSON
            dict: 'TEXT',     # Stored as JSON
        }
        return type_map.get(py_type, 'TEXT')

    @classmethod
    def _normalize_index_spec(cls, spec):
        """Normalize index spec to (columns_tuple, is_unique, name)."""
        columns_spec, is_unique, custom_name = spec
        columns = (columns_spec,) if isinstance(columns_spec, str) else tuple(columns_spec)
        name = custom_name or f"idx_{cls._table}_{'_'.join(columns)}"
        return (columns, is_unique, name)

    @classmethod
    def _create_index(cls, columns, is_unique, name):
        """Create a single index."""
        unique_clause = "UNIQUE " if is_unique else ""
        cls._db.execute(f"CREATE {unique_clause}INDEX IF NOT EXISTS {name} ON {cls._table}({', '.join(columns)})", commit=True)

    @classmethod
    def _create_all_indexes(cls):
        """Create all indexes defined in _indexes."""
        if not hasattr(cls, '_indexes') or not cls._indexes:
            return
        for spec in cls._indexes:
            columns, is_unique, name = cls._normalize_index_spec(spec)
            cls._create_index(columns, is_unique, name)

    @classmethod
    def set_db(cls, db: SQLiteDB):
        cls._db = db
        cls._column_order_cache = None  # Reset cache when database changes
        # Ensure table exists
        if not db.table_exists(cls._table):
            field_defs = []
            for i, field in enumerate(cls._fields):
                # _types[0] is pk type, so field types start at index 1
                if cls._types and len(cls._types) > i + 1:
                    sqlite_type = cls._python_type_to_sqlite(cls._types[i + 1])
                else:
                    sqlite_type = 'TEXT'
                field_defs.append(f"{field} {sqlite_type}")
            fields_def = ', '.join(field_defs)
            pk_def = f"{cls._pk} TEXT PRIMARY KEY" if not cls._auto_increment else f"{cls._pk} INTEGER PRIMARY KEY AUTOINCREMENT"

            fk_defs = []
            if hasattr(cls, '_foreign_keys') and cls._foreign_keys:
                for fk in cls._foreign_keys:
                    column, ref_table, ref_column = fk[:3]
                    cascade = fk[3] if len(fk) > 3 else ''
                    fk_sql = f"FOREIGN KEY ({column}) REFERENCES {ref_table}({ref_column})"
                    if cascade:
                        fk_sql += f" ON DELETE {cascade}"
                    fk_defs.append(fk_sql)

            if fk_defs:
                fk_clause = ', ' + ', '.join(fk_defs)
            else:
                fk_clause = ''

            create_table_sql = f"CREATE TABLE IF NOT EXISTS {cls._table} ({pk_def}, {fields_def}{fk_clause})"
            db.create_table(create_table_sql)
            cls._create_all_indexes()
        # Check for missing columns and add them
        existing_columns = [col[1] for col in db.fetchall(f"PRAGMA table_info({cls._table})")]
        for i, field in enumerate(cls._fields):
            if field not in existing_columns:
                if cls._types and len(cls._types) > i + 1:
                    sqlite_type = cls._python_type_to_sqlite(cls._types[i + 1])
                else:
                    sqlite_type = 'TEXT'
                default_clause = ''
                if hasattr(cls, '_defaults') and field in cls._defaults:
                    default_val = cls._defaults[field]
                    default_clause = f" DEFAULT {default_val}"
                db.execute(f"ALTER TABLE {cls._table} ADD COLUMN {field} {sqlite_type}{default_clause}", commit=True)
                cls._column_order_cache = None  # Reset cache when schema changes
        # Create indexes (will use IF NOT EXISTS, safe for migrations)
        cls._create_all_indexes()

    @classmethod
    def all(cls: Type[T]) -> List[T]:
        rows = cls._db.fetchall(f"SELECT * FROM {cls._table}")
        return [cls.from_row(row) for row in rows]

    @classmethod
    def get(cls: Type[T], pk_value: Any) -> Optional[T]:
        row = cls._db.fetchone(
            f"SELECT * FROM {cls._table} WHERE {cls._pk}=?", (pk_value,))
        return cls.from_row(row) if row else None

    @classmethod
    def one(cls: Type[T]) -> Optional[T]:
        row = cls._db.fetchone(f"SELECT * FROM {cls._table} LIMIT 1")
        return cls.from_row(row) if row else None

    @classmethod
    def from_row(cls: Type[T], row: Tuple, clean_columns: list = []) -> T:
        if not row:
            return None
        obj = cls()
        
        try:
            # Get actual column order from database schema
            actual_columns = cls.get_actual_column_order()
            expected_fields = [cls._pk] + cls._fields
            
            # Create a mapping from actual column positions to expected field positions
            column_mapping = {}
            for i, actual_col in enumerate(actual_columns):
                if actual_col in expected_fields:
                    expected_index = expected_fields.index(actual_col)
                    column_mapping[i] = expected_index
            
            # Process each column in the row based on the mapping
            for actual_pos, row_value in enumerate(row):
                if actual_pos not in column_mapping:
                    continue  # Skip unknown columns
                    
                expected_pos = column_mapping[actual_pos]
                field = expected_fields[expected_pos]
                field_type = cls._types[expected_pos]
                
                
                if field in clean_columns and isinstance(row_value, str):
                    # if get_stats_config().regex_out_punctuation:
                    row_value = punctuation_regex.sub('', row_value).strip() 
                    if get_stats_config().regex_out_repetitions:
                        row_value = repeating_chars_regex.sub(r'\1\1\1', row_value)
                
                cls._set_field_value(obj, field, field_type, row_value, expected_pos == 0 and field == cls._pk)
                
        except Exception as e:
            # Fallback to original behavior if schema-based mapping fails
            logger.warning(f"Column mapping failed for {cls._table}, falling back to positional mapping: {e}")
            expected_fields = [cls._pk] + cls._fields
            for i, field in enumerate(expected_fields):
                if i >= len(row):
                    break  # Safety check
                field_type = cls._types[i]
                cls._set_field_value(obj, field, field_type, row[i], i == 0 and field == cls._pk)
                    
        return obj
    
    @classmethod
    def _set_field_value(cls, obj, field: str, field_type: type, row_value, is_pk: bool = False):
        """Helper method to set field value with proper type conversion."""
        if is_pk:
            if field_type is int:
                setattr(obj, field, int(row_value) if row_value is not None else None)
            elif field_type is str:
                setattr(obj, field, str(row_value) if row_value is not None else None)
            return
            
        if field_type is str:
            if not row_value:
                setattr(obj, field, "")
            elif isinstance(row_value, str) and (row_value.startswith('[') or row_value.startswith('{')):
                try:
                    setattr(obj, field, json.loads(row_value))
                except json.JSONDecodeError:
                    setattr(obj, field, row_value)
            else:
                setattr(obj, field, str(row_value) if row_value is not None else None)
        elif field_type is list:
            try:
                setattr(obj, field, json.loads(row_value) if row_value else [])
            except json.JSONDecodeError:
                setattr(obj, field, [])
        elif field_type is int:
            setattr(obj, field, int(row_value) if row_value is not None else None)
        elif field_type is float:
            if row_value is None:
                setattr(obj, field, None)
            elif isinstance(row_value, str):
                # Try to parse datetime strings to Unix timestamp
                try:
                    # First try direct float conversion
                    setattr(obj, field, float(row_value))
                except ValueError:
                    # If that fails, try parsing as datetime string
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(row_value.replace(' ', 'T'))
                        setattr(obj, field, dt.timestamp())
                    except (ValueError, AttributeError):
                        # If all parsing fails, set to None
                        logger.warning(f"Could not convert '{row_value}' to float or datetime, setting to None")
                        setattr(obj, field, None)
            else:
                setattr(obj, field, float(row_value))
        elif field_type is bool:
            # Convert from SQLite: 0/1 (int), '0'/'1' (str), or None -> bool
            # Default to False for None/empty, True only for 1 or '1'
            setattr(obj, field, row_value == 1 or row_value == '1')
        elif field_type is dict:
            try:
                setattr(obj, field, json.loads(row_value) if row_value else {})
            except json.JSONDecodeError:
                setattr(obj, field, {})
        else:
            setattr(obj, field, row_value)

    def save(self, retry=1):
        try:
            for field in self._fields:
                field_value = getattr(self, field)
                if isinstance(field_value, list):
                    setattr(self, field, json.dumps(field_value))
                elif isinstance(field_value, dict):
                    setattr(self, field, json.dumps(field_value))
                elif isinstance(field_value, bool):
                    # Convert boolean to integer (0 or 1) for SQLite storage
                    setattr(self, field, 1 if field_value else 0)
            data = {field: getattr(self, field) for field in self._fields}
            pk_val = getattr(self, self._pk, None)
            if pk_val is None:
                # Insert
                keys = ', '.join(data.keys())
                placeholders = ', '.join(['?'] * len(data))
                values = tuple(data.values())
                query = f"INSERT INTO {self._table} ({keys}) VALUES ({placeholders})"
                cur = self._db.execute(query, values, commit=True)
                setattr(self, self._pk, cur.lastrowid)
                logger.debug(f"Inserted into {self._table} id={cur.lastrowid}")
            else:
                # Update
                set_clause = ', '.join([f"{k}=?" for k in data.keys()])
                values = tuple(data.values())
                query = f"UPDATE {self._table} SET {set_clause} WHERE {self._pk}=?"
                self._db.execute(query, values + (pk_val,), commit=True)
                logger.debug(f"Updated {self._table} id={pk_val}")
        except sqlite3.OperationalError as e:
            if retry <= 0:
                logger.error(f"Failed to save record to {self._table}: {e}")
                return
            if "no column named" in str(e):
                new_column = str(e).split("no column named ")[1].strip()
                logger.info(f"Adding missing column {new_column} to {self._table}")
                # Get type of new column from self._types by matching column name in _fields
                if new_column in self._fields:
                    self.add_column(new_column)
                    self.save(retry=retry - 1)  # Retry after adding column

    def add(self, retry=1):
        try:
            pk_val = getattr(self, self._pk, None)
            if self._auto_increment:
                self.save()
            elif pk_val is None:
                raise ValueError(
                    f"Primary key {self._pk} must be set for non-auto-increment tables.")
            else:
                # Serialize list and dict fields to JSON, convert booleans to integers
                for field in self._fields:
                    field_value = getattr(self, field)
                    if isinstance(field_value, (list, dict)):
                        setattr(self, field, json.dumps(field_value))
                    elif isinstance(field_value, bool):
                        # Convert boolean to integer (0 or 1) for SQLite storage
                        setattr(self, field, 1 if field_value else 0)
                
                keys = ', '.join(self._fields + [self._pk])
                placeholders = ', '.join(['?'] * (len(self._fields) + 1))
                values = tuple(getattr(self, field)
                            for field in self._fields) + (pk_val,)
                query = f"INSERT INTO {self._table} ({keys}) VALUES ({placeholders})"
                self._db.execute(query, values, commit=True)
        except sqlite3.OperationalError as e:
            if retry <= 0:
                logger.error(f"Failed to add record to {self._table}: {e}")
                return
            if "no column named" in str(e):
                new_column = str(e).split("no column named ")[1].strip()
                logger.info(f"Adding missing column {new_column} to {self._table}")
                # Get type of new column from self._types by matching column name in _fields
                if new_column in self._fields:
                    self.add_column(new_column)
                    self.add(retry=retry - 1)  # Retry after adding column
            
    def add_column(self, column_name: str, new_column_type: str = "TEXT"):
        try:
            self._db.execute(
                f"ALTER TABLE {self._table} ADD COLUMN {column_name} {new_column_type}", commit=True)
            self.__class__._column_order_cache = None  # Reset cache when schema changes
            logger.info(f"Added column {column_name} to {self._table}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                logger.warning(
                    f"Column {column_name} already exists in {self._table}.")
            else:
                logger.error(
                    f"Failed to add column {column_name} to {self._table}: {e}")

    def delete(self):
        pk_val = getattr(self, self._pk, None)
        if pk_val is not None:
            query = f"DELETE FROM {self._table} WHERE {self._pk}=?"
            self._db.execute(query, (pk_val,), commit=True)

    def print(self):
        pk_val = getattr(self, self._pk, None)
        logger.info(f"{self._table} Record (id={pk_val}): " +
                    ', '.join([f"{field}={getattr(self, field)}" for field in self._fields]))

    @classmethod
    def drop(cls):
        cls._db.execute(f"DROP TABLE IF EXISTS {cls._table}", commit=True)
        
    @classmethod
    def has_column(cls, column_name: str) -> bool:
        row = cls._db.fetchone(
            f"PRAGMA table_info({cls._table})")
        if not row:
            return False
        columns = [col[1] for col in cls._db.fetchall(
            f"PRAGMA table_info({cls._table})")]
        return column_name in columns
    
    @classmethod
    def rename_column(cls, old_column: str, new_column: str):
        cls._db.execute(
            f"ALTER TABLE {cls._table} RENAME COLUMN {old_column} TO {new_column}", commit=True)
        cls._column_order_cache = None  # Reset cache when schema changes
        
    @classmethod
    def drop_column(cls, column_name: str):
        cls._db.execute(
            f"ALTER TABLE {cls._table} DROP COLUMN {column_name}", commit=True)
        cls._column_order_cache = None  # Reset cache when schema changes
        
    @classmethod
    def get_column_type(cls, column_name: str) -> Optional[str]:
        row = cls._db.fetchone(
            f"PRAGMA table_info({cls._table})")
        if not row:
            return None
        columns = cls._db.fetchall(
            f"PRAGMA table_info({cls._table})")
        for col in columns:
            if col[1] == column_name:
                return col[2]  # Return the type
        return None
        
    @classmethod
    def alter_column_type(cls, old_column: str, new_column: str, new_type: str):
        # Add new column
        cls._db.execute(
            f"ALTER TABLE {cls._table} ADD COLUMN {new_column} {new_type}", commit=True)
        # Copy and cast data
        cls._db.execute(
            f"UPDATE {cls._table} SET {new_column} = CAST({old_column} AS {new_type})", commit=True)
        cls._db.execute(
            f"ALTER TABLE {cls._table} DROP COLUMN {old_column}", commit=True)
        cls._column_order_cache = None  # Reset cache when schema changes
        
    @classmethod
    def get_actual_column_order(cls) -> List[str]:
        """Get the actual column order from the database schema."""
        if cls._column_order_cache is not None:
            return cls._column_order_cache
            
        # Use direct database access to avoid recursion through from_row()
        with cls._db._lock:
            import sqlite3
            with sqlite3.connect(cls._db.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({cls._table})")
                columns_info = cursor.fetchall()
                
        # Each row is (cid, name, type, notnull, dflt_value, pk)
        # Sort by column id (cid) to get the actual order
        sorted_columns = sorted(columns_info, key=lambda x: x[0])
        column_order = [col[1] for col in sorted_columns]
        
        # Cache the result
        cls._column_order_cache = column_order
        return column_order
        
    @classmethod
    def get_expected_column_list(cls) -> str:
        """Get comma-separated list of columns in expected order for explicit SELECT queries."""
        expected_fields = [cls._pk] + cls._fields
        return ', '.join(expected_fields)


class AIModelsTable(SQLiteDBTable):
    _table = 'ai_models'
    _fields = ['gemini_models', 'groq_models', 'ollama_models', 'lm_studio_models', 'openai_models', 'last_updated']
    _types = [int,  # Includes primary key type
              list, list, list, list, list, float]
    _pk = 'id'

    def __init__(self, id: Optional[int] = None, gemini_models: list = None, groq_models: list = None, ollama_models: list = None, lm_studio_models: list = None, openai_models: list = None, last_updated: Optional[float] = None):
        self.id = id
        self.gemini_models = gemini_models if gemini_models is not None else []
        self.groq_models = groq_models if groq_models is not None else []
        self.ollama_models = ollama_models if ollama_models is not None else []
        self.lm_studio_models = lm_studio_models if lm_studio_models is not None else []
        self.openai_models = openai_models if openai_models is not None else []
        self.last_updated = last_updated

    @classmethod
    def get_gemini_models(cls) -> List[str]:
        rows = cls.all()
        return rows[0].gemini_models if rows else []

    @classmethod
    def get_groq_models(cls) -> List[str]:
        rows = cls.all()
        return rows[0].groq_models if rows else []

    @classmethod
    def get_ollama_models(cls) -> List[str]:
        rows = cls.all()
        return rows[0].ollama_models if rows else []

    @classmethod
    def get_lm_studio_models(cls) -> List[str]:
        rows = cls.all()
        return rows[0].lm_studio_models if rows else []
    
    @classmethod
    def get_openai_models(cls) -> List[str]:
        rows = cls.all()
        return rows[0].openai_models if rows else []

    @classmethod
    def get_models(cls) -> Optional['AIModelsTable']:
        return cls.one()
    
    @classmethod
    def store_models(cls, gemini_models: List[str], groq_models: List[str], ollama_models: List[str] = None, lm_studio_models: List[str] = None, openai_models: List[str] = None):
        """Store or update all AI models at once."""
        models = cls.one()
        if not models:
            new_model = cls(
                gemini_models=gemini_models,
                groq_models=groq_models,
                ollama_models=ollama_models or [],
                lm_studio_models=lm_studio_models or [],
                openai_models=openai_models or [],
                last_updated=time.time()
            )
            new_model.save()
            return
        
        models.gemini_models = gemini_models
        models.groq_models = groq_models
        if ollama_models is not None:
            models.ollama_models = ollama_models
        if lm_studio_models is not None:
            models.lm_studio_models = lm_studio_models
        if openai_models is not None:
            models.openai_models = openai_models
        models.last_updated = time.time()
        models.save()

    @classmethod
    def update_models(cls, gemini_models: List[str], groq_models: List[str], ollama_models: List[str] = None, lm_studio_models: List[str] = None, openai_models: List[str] = None):
        models = cls.one()
        if not models:
            new_model = cls(gemini_models=gemini_models,
                            groq_models=groq_models, 
                            ollama_models=ollama_models,
                            lm_studio_models=lm_studio_models,
                            openai_models=openai_models,
                            last_updated=time.time())
            new_model.save()
            return
        if gemini_models:
            models.gemini_models = gemini_models
        if groq_models:
            models.groq_models = groq_models
        if ollama_models:
            models.ollama_models = ollama_models
        if lm_studio_models:
            models.lm_studio_models = lm_studio_models
        if openai_models:
            models.openai_models = openai_models
        models.last_updated = time.time()
        models.save()

    @classmethod
    def set_gemini_models(cls, models: List[str]):
        models = cls.all()
        if not models:
            new_model = cls(gemini_models=models,
                            groq_models=[], last_updated=time.time())
            new_model.save()
            return
        for model in models:
            model.gemini_models = models
            model.last_updated = time.time()
            model.save()

    @classmethod
    def set_groq_models(cls, models: List[str]):
        models = cls.all()
        if not models:
            new_model = cls(gemini_models=[], groq_models=models,
                            last_updated=time.time())
            new_model.save()
            return
        for model in models:
            model.groq_models = models
            model.last_updated = time.time()
            model.save()


class GameLinesTable(SQLiteDBTable):
    _table = 'game_lines'
    _fields = ['game_name', 'line_text', 'screenshot_in_anki',
               'audio_in_anki', 'screenshot_path', 'audio_path', 'replay_path',
               'translation', 'timestamp', 'original_game_name', 'game_id',
               'note_ids', 'last_modified', 'tokenized',
               'total_length', 'filtered_length', 'word_count', 'kanji_count']
    _types = [str,  # Includes primary key type
              str, str, str, str, str, str, str, str, float, str, str, list, float, int,
              int, int, int, int]
    _pk = 'id'
    _auto_increment = False  # Use string IDs
    _defaults = {
        'tokenized': 0,
        'total_length': 0,
        'filtered_length': 0,
        'word_count': 0,
        'kanji_count': 0
    }
    _indexes = [
        ('tokenized', False, None),
    ]

    def __init__(self, id: Optional[str] = None,
                 game_name: Optional[str] = None,
                 line_text: Optional[str] = None,
                 context: Optional[str] = None,
                 timestamp: Optional[float] = None,
                 screenshot_in_anki: Optional[str] = None,
                 audio_in_anki: Optional[str] = None,
                 screenshot_path: Optional[str] = None,
                 audio_path: Optional[str] = None,
                 replay_path: Optional[str] = None,
                 translation: Optional[str] = None,
                 original_game_name: Optional[str] = None,
                 game_id: Optional[str] = None,
                 note_ids: Optional[List[str]] = None,
                 last_modified: Optional[float] = None,
                 tokenized: Optional[int] = 0,
                 total_length: Optional[int] = 0,
                 filtered_length: Optional[int] = 0,
                 word_count: Optional[int] = 0,
                 kanji_count: Optional[int] = 0):
        self.id = id
        self.game_name = game_name
        self.line_text = line_text
        self.context = context
        self.timestamp = float(timestamp) if timestamp is not None else datetime.now().timestamp()
        self.screenshot_in_anki = screenshot_in_anki if screenshot_in_anki is not None else ''
        self.audio_in_anki = audio_in_anki if audio_in_anki is not None else ''
        self.screenshot_path = screenshot_path if screenshot_path is not None else ''
        self.audio_path = audio_path if audio_path is not None else ''
        self.replay_path = replay_path if replay_path is not None else ''
        self.translation = translation if translation is not None else ''
        self.original_game_name = original_game_name if original_game_name is not None else ''
        self.game_id = game_id if game_id is not None else ''
        self.note_ids = note_ids
        self.last_modified = last_modified if last_modified is not None else time.time()
        self.tokenized = tokenized if tokenized is not None else 0
        self.total_length = total_length if total_length is not None else 0
        self.filtered_length = filtered_length if filtered_length is not None else 0
        self.word_count = word_count if word_count is not None else 0
        self.kanji_count = kanji_count if kanji_count is not None else 0

    @classmethod
    def all(cls, for_stats: bool = False) -> List['GameLinesTable']:
        rows = cls._db.fetchall(f"SELECT * FROM {cls._table}")
        clean_columns = ['line_text'] if for_stats else []
        return [cls.from_row(row, clean_columns=clean_columns) for row in rows]

    @classmethod
    def get_all_lines_for_scene(cls, game_name: str) -> List['GameLinesTable']:
        rows = cls._db.fetchall(
            f"SELECT * FROM {cls._table} WHERE game_name=?", (game_name,))
        return [cls.from_row(row) for row in rows]

    @classmethod
    def get_all_games_with_lines(cls) -> List[str]:
        rows = cls._db.fetchall(f"SELECT DISTINCT game_name FROM {cls._table}")
        return [row[0] for row in rows if row[0] is not None]

    @classmethod
    def update(cls, line_id: str, audio_in_anki: Optional[str] = None, screenshot_in_anki: Optional[str] = None, audio_path: Optional[str] = None, screenshot_path: Optional[str] = None, replay_path: Optional[str] = None, translation: Optional[str] = None, note_id: Optional[str] = None):
        line = cls.get(line_id)
        if not line:
            logger.warning(f"GameLine with id {line_id} not found for update, maybe testing?")
            return
        if screenshot_path is not None:
            line.screenshot_path = screenshot_path
        if audio_path is not None:
            line.audio_path = audio_path
        if replay_path is not None:
            line.replay_path = replay_path
        if screenshot_in_anki is not None:
            line.screenshot_in_anki = screenshot_in_anki
        if audio_in_anki is not None:
            line.audio_in_anki = audio_in_anki
        if translation is not None:
            line.translation = translation
        if note_id is not None:
            if note_id not in line.note_ids:
                if not line.note_ids:
                    line.note_ids = []
                line.note_ids.append(note_id)
        line.last_modified = time.time()
        line.save()
        logger.debug(f"Updated GameLine id={line_id} paths.")

    @classmethod
    def add_line(cls, gameline: GameLine, game_id: Optional[str] = None):
        if get_config().advanced.dont_collect_stats:
            return None
        new_line = cls(id=gameline.id, game_name=gameline.scene,
                       line_text=gameline.text, timestamp=gameline.time.timestamp(),
                       game_id=game_id if game_id else '')
        # logger.info("Adding GameLine to DB: %s", new_line)
        new_line.add()

        if get_config().advanced.enable_realtime_tokenization:
            from GameSentenceMiner.util.tokenization_service import get_tokenization_service
            try:
                get_tokenization_service().tokenize_line(
                    gameline.id,
                    gameline.text,
                    gameline.time.timestamp(),
                    game_id or ''
                )
            except Exception as e:
                logger.error(f"Realtime tokenization failed for line {gameline.id}: {e}")

        return new_line
    
    @classmethod
    def add_lines(cls, gamelines: List[GameLine]):
        new_lines = [cls(id=gl.id, game_name=gl.scene,
                         line_text=gl.text, timestamp=gl.time.timestamp()) for gl in gamelines]
        # logger.info("Adding %d GameLines to DB", len(new_lines))
        params = [(line.id, line.game_name, line.line_text, line.timestamp, line.screenshot_in_anki,
                   line.audio_in_anki, line.screenshot_path, line.audio_path, line.replay_path, line.translation)
                  for line in new_lines]
        cls._db.executemany(
            f"INSERT INTO {cls._table} (id, game_name, line_text, timestamp, screenshot_in_anki, audio_in_anki, screenshot_path, audio_path, replay_path, translation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params,
            commit=True
        )
        
    @classmethod
    def get_lines_filtered_by_timestamp(cls, start: Optional[float] = None, end: Optional[float] = None, for_stats=False) -> List['GameLinesTable']:
        """
        Fetches all lines optionally filtered by start and end timestamps.
        If start or end is None, that bound is ignored.
        """
        query = f"SELECT * FROM {cls._table}"
        conditions = []
        params = []

        # Add timestamp conditions if provided
        if start is not None:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            conditions.append("timestamp <= ?")
            params.append(end)

        # Combine conditions into WHERE clause if any
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Sort by timestamp ascending
        query += " ORDER BY timestamp ASC"

        # Execute the query
        rows = cls._db.fetchall(query, tuple(params))
        clean_columns = ['line_text'] if for_stats else []
        return [cls.from_row(row, clean_columns=clean_columns) for row in rows]


class GoalsTable(SQLiteDBTable):
    """
    Table for tracking daily goal completions and streaks.
    One entry per day when user completes their dailies.
    Includes version tracking for multi-device synchronization.
    """
    _table = 'goals'
    _fields = ['date', 'current_goals', 'goals_settings', 'last_updated', 'goals_version']
    _types = [
        int,   # id (primary key)
        str,   # date
        str,   # current_goals
        str,   # goals_settings
        float, # last_updated
        str    # goals_version (JSON: {"goals": 1, "easyDays": 1, "ankiConnect": 1})
    ]
    _pk = 'id'
    _auto_increment = True

    def __init__(self, id: Optional[int] = None, date: Optional[str] = None,
                 current_goals: Optional[str] = None,
                 goals_settings: Optional[str] = None, last_updated: Optional[float] = None,
                 goals_version: Optional[str] = None):
        self.id = id
        self.date = date if date is not None else ''
        self.current_goals = current_goals if current_goals is not None else '[]'
        self.goals_settings = goals_settings if goals_settings is not None else '{}'
        self.last_updated = last_updated
        self.goals_version = goals_version if goals_version is not None else '{"goals": 0, "easyDays": 0, "ankiConnect": 0}'

    @classmethod
    def get_by_date(cls, date_str: str) -> Optional['GoalsTable']:
        """Get goals entry for a specific date (YYYY-MM-DD format)."""
        row = cls._db.fetchone(
            f"SELECT * FROM {cls._table} WHERE date=?", (date_str,))
        return cls.from_row(row) if row else None

    @classmethod
    def get_latest(cls) -> Optional['GoalsTable']:
        """Get the most recent goals entry (excludes 'current' entry)."""
        row = cls._db.fetchone(
            f"SELECT * FROM {cls._table} WHERE date != 'current' ORDER BY date DESC LIMIT 1")
        return cls.from_row(row) if row else None

    @classmethod
    def create_entry(cls, date_str: str, current_goals_json: str,
                     goals_settings_json: Optional[str] = None, last_updated: Optional[float] = None,
                     goals_version: Optional[str] = None) -> 'GoalsTable':
        """Create a new goals entry for a specific date with version tracking."""
        # Default version if not provided
        if goals_version is None:
            goals_version = json.dumps({"goals": 1, "easyDays": 1, "ankiConnect": 1})
        
        new_entry = cls(date=date_str, current_goals=current_goals_json,
                       goals_settings=goals_settings_json if goals_settings_json else '{}',
                       last_updated=last_updated if last_updated is not None else time.time(),
                       goals_version=goals_version)
        new_entry.save()
        return new_entry

    @classmethod
    def calculate_streak(cls, today_str: str, timezone_str: str = 'UTC') -> Tuple[int, int]:
        """
        Calculate the current streak and longest streak for today.
        Returns tuple of (current_streak, longest_streak).
        
        Current streak: consecutive days ending today or yesterday.
        This means that if you complete yesterday's goals today (i.e., you missed marking them as complete on the actual day),
        your streak will continue, allowing you to maintain your streak even if you finish dailies a day late.
        
        Longest streak: maximum consecutive days from all historical data (stored in goals_settings JSON).
        
        Args:
            today_str: Date string in YYYY-MM-DD format (should be in user's local timezone)
            timezone_str: IANA timezone string (e.g., 'Asia/Tokyo', 'UTC') - used for logging only
        """
        try:
            today_date = datetime.strptime(today_str, '%Y-%m-%d').date()
        except (ValueError, AttributeError) as e:
            logger.error(f"Invalid date format for today_str: {today_str}: {e}")
            return (1, 1)
        
        # Get the latest historical entry (excludes "current")
        latest = cls.get_latest()
        
        if not latest:
            # No historical entries yet, this will be the first
            return (1, 1)
        
        # Parse the latest entry's date
        try:
            latest_date = datetime.strptime(latest.date, '%Y-%m-%d').date()
        except (ValueError, AttributeError) as e:
            logger.error(f"Invalid date in latest entry: {latest.date}: {e}")
            return (1, 1)
        
        # Calculate current streak
        current_streak = 0
        yesterday = today_date - timedelta(days=1)
        
        # Check if streak is still active (latest entry is today or yesterday)
        if latest_date == today_date or latest_date == yesterday:
            # Get all historical entries to count consecutive days backwards
            # Fetch entries in descending order (most recent first)
            query = f"SELECT * FROM {cls._table} WHERE date != 'current' ORDER BY date DESC"
            rows = cls._db.fetchall(query)
            
            if rows:
                current_streak = 1
                prev_date = datetime.strptime(rows[0][1], '%Y-%m-%d').date()  # rows[0][1] is the date column
                
                # Count consecutive days backwards from the most recent entry
                for i in range(1, len(rows)):
                    entry_date = datetime.strptime(rows[i][1], '%Y-%m-%d').date()
                    expected_prev_date = prev_date - timedelta(days=1)
                    
                    if entry_date == expected_prev_date:
                        current_streak += 1
                        prev_date = entry_date
                    else:
                        # Streak is broken
                        break
        else:
            # Streak is broken (latest entry is older than yesterday)
            current_streak = 0
        
        # Get longest streak from goals_settings JSON
        previous_longest = 0
        try:
            if latest.goals_settings:
                settings = json.loads(latest.goals_settings) if isinstance(latest.goals_settings, str) else latest.goals_settings
                previous_longest = settings.get('longestStreak', 0)
        except (json.JSONDecodeError, AttributeError):
            previous_longest = 0
        
        # Calculate new longest streak
        longest_streak = max(current_streak, previous_longest)
        
        return (current_streak, longest_streak)


class WordsTable(SQLiteDBTable):
    """
    Table for storing unique words with frequency tracking.
    Each row is unique by (headword, word, reading) combination.
    Uses INTEGER auto-increment primary key for normalized schema.
    """

    _table = 'words'
    _fields = [
        'headword',    # TEXT (base/dictionary form)
        'word',        # TEXT (surface form as appeared)
        'reading',     # TEXT (katakana reading, "-" if none/same)
        'first_seen',  # REAL (Unix timestamp)
        'last_seen',   # REAL (Unix timestamp)
        'frequency',   # INTEGER (occurrence count)
        'pos'          # TEXT (part of speech from Mecab)
    ]
    _types = [int, str, str, str, float, float, int, str]  # Includes primary key type (INTEGER)
    _pk = 'id'
    _auto_increment = True  # Use INTEGER auto-increment primary key
    _indexes = [
        ('word', False, 'idx_words_word'),
        (('headword', 'word', 'reading'), True, 'idx_words_unique'),
    ]

    def __init__(self, id: int = None, headword: str = '', word: str = '', reading: str = '-',
                 first_seen: float = 0.0, last_seen: float = 0.0, frequency: int = 0, pos: str = ''):
        super().__init__()
        self.id = id
        self.headword = headword
        self.word = word
        self.reading = reading
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.frequency = frequency
        self.pos = pos

    @classmethod
    def get_by_unique_key(cls, headword: str, word: str, reading: str) -> Optional['WordsTable']:
        """Get a word by its unique key (headword, word, reading)."""
        result = cls._db.fetchone(
            f"SELECT * FROM {cls._table} WHERE headword=? AND word=? AND reading=?",
            (headword, word, reading)
        )
        if result:
            return cls.from_row(result)
        return None

    @classmethod
    def get_by_word(cls, word: str, reading: str = None) -> Optional[dict]:
        """
        Get aggregated stats for a word (surface form), optionally filtered by reading.
        Returns dict with total_frequency summed across all matching rows.
        """
        if reading:
            results = cls._db.fetchall(
                f"SELECT * FROM {cls._table} WHERE word=? AND reading=?",
                (word, reading)
            )
        else:
            results = cls._db.fetchall(
                f"SELECT * FROM {cls._table} WHERE word=?",
                (word,)
            )
        if not results:
            return None

        rows = [cls.from_row(r) for r in results]
        return {
            'word': word,
            'total_frequency': sum(r.frequency for r in rows),
            'rows': rows
        }

    @classmethod
    def get_by_headword(cls, headword: str, reading: str = None) -> Optional[dict]:
        """
        Get aggregated stats for a headword (dictionary form), optionally filtered by reading.
        Returns dict with total_frequency summed across all matching rows.
        """
        if reading:
            results = cls._db.fetchall(
                f"SELECT * FROM {cls._table} WHERE headword=? AND reading=?",
                (headword, reading)
            )
        else:
            results = cls._db.fetchall(
                f"SELECT * FROM {cls._table} WHERE headword=?",
                (headword,)
            )
        if not results:
            return None

        rows = [cls.from_row(r) for r in results]
        return {
            'headword': headword,
            'total_frequency': sum(r.frequency for r in rows),
            'rows': rows
        }

    @classmethod
    def get_or_create_word(cls, headword: str, word: str, reading: str,
                           timestamp: float, pos: str = '') -> 'WordsTable':
        """
        Get existing word or create new one.
        Updates last_seen and increments frequency on existing words.
        Uniqueness is based on (headword, word, reading) combination.
        """
        existing = cls.get_by_unique_key(headword, word, reading)

        if existing:
            existing.last_seen = timestamp
            existing.frequency += 1
            existing.save()
            return existing
        else:
            new_word = cls(
                headword=headword,
                word=word,
                reading=reading,
                first_seen=timestamp,
                last_seen=timestamp,
                frequency=1,
                pos=pos
            )
            new_word.save()
            return new_word

    @classmethod
    def recalculate_frequency_for_ids(cls, word_ids: List[int]) -> int:
        """Recalculate frequency for specific word IDs by counting remaining occurrences."""
        if not word_ids:
            return 0

        placeholders = ','.join(['?' for _ in word_ids])
        counts = cls._db.fetchall(
            f"SELECT word_id, COUNT(*) FROM word_occurrences WHERE word_id IN ({placeholders}) GROUP BY word_id",
            tuple(word_ids)
        )
        count_map = {row[0]: row[1] for row in counts}
        updates = [(count_map.get(wid, 0), wid) for wid in word_ids]
        cls._db.execute_in_transaction([
            (f"UPDATE {cls._table} SET frequency=? WHERE id=?", updates)
        ])

        return len(word_ids)

    @classmethod
    def recalculate_all_frequencies(cls) -> int:
        """Recalculate frequency for ALL words using a single efficient query."""
        cls._db.execute("""
            UPDATE words SET frequency = (
                SELECT COUNT(*) FROM word_occurrences WHERE word_occurrences.word_id = words.id
            )
        """, commit=True)
        result = cls._db.fetchone(f"SELECT COUNT(*) FROM {cls._table}")
        return result[0] if result else 0

    @classmethod
    def bulk_get_by_keys(cls, keys: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], 'WordsTable']:
        """
        Get multiple words by their unique keys (headword, word, reading) in a single query.
        Returns a dictionary mapping (headword, word, reading) tuples to WordsTable objects.

        Args:
            keys: List of (headword, word, reading) tuples

        Returns:
            Dictionary mapping keys to WordsTable objects for existing words
        """
        if not keys:
            return {}

        # Build query with OR conditions for each key
        # SQLite doesn't support tuple IN, so we use OR conditions
        # Chunk to avoid SQLite expression tree depth limit (1000)
        # Each key adds ~3 depth, so 250 keys = ~750 depth (safe margin)
        CHUNK_SIZE = 250
        result_dict = {}

        for i in range(0, len(keys), CHUNK_SIZE):
            chunk = keys[i:i + CHUNK_SIZE]
            conditions = []
            params = []
            for headword, word, reading in chunk:
                conditions.append("(headword=? AND word=? AND reading=?)")
                params.extend([headword, word, reading])

            query = f"SELECT * FROM {cls._table} WHERE " + " OR ".join(conditions)
            results = cls._db.fetchall(query, tuple(params))

            for row in results:
                obj = cls.from_row(row)
                result_dict[(obj.headword, obj.word, obj.reading)] = obj

        return result_dict

    @classmethod
    def bulk_insert(cls, words_data: List[Tuple[str, str, str, float, float, int, str]]) -> None:
        """
        Bulk insert multiple words in a single transaction.

        Args:
            words_data: List of tuples (headword, word, reading, first_seen, last_seen, frequency, pos)
        """
        if not words_data:
            return

        query = f"""
            INSERT INTO {cls._table} (headword, word, reading, first_seen, last_seen, frequency, pos)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cls._db.executemany(query, words_data, commit=True)

    @classmethod
    def bulk_update_frequency(cls, updates: List[Tuple[float, int, int]]) -> None:
        """
        Bulk update last_seen and frequency for multiple words.

        Args:
            updates: List of tuples (last_seen, frequency_increment, word_id)
                     Note: This adds to the current frequency, not sets it
        """
        if not updates:
            return

        # Use a single UPDATE with CASE for efficiency
        query = f"""
            UPDATE {cls._table}
            SET last_seen = ?, frequency = frequency + ?
            WHERE id = ?
        """
        cls._db.executemany(query, updates, commit=True)


class KanjiTable(SQLiteDBTable):
    """
    Table for storing unique kanji characters with frequency tracking.
    Uses INTEGER auto-increment primary key for normalized schema.
    """

    _table = 'kanji'
    _fields = [
        'kanji',  # TEXT (the kanji character) with UNIQUE constraint
        'first_seen',  # REAL (Unix timestamp)
        'last_seen',  # REAL (Unix timestamp)
        'frequency'  # INTEGER (total occurrence count)
    ]
    _types = [int, str, float, float, int]  # Includes primary key type (INTEGER)
    _pk = 'id'
    _auto_increment = True  # Use INTEGER auto-increment primary key
    _indexes = [
        ('kanji', True, 'idx_kanji_char'),
    ]

    def __init__(self, id: int = None, kanji: str = '', first_seen: float = 0.0,
                 last_seen: float = 0.0, frequency: int = 0):
        super().__init__()
        self.id = id
        self.kanji = kanji
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.frequency = frequency

    @classmethod
    def get_by_character(cls, character: str) -> Optional['KanjiTable']:
        """Get a kanji by its character."""
        result = cls._db.fetchone(
            f"SELECT * FROM {cls._table} WHERE kanji=?",
            (character,)
        )
        if result:
            return cls.from_row(result)
        return None

    @classmethod
    def get_or_create_kanji(cls, character: str, timestamp: float) -> 'KanjiTable':
        """
        Get existing kanji or create new one.
        Updates last_seen and increments frequency on existing kanji.
        """
        existing = cls.get_by_character(character)

        if existing:
            existing.last_seen = timestamp
            existing.frequency += 1
            existing.save()
            return existing
        else:
            new_kanji = cls(
                kanji=character,
                first_seen=timestamp,
                last_seen=timestamp,
                frequency=1
            )
            new_kanji.save()
            return new_kanji

    @classmethod
    def recalculate_frequency_for_ids(cls, kanji_ids: List[int]) -> int:
        """Recalculate frequency for specific kanji IDs by counting remaining occurrences."""
        if not kanji_ids:
            return 0

        placeholders = ','.join(['?' for _ in kanji_ids])
        counts = cls._db.fetchall(
            f"SELECT kanji_id, COUNT(*) FROM kanji_occurrences WHERE kanji_id IN ({placeholders}) GROUP BY kanji_id",
            tuple(kanji_ids)
        )
        count_map = {row[0]: row[1] for row in counts}
        updates = [(count_map.get(kid, 0), kid) for kid in kanji_ids]
        cls._db.execute_in_transaction([
            (f"UPDATE {cls._table} SET frequency=? WHERE id=?", updates)
        ])

        return len(kanji_ids)

    @classmethod
    def recalculate_all_frequencies(cls) -> int:
        """Recalculate frequency for ALL kanji using a single efficient query."""
        cls._db.execute("""
            UPDATE kanji SET frequency = (
                SELECT COUNT(*) FROM kanji_occurrences WHERE kanji_occurrences.kanji_id = kanji.id
            )
        """, commit=True)
        result = cls._db.fetchone(f"SELECT COUNT(*) FROM {cls._table}")
        return result[0] if result else 0

    @classmethod
    def bulk_get_by_characters(cls, characters: List[str]) -> Dict[str, 'KanjiTable']:
        """
        Get multiple kanji by their characters in a single query.
        Returns a dictionary mapping characters to KanjiTable objects.

        Args:
            characters: List of kanji characters

        Returns:
            Dictionary mapping characters to KanjiTable objects for existing kanji
        """
        if not characters:
            return {}

        # Chunk to avoid SQLite parameter limit (999 in older versions)
        CHUNK_SIZE = 500
        result_dict = {}

        for i in range(0, len(characters), CHUNK_SIZE):
            chunk = characters[i:i + CHUNK_SIZE]
            placeholders = ','.join(['?' for _ in chunk])
            query = f"SELECT * FROM {cls._table} WHERE kanji IN ({placeholders})"
            results = cls._db.fetchall(query, tuple(chunk))

            for row in results:
                obj = cls.from_row(row)
                result_dict[obj.kanji] = obj

        return result_dict

    @classmethod
    def bulk_insert(cls, kanji_data: List[Tuple[str, float, float, int]]) -> None:
        """
        Bulk insert multiple kanji in a single transaction.

        Args:
            kanji_data: List of tuples (kanji, first_seen, last_seen, frequency)
        """
        if not kanji_data:
            return

        query = f"""
            INSERT INTO {cls._table} (kanji, first_seen, last_seen, frequency)
            VALUES (?, ?, ?, ?)
        """
        cls._db.executemany(query, kanji_data, commit=True)

    @classmethod
    def bulk_update_frequency(cls, updates: List[Tuple[float, int, int]]) -> None:
        """
        Bulk update last_seen and frequency for multiple kanji.

        Args:
            updates: List of tuples (last_seen, frequency_increment, kanji_id)
                     Note: This adds to the current frequency, not sets it
        """
        if not updates:
            return

        query = f"""
            UPDATE {cls._table}
            SET last_seen = ?, frequency = frequency + ?
            WHERE id = ?
        """
        cls._db.executemany(query, updates, commit=True)


class WordOccurrencesTable(SQLiteDBTable):
    """
    Normalized mapping table linking words to game lines and games.
    Uses integer word_id FK instead of denormalized text columns.
    """

    _table = 'word_occurrences'
    _fields = [
        'word_id',    # INTEGER (FK to words.id)
        'line_id',    # TEXT (FK to game_lines.id)
        'game_id',    # TEXT (FK to games.id)
        'timestamp',  # REAL (Unix timestamp from game_lines)
        'position'    # INTEGER (position of word in sentence)
    ]
    _types = [int, int, str, str, float, int]  # Includes primary key type (INTEGER)
    _pk = 'id'
    _auto_increment = True  # Use INTEGER auto-increment primary key
    _foreign_keys = [
        ('word_id', 'words', 'id', 'CASCADE'),
        ('line_id', 'game_lines', 'id', 'CASCADE'),
        ('game_id', 'games', 'id', 'CASCADE'),
    ]
    _indexes = [
        ('word_id', False, 'idx_word_occ_word_id'),
        ('line_id', False, 'idx_word_occ_line_id'),
        (('game_id', 'timestamp'), False, 'idx_word_occ_game_timestamp'),
        (('word_id', 'line_id', 'position'), True, 'idx_word_occ_unique'),
    ]

    def __init__(self, id: int = None, word_id: int = None, line_id: str = '',
                 game_id: str = None, timestamp: float = 0.0, position: int = 0):
        super().__init__()
        self.id = id
        self.word_id = word_id
        self.line_id = line_id
        # Use None instead of empty string for optional FK to avoid constraint failures
        self.game_id = game_id if game_id else None
        self.timestamp = timestamp
        self.position = position

    @classmethod
    def add_mapping(cls, word_id: int, line_id: str, game_id: str = None,
                    timestamp: float = 0.0, position: int = 0):
        """Create a new word occurrence mapping."""
        mapping = cls(
            word_id=word_id,
            line_id=line_id,
            game_id=game_id if game_id else None,
            timestamp=timestamp,
            position=position
        )
        mapping.save()
        return mapping

    @classmethod
    def get_words_for_line(cls, line_id: str) -> list:
        """Get all word occurrences for a game line."""
        results = cls._db.fetchall(
            f"SELECT * FROM {cls._table} WHERE line_id=?",
            (line_id,)
        )
        return [cls.from_row(result) for result in results]

    @classmethod
    def get_words_for_line_with_details(cls, line_id: str) -> list:
        """Get word occurrences with word details via JOIN."""
        return cls._db.fetchall("""
            SELECT wo.id, wo.word_id, wo.line_id, wo.game_id, wo.timestamp, wo.position,
                   w.headword, w.word, w.reading, w.pos, w.frequency
            FROM word_occurrences wo
            JOIN words w ON wo.word_id = w.id
            WHERE wo.line_id = ?
            ORDER BY wo.position
        """, (line_id,))

    @classmethod
    def get_occurrences_for_word(cls, word_id: int) -> list:
        """Get all occurrences of a specific word."""
        results = cls._db.fetchall(
            f"SELECT * FROM {cls._table} WHERE word_id=?",
            (word_id,)
        )
        return [cls.from_row(result) for result in results]

    @classmethod
    def get_occurrences_for_game(cls, game_id: str) -> list:
        """Get all word occurrences for a specific game."""
        results = cls._db.fetchall(
            f"SELECT * FROM {cls._table} WHERE game_id=?",
            (game_id,)
        )
        return [cls.from_row(result) for result in results]

    @classmethod
    def bulk_insert(cls, occurrences_data: List[Tuple[int, str, str, float, int]]) -> None:
        """
        Bulk insert multiple word occurrences in a single transaction.

        Args:
            occurrences_data: List of tuples (word_id, line_id, game_id, timestamp, position)
        """
        if not occurrences_data:
            return

        query = f"""
            INSERT OR IGNORE INTO {cls._table} (word_id, line_id, game_id, timestamp, position)
            VALUES (?, ?, ?, ?, ?)
        """
        cls._db.executemany(query, occurrences_data, commit=True)


class KanjiOccurrencesTable(SQLiteDBTable):
    """
    Normalized mapping table linking kanji to game lines and games.
    Uses integer kanji_id FK instead of denormalized text column.
    """

    _table = 'kanji_occurrences'
    _fields = [
        'kanji_id',   # INTEGER (FK to kanji.id)
        'line_id',    # TEXT (FK to game_lines.id)
        'game_id',    # TEXT (FK to games.id)
        'timestamp',  # REAL (Unix timestamp from game_lines)
        'position'    # INTEGER (position of kanji in sentence)
    ]
    _types = [int, int, str, str, float, int]  # Includes primary key type (INTEGER)
    _pk = 'id'
    _auto_increment = True  # Use INTEGER auto-increment primary key
    _foreign_keys = [
        ('kanji_id', 'kanji', 'id', 'CASCADE'),
        ('line_id', 'game_lines', 'id', 'CASCADE'),
        ('game_id', 'games', 'id', 'CASCADE'),
    ]
    _indexes = [
        ('kanji_id', False, 'idx_kanji_occ_kanji_id'),
        ('line_id', False, 'idx_kanji_occ_line_id'),
        (('game_id', 'timestamp'), False, 'idx_kanji_occ_game_timestamp'),
        (('kanji_id', 'line_id', 'position'), True, 'idx_kanji_occ_unique'),
    ]

    def __init__(self, id: int = None, kanji_id: int = None, line_id: str = '',
                 game_id: str = None, timestamp: float = 0.0, position: int = 0):
        super().__init__()
        self.id = id
        self.kanji_id = kanji_id
        self.line_id = line_id
        # Use None instead of empty string for optional FK to avoid constraint failures
        self.game_id = game_id if game_id else None
        self.timestamp = timestamp
        self.position = position

    @classmethod
    def add_mapping(cls, kanji_id: int, line_id: str, game_id: str = None,
                    timestamp: float = 0.0, position: int = 0):
        """Create a new kanji occurrence mapping."""
        mapping = cls(
            kanji_id=kanji_id,
            line_id=line_id,
            game_id=game_id if game_id else None,
            timestamp=timestamp,
            position=position
        )
        mapping.save()
        return mapping

    @classmethod
    def get_kanji_for_line(cls, line_id: str) -> list:
        """Get all kanji occurrences for a game line."""
        results = cls._db.fetchall(
            f"SELECT * FROM {cls._table} WHERE line_id=?",
            (line_id,)
        )
        return [cls.from_row(result) for result in results]

    @classmethod
    def get_kanji_for_line_with_details(cls, line_id: str) -> list:
        """Get kanji occurrences with kanji details via JOIN."""
        return cls._db.fetchall("""
            SELECT ko.id, ko.kanji_id, ko.line_id, ko.game_id, ko.timestamp, ko.position,
                   k.kanji, k.frequency
            FROM kanji_occurrences ko
            JOIN kanji k ON ko.kanji_id = k.id
            WHERE ko.line_id = ?
            ORDER BY ko.position
        """, (line_id,))

    @classmethod
    def get_occurrences_for_kanji(cls, kanji_id: int) -> list:
        """Get all occurrences of a specific kanji."""
        results = cls._db.fetchall(
            f"SELECT * FROM {cls._table} WHERE kanji_id=?",
            (kanji_id,)
        )
        return [cls.from_row(result) for result in results]

    @classmethod
    def get_occurrences_for_game(cls, game_id: str) -> list:
        """Get all kanji occurrences for a specific game."""
        results = cls._db.fetchall(
            f"SELECT * FROM {cls._table} WHERE game_id=?",
            (game_id,)
        )
        return [cls.from_row(result) for result in results]

    @classmethod
    def bulk_insert(cls, occurrences_data: List[Tuple[int, str, str, float, int]]) -> None:
        """
        Bulk insert multiple kanji occurrences in a single transaction.

        Args:
            occurrences_data: List of tuples (kanji_id, line_id, game_id, timestamp, position)
        """
        if not occurrences_data:
            return

        query = f"""
            INSERT INTO {cls._table} (kanji_id, line_id, game_id, timestamp, position)
            VALUES (?, ?, ?, ?, ?)
        """
        cls._db.executemany(query, occurrences_data, commit=True)


# Ensure database directory exists and return path
def get_db_directory(test=False, delete_test=False) -> str:
    if platform == 'win32':  # Windows
        appdata_dir = os.getenv('APPDATA')
    else:  # macOS and Linux
        appdata_dir = os.path.expanduser('~/.config')
    config_dir = os.path.join(appdata_dir, 'GameSentenceMiner')
    # Create the directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    path = os.path.join(config_dir, 'gsm.db' if not test else 'gsm_test.db')
    if test and delete_test:
        if os.path.exists(path):
            os.remove(path)
    return sanitize_and_resolve_path(path)


# Backup and compress the database on load, with today's date, up to 5 days ago (clean up old backups)
def backup_db(db_path: str):
    
    # Create a of the backups on migration
    pre_jiten_merge_backup = os.path.join(os.path.dirname(db_path), "backup", "database", "pre_jiten")
    if not os.path.exists(pre_jiten_merge_backup):
        os.makedirs(pre_jiten_merge_backup, exist_ok=True)
        for fname in os.listdir(os.path.join(os.path.dirname(db_path), "backup", "database")):
            fpath = os.path.join(os.path.dirname(db_path), "backup", "database", fname)
            if os.path.isfile(fpath):
                shutil.copy2(fpath, pre_jiten_merge_backup)
                
    backup_dir = os.path.join(os.path.dirname(db_path), "backup", "database")
    os.makedirs(backup_dir, exist_ok=True)
    today = time.strftime("%Y-%m-%d")
    backup_file = os.path.join(backup_dir, f"gsm_{today}.db.gz")
    
    # Test, remove backups older than 60 minutes
    # cutoff = time.time() - 60 * 60
    # Clean up backups older than 5 days
    cutoff = time.time() - 5 * 24 * 60 * 60
    for fname in os.listdir(backup_dir):
        fpath = os.path.join(backup_dir, fname)
        if fname.startswith("gsm_") and fname.endswith(".db.gz"):
            try:
                file_time = os.path.getmtime(fpath)
                if file_time < cutoff:
                    os.remove(fpath)
                    logger.info(f"Old backup removed: {fpath}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {fpath}: {e}")

    # Create backup if not already present for today
    if not os.path.exists(backup_file):
        with open(db_path, "rb") as f_in, open(backup_file, "wb") as f_out:
            with gzip.GzipFile(fileobj=f_out, mode="wb") as gz_out:
                shutil.copyfileobj(f_in, gz_out)
        logger.success(f"Database backup created: {backup_file}")


db_path = get_db_directory()
if os.path.exists(db_path):
    backup_db(db_path)

# db_path = get_db_directory(test=True, delete_test=False)

# Default: normal read/write, but allow environment variable to override for read-only mode
_gsm_db_read_only = os.environ.get("GSM_DB_READ_ONLY", "0") == "1"
gsm_db = SQLiteDB(db_path, read_only=_gsm_db_read_only)

# Import GamesTable, CronTable, and StatsRollupTable after gsm_db is created to avoid circular import
from GameSentenceMiner.util.games_table import GamesTable
from GameSentenceMiner.util.cron_table import CronTable
from GameSentenceMiner.util.stats_rollup_table import StatsRollupTable

for cls in [AIModelsTable, GameLinesTable, GoalsTable, GamesTable, CronTable, StatsRollupTable,
            WordsTable, KanjiTable, WordOccurrencesTable, KanjiOccurrencesTable]:
    cls.set_db(gsm_db)
    # Uncomment to start fresh every time
    # cls.drop()
    # cls.set_db(gsm_db)  # --- IGNORE ---
    
logger.info("Database initialized at {}", db_path)
# GameLinesTable.drop_column('timestamp')

# if GameLinesTable.has_column('timestamp_old'):
#     GameLinesTable.alter_column_type('timestamp_old', 'timestamp', 'TEXT')
#     logger.info("Altered 'timestamp_old' column to 'timestamp' with TEXT type in GameLinesTable.")


def get_affected_word_kanji_ids(line_ids: List[str]) -> Tuple[List[int], List[int]]:
    """Get word_ids and kanji_ids that have occurrences for the given line_ids.
    Call BEFORE deleting lines to capture affected IDs."""
    if not line_ids:
        return ([], [])

    # Chunk to avoid SQLite parameter limit
    CHUNK_SIZE = 500
    word_ids = set()
    kanji_ids = set()

    for i in range(0, len(line_ids), CHUNK_SIZE):
        chunk = line_ids[i:i + CHUNK_SIZE]
        placeholders = ','.join(['?' for _ in chunk])

        word_results = WordOccurrencesTable._db.fetchall(
            f"SELECT DISTINCT word_id FROM word_occurrences WHERE line_id IN ({placeholders})",
            tuple(chunk)
        )
        word_ids.update(row[0] for row in word_results)

        kanji_results = KanjiOccurrencesTable._db.fetchall(
            f"SELECT DISTINCT kanji_id FROM kanji_occurrences WHERE line_id IN ({placeholders})",
            tuple(chunk)
        )
        kanji_ids.update(row[0] for row in kanji_results)

    return (list(word_ids), list(kanji_ids))


def recalculate_frequencies_after_deletion(word_ids: List[int], kanji_ids: List[int]) -> dict:
    """Recalculate word and kanji frequencies. Call AFTER deleting lines."""
    words_updated = WordsTable.recalculate_frequency_for_ids(word_ids)
    kanji_updated = KanjiTable.recalculate_frequency_for_ids(kanji_ids)
    return {'words_updated': words_updated, 'kanji_updated': kanji_updated}


def check_and_run_migrations():
    def migrate_timestamp():
        if GameLinesTable.has_column('timestamp') and GameLinesTable.get_column_type('timestamp') != 'REAL':
            logger.info("Migrating 'timestamp' column to REAL type in GameLinesTable.")
            # Rename 'timestamp' to 'timestamp_old'
            GameLinesTable.rename_column('timestamp', 'timestamp_old')
            # Copy and cast data from old column to new column
            GameLinesTable.alter_column_type('timestamp_old', 'timestamp', 'REAL')
            logger.success("Migrated 'timestamp' column to REAL type in GameLinesTable.")
    
    def migrate_obs_scene_name():
        """
        Add obs_scene_name column to games table and populate it from game_lines.
        This migration ensures existing games have their OBS scene names preserved.
        """
        if not GamesTable.has_column('obs_scene_name'):
            logger.info("Adding 'obs_scene_name' column to games table...")
            GamesTable._db.execute(
                f"ALTER TABLE {GamesTable._table} ADD COLUMN obs_scene_name TEXT",
                commit=True
            )
            logger.info("Added 'obs_scene_name' column to games table.")
            
            # Populate obs_scene_name for existing games by querying game_lines
            logger.info("Populating obs_scene_name from game_lines...")
            all_games = GamesTable.all()
            updated_count = 0
            
            for game in all_games:
                # Find the first game_line with this game_id to get the original game_name
                result = GameLinesTable._db.fetchone(
                    f"SELECT game_name FROM {GameLinesTable._table} WHERE game_id=? LIMIT 1",
                    (game.id,)
                )
                
                if result and result[0]:
                    obs_scene_name = result[0]
                    # Update the game with the obs_scene_name
                    GamesTable._db.execute(
                        f"UPDATE {GamesTable._table} SET obs_scene_name=? WHERE id=?",
                        (obs_scene_name, game.id),
                        commit=True
                    )
                    updated_count += 1
                    logger.debug(f"Set obs_scene_name='{obs_scene_name}' for game id={game.id}")
            
            logger.info(f"Migration complete: Updated {updated_count} games with obs_scene_name from game_lines.")
        else:
            logger.debug("obs_scene_name column already exists in games table, skipping migration.")
        """
        Convert datetime strings in cron_table to Unix timestamps.
        This migration handles legacy data that may have datetime strings instead of floats.
        """
        try:
            # Get all rows directly from database to check for datetime strings
            rows = CronTable._db.fetchall(f"SELECT id, last_run, next_run, created_at FROM {CronTable._table}")
            
            updates_needed = []
            for row in rows:
                cron_id, last_run, next_run, created_at = row
                needs_update = False
                new_last_run = last_run
                new_next_run = next_run
                new_created_at = created_at
                
                # Check and convert last_run
                if last_run and isinstance(last_run, str) and not last_run.replace('.', '', 1).isdigit():
                    try:
                        dt = datetime.fromisoformat(last_run.replace(' ', 'T'))
                        new_last_run = dt.timestamp()
                        needs_update = True
                    except (ValueError, AttributeError):
                        logger.warning(f"Could not parse last_run '{last_run}' for cron id={cron_id}")
                        new_last_run = None
                        needs_update = True
                
                # Check and convert next_run
                if next_run and isinstance(next_run, str) and not next_run.replace('.', '', 1).isdigit():
                    try:
                        dt = datetime.fromisoformat(next_run.replace(' ', 'T'))
                        new_next_run = dt.timestamp()
                        needs_update = True
                    except (ValueError, AttributeError):
                        logger.warning(f"Could not parse next_run '{next_run}' for cron id={cron_id}")
                        new_next_run = time.time()
                        needs_update = True
                
                # Check and convert created_at
                if created_at and isinstance(created_at, str) and not created_at.replace('.', '', 1).isdigit():
                    try:
                        dt = datetime.fromisoformat(created_at.replace(' ', 'T'))
                        new_created_at = dt.timestamp()
                        needs_update = True
                    except (ValueError, AttributeError):
                        logger.warning(f"Could not parse created_at '{created_at}' for cron id={cron_id}")
                        new_created_at = time.time()
                        needs_update = True
                
                if needs_update:
                    updates_needed.append((new_last_run, new_next_run, new_created_at, cron_id))
            
            # Apply updates
            if updates_needed:
                logger.info(f"Migrating {len(updates_needed)} cron entries with datetime strings to Unix timestamps...")
                for new_last_run, new_next_run, new_created_at, cron_id in updates_needed:
                    CronTable._db.execute(
                        f"UPDATE {CronTable._table} SET last_run=?, next_run=?, created_at=? WHERE id=?",
                        (new_last_run, new_next_run, new_created_at, cron_id),
                        commit=True
                    )
                logger.success(f"✅ Migrated {len(updates_needed)} cron entries to Unix timestamps")
            else:
                logger.debug("No cron timestamp migration needed")
                
        except Exception as e:
            logger.error(f"Error during cron timestamp migration: {e}")
    
    def migrate_jiten_cron_job():
        """
        Create the daily jiten.moe update cron job if it doesn't exist.
        This ensures the cron job is automatically registered on database initialization.
        """
        existing_cron = CronTable.get_by_name('jiten_sync')
        if not existing_cron:
            logger.info("Creating daily jiten.moe update scheduled task...")
            # Calculate next run: yesterday to ensure it runs ASAP on first startup
            now = datetime.now()
            yesterday = now - timedelta(days=1)
            
            CronTable.create_cron_entry(
                name='jiten_sync',
                description='Automatically update all linked games from jiten.moe database (respects manual overrides)',
                next_run=yesterday.timestamp(),
                schedule='daily'
            )
            logger.success(f"✅ Created jiten_sync scheduled task - next run: {yesterday.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            if existing_cron.schedule == 'daily':
                logger.debug("jiten_sync scheduled task already set to daily schedule, skipping update.")
                return
            # Update existing cron to daily schedule and set next_run to yesterday to run ASAP
            now = datetime.now()
            yesterday = now - timedelta(days=1)
            existing_cron.schedule = 'daily'
            existing_cron.next_run = yesterday.timestamp()
            existing_cron.save()
            logger.success(f"✅ Updated jiten_sync scheduled task to daily schedule - next run: {yesterday.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def migrate_daily_rollup_cron_job():
        """
        Create the daily statistics rollup cron job if it doesn't exist.
        This ensures the cron job is automatically registered on database initialization.
        """
        existing_cron = CronTable.get_by_name('daily_stats_rollup')
        if not existing_cron:
            logger.info("Creating daily statistics rollup scheduled task...")
            # Schedule for 1 minute ago to ensure it runs immediately on first startup
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)
            
            CronTable.create_cron_entry(
                name='daily_stats_rollup',
                description='Roll up daily statistics for all dates up to yesterday',
                next_run=one_minute_ago.timestamp(),
                schedule='daily'
            )
            logger.info(f"✅ Created daily_stats_rollup scheduled task - scheduled to run immediately (next_run: {one_minute_ago.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            logger.debug("daily_stats_rollup scheduled task already exists, skipping creation.")
    
    def migrate_populate_games_cron_job():
        """
        Create the one-time populate_games cron job if it doesn't exist.
        This ensures games table is populated before the daily rollup runs.
        Runs once and auto-disables (schedule='once').
        """
        existing_cron = CronTable.get_by_name('populate_games')
        if not existing_cron:
            logger.info("Creating one-time populate_games scheduled task...")
            # Schedule to run immediately (2 minutes ago to ensure it runs before rollup)
            now = datetime.now()
            two_minutes_ago = now - timedelta(minutes=2)
            
            CronTable.create_cron_entry(
                name='populate_games',
                description='One-time auto-creation of game records from game_lines (runs before rollup)',
                next_run=two_minutes_ago.timestamp(),
                schedule='weekly'  # Will auto-disable after running
            )
            logger.info(f"✅ Created populate_games scheduled task - scheduled to run immediately (next_run: {two_minutes_ago.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            logger.debug("populate_games scheduled task already exists, skipping creation.")
    
    def migrate_user_plugins_cron_job():
        """
        Create the user_plugins cron job if it doesn't exist.
        """
        existing_cron = CronTable.get_by_name('user_plugins')
        if not existing_cron:
            logger.info("Creating user_plugins scheduled task...")
            # Schedule to run immediately (2 minutes ago)
            now = datetime.now()
            two_minutes_ago = now - timedelta(minutes=2)
            
            CronTable.create_cron_entry(
                name='user_plugins',
                description='Custom user plugins',
                next_run=two_minutes_ago.timestamp(),
                schedule='minutely'  # by default gsm checks crons every 5 mins, so this actually runs every 15 mins
            )
            logger.info(f"✅ Created user_plugins scheduled task - scheduled to run immediately (next_run: {two_minutes_ago.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            logger.debug("user_plugins scheduled task already exists, skipping creation.")
    
    def migrate_jiten_upgrader_cron_job():
        """
        Create the weekly jiten_upgrader cron job if it doesn't exist.
        This checks if VNDB/AniList-only games now exist on Jiten.moe and auto-links them.
        Runs weekly on Sunday at 3:00 AM local time.
        """
        existing_cron = CronTable.get_by_name('jiten_upgrader')
        if not existing_cron:
            logger.info("Creating weekly jiten_upgrader scheduled task...")
            
            # Calculate next Sunday at 3:00 AM local time
            now = datetime.now()
            days_until_sunday = (6 - now.weekday()) % 7
            
            # If today is Sunday and it's after 3 AM, schedule for next week
            if days_until_sunday == 0 and now.hour >= 3:
                days_until_sunday = 7
            
            next_sunday = now.replace(hour=3, minute=0, second=0, microsecond=0) + timedelta(days=days_until_sunday)
            
            CronTable.create_cron_entry(
                name='jiten_upgrader',
                description='Weekly check for games with VNDB/AniList IDs to auto-link to Jiten.moe',
                next_run=next_sunday.timestamp(),
                schedule='weekly'
            )
            logger.info(f"✅ Created jiten_upgrader scheduled task - next run: {next_sunday.strftime('%Y-%m-%d %H:%M:%S')} (Sunday 3:00 AM)")
        else:
            logger.debug("jiten_upgrader scheduled task already exists, skipping creation.")
    
    def migrate_genres_and_tags():
        """
        Add genres and tags columns to games table.
        These fields store data from jiten.moe API.
        After adding columns, trigger jiten update to populate the new fields.
        """
        columns_added = False
        
        if not GamesTable.has_column('genres'):
            logger.info("Adding 'genres' column to games table...")
            GamesTable._db.execute(
                f"ALTER TABLE {GamesTable._table} ADD COLUMN genres TEXT",
                commit=True
            )
            logger.success("✅ Added 'genres' column to games table.")
            columns_added = True
        else:
            logger.debug("genres column already exists in games table, skipping migration.")
        
        if not GamesTable.has_column('tags'):
            logger.info("Adding 'tags' column to games table...")
            GamesTable._db.execute(
                f"ALTER TABLE {GamesTable._table} ADD COLUMN tags TEXT",
                commit=True
            )
            logger.success("✅ Added 'tags' column to games table.")
            columns_added = True
        else:
            logger.debug("tags column already exists in games table, skipping migration.")
        
        # If we added columns, trigger jiten update to populate them
        if columns_added:
            logger.info("🔄 Triggering jiten update to populate genres and tags...")
            try:
                from GameSentenceMiner.util.cron.jiten_update import update_all_jiten_games
                result = update_all_jiten_games()
                logger.success(f"✅ Jiten update completed: {result['updated_games']} games updated with genres and tags")
            except Exception as e:
                logger.error(f"⚠️ Failed to auto-update games with genres/tags: {e}")
                logger.info("You can manually run the update with: python -m GameSentenceMiner.util.cron.jiten_update")

    def migrate_tokenization_backfill_cron():
        """Create one-time backfill cron job if needed"""
        try:
            from GameSentenceMiner.util.cron_table import CronTable

            existing = CronTable.get_by_name('backfill_tokenization')
            if not existing:
                logger.info("Creating tokenization backfill cron...")
                CronTable.create_cron_entry(
                    name='backfill_tokenization',
                    description='One-time backfill of word/kanji tokenization for existing game lines',
                    next_run=(datetime.now() - timedelta(minutes=1)).timestamp(),
                    schedule='once',
                    enabled=True
                )
                logger.info("✅ Created backfill_tokenization cron job")
        except Exception as e:
            logger.error(f"⚠️ Failed to create backfill_tokenization cron: {e}")

    def migrate_daily_tokenization_cron():
        """Create daily tokenization catchup cron."""
        try:
            from GameSentenceMiner.util.cron_table import CronTable

            existing = CronTable.get_by_name('daily_tokenization')
            if not existing:
                logger.info("Creating daily tokenization catchup cron...")
                # Run at 1:00 AM daily (after daily stats rollup at 00:01)
                next_run = datetime.now().replace(hour=1, minute=0, second=0, microsecond=0)
                if next_run < datetime.now():
                    next_run += timedelta(days=1)

                CronTable.create_cron_entry(
                    name='daily_tokenization',
                    description='Daily catchup to tokenize any missed game lines',
                    next_run=next_run.timestamp(),
                    schedule='daily',
                    enabled=True
                )
                logger.info(f"✅ Created daily_tokenization cron job - next run: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                logger.debug("daily_tokenization cron job already exists, skipping creation.")
        except Exception as e:
            logger.error(f"⚠️ Failed to create daily_tokenization cron: {e}")

    migrate_timestamp()
    migrate_obs_scene_name()
    # migrate_cron_timestamps()  # Disabled - user will manually clean up data
    migrate_jiten_cron_job()
    migrate_populate_games_cron_job()  # Run BEFORE daily_rollup to ensure games exist
    migrate_daily_rollup_cron_job()
    migrate_genres_and_tags()  # Add genres and tags columns
    migrate_user_plugins_cron_job()
    migrate_jiten_upgrader_cron_job()  # Weekly check for new Jiten entries
    migrate_tokenization_backfill_cron()  # Create backfill cron if needed
    migrate_daily_tokenization_cron()  # Create daily catchup cron
        
check_and_run_migrations()
    
# all_lines = GameLinesTable.all()


# # Convert String timestamp to float timestamp
# for line in all_lines:
#     if isinstance(line.timestamp, str):
#         try:
#             line.timestamp = float(line.timestamp)
#         except ValueError:
#             # Handle invalid timestamp format
#             line.timestamp = 0.0
#     line.save()

# import random
# import uuid
# from datetime import datetime
# from GameSentenceMiner.util.text_log import GameLine
# from GameSentenceMiner.util.db import GameLinesTable

# # List of common Japanese characters (kanji, hiragana, katakana)
# japanese_chars = (
#     "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
#     "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
#     "亜唖娃阿哀愛挨悪握圧扱宛嵐安暗案闇以衣位囲医依委威為胃尉異移維緯"
#     # ... (add more kanji for more variety)
# )

# def random_japanese_text(length):
#     return ''.join(random.choices(japanese_chars, k=length))

# batch_size = 1000
# lines_batch = []

# def random_datetime(start_year=2024, end_year=2026):
#     start = datetime(start_year, 1, 1, 0, 0, 0)
#     end = datetime(end_year, 12, 31, 23, 59, 59)
#     delta = end - start
#     random_seconds = random.randint(0, int(delta.total_seconds()))
#     return start + timedelta(seconds=random_seconds)

# from datetime import timedelta

# for i in range(500000):  # Adjust for desired number of lines
#     line_text = random_japanese_text(random.randint(25, 40))
#     lines_batch.append(GameLine(
#         id=str(uuid.uuid1()),
#         text=line_text,
#         time=random_datetime(),
#         prev=None,
#         next=None,
#         index=i,
#         scene="RandomScene"
#     ))
    
#     if len(lines_batch) >= batch_size:
#         GameLinesTable.add_lines(lines_batch)
#         GameLinesTable2.add_lines(lines_batch)
#         lines_batch = []
#     if i % 1000 == 0:
#         print(f"Inserted {i} lines...")

# # Insert any remaining lines
# if lines_batch:
#     GameLinesTable.add_lines(lines_batch)
#     GameLinesTable2.add_lines(lines_batch)
# for _ in range(10):  # Run multiple times to see consistent timing
#     start_time = time.time()
#     GameLinesTable.all()
#     end_time = time.time()

#     print(f"Time taken to query all lines from GameLinesTable: {end_time - start_time:.2f} seconds")

#     start_time = time.time()
#     GameLinesTable2.all()
#     end_time = time.time()

#     print(f"Time taken to query all lines from GameLinesTable2: {end_time - start_time:.2f} seconds")

# print("Done populating GameLinesTable and GameLinesTable2 with random Japanese text.")
