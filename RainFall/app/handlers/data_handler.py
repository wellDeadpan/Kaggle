# -----------------------------
# app/handlers/data_handler.py
from pathlib import Path
from typing import Union, Dict
from fastapi import UploadFile
from io import BytesIO
import pandas as pd
from datetime import datetime



class DataHandler:
    def __init__(self, base_dir: str = 'data/temp'):
        """Initialize DataHandler with a base directory for file operations.

        Args:
            base_dir (str): Base directory path for storing files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.temp_store: Dict[str, pd.DataFrame] = {}  # Type hint for temp storage

    async def process_uploaded_file(self, file: UploadFile) -> pd.DataFrame:
        """Process an uploaded file and convert it to a DataFrame.

        Args:
            file (UploadFile): The uploaded file object

        Returns:
            pd.DataFrame: Processed and validated DataFrame

        Raises:
            ValueError: If file format is invalid
        """
        try:
            content = await file.read()
            df = pd.read_csv(BytesIO(content))
            df = self.clean_and_validate(df)
            self.temp_store["latest_df"] = df
            return df
        except Exception as e:
            raise ValueError(f"Failed to process file: {str(e)}")
        finally:
            await file.close()  # Ensure file is closed

    def _check_file_size(self, file_size: int, max_size_mb: int = 100) -> bool:
        return file_size <= max_size_mb * 1024 * 1024

    def save_file(self, file: bytes, filename: str) -> Path:
        """Save binary file data to disk.

        Args:
            file (bytes): Binary file content
            filename (str): Name to save the file as

        Returns:
            Path: Path where file was saved

        Raises:
            IOError: If file cannot be saved
        """
        path = self.base_dir / filename
        try:
            with open(path, 'wb') as f:
                f.write(file)
            return path
        except IOError as e:
            raise IOError(f"Failed to save file {filename}: {str(e)}")

    def load_file(self, filename: str) -> pd.DataFrame:
        """Load a CSV file into a DataFrame.

        Args:
            filename (str): Name of file to load

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = self.base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File {filename} not found")
        return pd.read_csv(path)

    def _validate_file_type(self, filename: str) -> bool:
        allowed_extensions = {'.csv', '.xlsx', '.parquet'}
        return Path(filename).suffix.lower() in allowed_extensions

    def delete_file(self, filename: str) -> bool:
        """Delete a file from storage.

        Args:
            filename (str): Name of file to delete

        Returns:
            bool: True if file was deleted, False if file didn't exist
        """
        path = self.base_dir / filename
        try:
            path.unlink(missing_ok=True)
            return True
        except Exception as e:
            return False

    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Cleaned and validated DataFrame

        Raises:
            ValueError: If DataFrame fails validation
        """
        # Add your data cleaning and validation logic here
        # For example:
        # - Remove duplicates
        # - Handle missing values
        # - Validate data types
        # - Check for required columns
        return df

    def cleanup_old_files(self, max_age_days: int = 7):
        """Delete files older than max_age_days."""
        current_time = datetime.now()
        for file_path in self.base_dir.glob('*'):
            if (current_time - datetime.fromtimestamp(file_path.stat().st_mtime)).days > max_age_days:
                file_path.unlink()