# evaluation_pipeline/preprocessing.py

"""
One-time data preprocessing utilities.

These functions are used to prepare raw data before evaluation.
Run once, save the results, then use the processed data.
"""

import time
import base64
import requests
from pathlib import Path
from typing import Optional, List, Union
from io import BytesIO

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================================
# IMAGE PREPROCESSING (ONE-TIME SETUP)
# ============================================================================

def convert_to_data_url(url: str, max_retries: int = 3) -> Optional[str]:
    """
    Convert various image URLs to base64 data URLs.
    Handles Google Drive URLs, regular image URLs (including WebP), and existing data URLs.
    
    Args:
        url: Image URL to convert
        max_retries: Maximum number of retry attempts
        
    Returns:
        Base64 data URL string, or None if conversion fails
    """
    # If already a base64 encoded data image, return as is
    if url.startswith('data:'):
        return url
    
    # Convert Google Drive URLs to direct download format
    if 'drive.google.com' in url:
        # Extract file ID from various Google Drive URL formats
        file_id = None
        if '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        
        if file_id:
            # Use direct download URL
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Download and convert image
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check if we got an HTML page (Google's download warning)
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                # Try alternative download method for Google Drive
                if 'drive.google.com' in url:
                    # Extract file ID and try different approach
                    if 'id=' in url:
                        file_id = url.split('id=')[1].split('&')[0]
                        # Try the uc?id= format without export=download
                        alt_url = f"https://drive.google.com/uc?id={file_id}"
                        response = requests.get(alt_url, headers=headers, timeout=30, stream=True)
                        response.raise_for_status()
            
            # Read image data
            image_data = BytesIO(response.content)
            
            # Verify it's a valid image and get format
            try:
                with Image.open(image_data) as img:
                    # Convert to RGB if necessary (handles RGBA, LA, P, etc.)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    
                    # Save as JPEG to BytesIO
                    output = BytesIO()
                    img.save(output, format='JPEG', quality=85, optimize=True)
                    output.seek(0)
                    
                    # Encode to base64
                    encoded_data = base64.b64encode(output.read()).decode('utf-8')
                    
                    # Add 1 second delay after successful download
                    time.sleep(1)
                    
                    return f"data:image/jpeg;base64,{encoded_data}"
                    
            except Exception as img_error:
                print(f"Image processing error for {url}: {img_error}")
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"Download attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
        except Exception as e:
            print(f"Unexpected error processing {url}: {e}")
            break
    
    print(f"Failed to convert {url} to data URL after {max_retries} attempts")
    return None


def process_image_data_item(item) -> Optional[Union[str, List]]:
    """
    Process a single image data item (can be string, list, or None).
    
    Args:
        item: Image data - can be a URL string, list of URLs, or None
        
    Returns:
        Converted data URL(s) or None
    """
    if item is None:
        return None
    
    # Check if it's a scalar value that's NaN
    try:
        if pd.isna(item):
            return None
    except (ValueError, TypeError):
        # If pd.isna() fails, item is likely a list/array, so continue processing
        pass
    
    if isinstance(item, str):
        return convert_to_data_url(item)
    elif isinstance(item, list):
        return [convert_to_data_url(url) if url else None for url in item]
    else:
        return None


def preprocess_all_images(df: pd.DataFrame, image_column: str = 'image_data') -> List:
    """
    Preprocess all images in a dataframe column.
    
    This is a one-time operation - save the results and load them later.
    
    Args:
        df: DataFrame containing image data
        image_column: Name of column with image URLs/data
        
    Returns:
        List of processed image data (base64 data URLs)
    """
    print(f"Preprocessing images from '{image_column}' column...")
    print(f"Total rows: {len(df)}")
    
    # Count rows with images (using try/except for safety)
    rows_with_images = 0
    for val in df[image_column]:
        try:
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                rows_with_images += 1
        except:
            rows_with_images += 1
    
    print(f"Rows with image data: {rows_with_images}")
    
    processed_images = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_data = row.get(image_column)
        processed = process_image_data_item(image_data)
        processed_images.append(processed)
    
    return processed_images