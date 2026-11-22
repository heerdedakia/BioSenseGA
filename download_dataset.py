"""
Download script for UCI Gas Sensor Array Drift Dataset.
Downloads the ZIP file and extracts batch files to the data/ directory.
"""
import os
import sys
import zipfile
import requests

def download_dataset(target_dir='data'):
    URL = 'https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip'
    ZIP_NAME = 'gas_sensor_dataset.zip'
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if batch files already exist
    existing = [f for f in os.listdir(target_dir) if f.startswith('batch') and f.endswith('.dat')]
    if len(existing) >= 10:
        print(f'Dataset already exists in {target_dir}/ ({len(existing)} batch files found)')
        return True
    
    print(f'Downloading dataset from UCI repository...')
    print(f'URL: {URL}')
    
    try:
        response = requests.get(URL, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(ZIP_NAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f'\rProgress: {pct:.1f}%', end='', flush=True)
        print('\nDownload complete!')
        
        # Extract
        print('Extracting files...')
        with zipfile.ZipFile(ZIP_NAME, 'r') as zf:
            # The ZIP contains a Dataset/ folder with batch files
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename.startswith('batch') and basename.endswith('.dat'):
                    # Extract to target_dir
                    with zf.open(member) as source:
                        target_path = os.path.join(target_dir, basename)
                        with open(target_path, 'wb') as target:
                            target.write(source.read())
                    print(f'  Extracted: {basename}')
        
        # Clean up ZIP
        os.remove(ZIP_NAME)
        
        # Verify
        extracted = [f for f in os.listdir(target_dir) if f.startswith('batch') and f.endswith('.dat')]
        print(f'\nSuccessfully extracted {len(extracted)} batch files to {target_dir}/')
        return len(extracted) > 0
        
    except requests.exceptions.RequestException as e:
        print(f'\nError downloading dataset: {e}')
        print('\nPlease download manually from:')
        print('https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset')
        print(f'Extract batch1.dat through batch10.dat into the {target_dir}/ directory.')
        return False
    except zipfile.BadZipFile:
        print('\nError: Downloaded file is not a valid ZIP.')
        return False

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'data'
    success = download_dataset(target)
    if not success:
        sys.exit(1)
