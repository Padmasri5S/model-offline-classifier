import os
import csv
import hashlib
from pathlib import Path
from PIL import Image
import imagehash
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
def get_image_hash(image_path, hash_type='phash'):
    """
    Generate perceptual hash of image.
    Use 'phash' for perceptual (catches similar images),
    or 'dhash' for difference hash (more strict),
    or 'md5' for exact byte matching
    """
    try:
        if hash_type == 'md5':
            # Exact file hash
            hash_md5 = hashlib.md5()
            with open(image_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        else:
            img = Image.open(image_path)
            if hash_type == 'phash':
                return str(imagehash.phash(img))
            elif hash_type == 'dhash':
                return str(imagehash.dhash(img))
    except Exception as e:
        print(f"Error hashing {image_path}: {e}")
        return None
def find_duplicate_images(image_dir, hash_type='phash'):
    """
    Find duplicate images using hashing.
    Returns dict: {hash: [list of image paths]}
    """
    hash_dict = defaultdict(list)
    
    print(f"Scanning {image_dir} for duplicates (method: {hash_type})...")
    
    image_files = list(Path(image_dir).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']]
    
    print(f"Found {len(image_files)} image files to process...")
    
    for image_file in tqdm(image_files, desc="Hashing images"):
        image_hash = get_image_hash(str(image_file), hash_type)
        
        if image_hash:
            hash_dict[image_hash].append(str(image_file))
        else:
            print(f"Failed to hash: {image_file.name}")
    
    return hash_dict
def save_duplicates_to_csv(hash_dict, output_path):
    """
    Save only groups with 2+ images (actual duplicates)
    """
    duplicates_data = []
    group_id = 0
    
    for image_hash, images in hash_dict.items():
        if len(images) > 1:  # Only groups with duplicates
            for idx, img_path in enumerate(images):
                duplicates_data.append({
                    'group_id': group_id,
                    'is_original': idx == 0,  # Mark first as original
                    'hash': image_hash,
                    'image_path': img_path,
                    'filename': Path(img_path).name
                })
            group_id += 1
    
    if duplicates_data:
        df = pd.DataFrame(duplicates_data)
        df.to_csv(output_path, index=False)
        print(f"Found {group_id} duplicate groups with {len(duplicates_data)} total images")
        print(f"Saved to {output_path}")
        return df
    else:
        print("No duplicates found!")
        # Create empty dataframe with correct columns
        df = pd.DataFrame(columns=['group_id', 'is_original', 'hash', 'image_path', 'filename'])
        df.to_csv(output_path, index=False)
        return df
def main():
    image_dir = r"C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images"
    output_dir = Path(image_dir).parent
    
    print("=" * 70)
    print("DUPLICATE IMAGE DETECTOR")
    print("=" * 70)
    
    # Try multiple hash methods (most sensitive to least)
    for hash_method in ['md5', 'phash', 'dhash']:
        print(f"\n--- Trying hash method: {hash_method} ---")
        
        hash_dict = find_duplicate_images(image_dir, hash_type=hash_method)
        
        # Count potential duplicates
        dup_count = sum(1 for images in hash_dict.values() if len(images) > 1)
        print(f"Potential duplicate groups: {dup_count}")
        
        if dup_count > 0:
            # Save results
            output_file = output_dir / f"duplicates_detected_{hash_method}.csv"
            duplicates_df = save_duplicates_to_csv(hash_dict, output_file)
            
            # Print summary
            if len(duplicates_df) > 0:
                total_duplicates = len(duplicates_df)
                duplicate_groups = duplicates_df['group_id'].nunique() if 'group_id' in duplicates_df.columns else 0
                
                print(f"\n=== SUMMARY ({hash_method}) ===")
                print(f"Duplicate groups found: {duplicate_groups}")
                print(f"Total duplicate images: {total_duplicates}")
                print(f"\nFirst few duplicates:")
                print(duplicates_df.head(15))
            break
    else:
        print("\nNo duplicates found with any method.")
if __name__ == "__main__":
    main()

'''
PS C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model> python .\fix_duplicates.py
======================================================================
DUPLICATE IMAGE DETECTOR
======================================================================

--- Trying hash method: md5 ---
Scanning C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images for duplicates (method: md5)...
Found 3321 image files to process...
Hashing images: 100%|██████████████████████| 3321/3321 [00:07<00:00, 432.53it/s]
Potential duplicate groups: 0

--- Trying hash method: phash ---
Scanning C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images for duplicates (method: phash)...
Found 3321 image files to process...
Hashing images: 100%|███████████████████████| 3321/3321 [00:43<00:00, 75.54it/s] 
Potential duplicate groups: 0

--- Trying hash method: dhash ---
Scanning C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images for duplicates (method: dhash)...
Found 3321 image files to process...
Hashing images: 100%|███████████████████████| 3321/3321 [00:38<00:00, 86.47it/s] 
Potential duplicate groups: 2
Found 2 duplicate groups with 4 total images
Saved to C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\duplicates_detected_dhash.csv

=== SUMMARY (dhash) ===
Duplicate groups found: 2
Total duplicate images: 4

First few duplicates:
   group_id  ...                                  filename
0         0  ...  11654d0f-1174-4658-ae6d-06362e0648ae.jpg
1         0  ...  facfe9e3-381e-4d93-9874-e2b698d320c5.jpg
2         1  ...  250ab6c0-c194-4503-89a9-f999b94dc316.jpg
3         1  ...  81f8d7b1-93bb-419c-8e6a-0f4a94cf8149.jpg

[4 rows x 5 columns]
PS C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model>

Key improvements:
	• ✅ Added MD5 (exact byte matching) as first method — catches true duplicates
	• ✅ Added tqdm progress bar so you can see hashing progress (3321 images takes a minute)
	• ✅ Tries multiple hash methods (MD5 → phash → dhash)
	• ✅ Better error handling for empty dataframes
	• ✅ Added is_original flag to mark the first image in each duplicate group
	• ✅ More verbose output to debug what's happening
Run it:

If it still finds 0 duplicates with MD5, then your 3321 images are all truly unique (no byte-identical files), which is actually good! The earlier missing_images_audit.csv showed Hamming distances of 14-18, which means the missing images are visually similar but not identical.
'''
