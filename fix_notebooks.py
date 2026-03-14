import json
import glob
from pathlib import Path

def fix_notebooks():
    paths = glob.glob('notebooks/kaggle/macro/mask-r-cnn/*.ipynb')
    
    for path in paths:
        path = Path(path)
        print(f"Processing {path.name}...")
        
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        modified = False
        
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
                
            new_source = []
            for line in cell['source']:
                # The script redundantly injected commas instead of path divisions
                if "with open(LOCAL_" in line and "annotations.json" in line:
                    prefix = line.split(" / ")[0] # gets "with open(LOCAL_FIBER_DIR"
                    new_source.append(f"{prefix} / 'train' / 'annotations.json') as f:\n")
                    modified = True
                    continue
                
                # Fix the random repeating train/train/val strings
                if "samples = sorted(" in line and "images').glob" in line:
                    prefix = line.split(" / ")[0] # gets "samples = sorted((LOCAL_FIBER_DIR"
                    ext = line.rsplit(".", 1)[1] # gets "png'))\n" or "jpg'))\n"
                    new_source.append(f"{prefix} / 'val' / 'images').glob('*.{ext}")
                    modified = True
                    continue
                
                # Also fix the copytree source logic since kaggle downloads the top parent
                if "candidates = sorted(" in line:
                    # Let's cleanly inject a fix below this if it needs to go up a dir
                    pass
                    
                new_source.append(line)
            
            if len(new_source) != len(cell['source']) or any(a != b for a, b in zip(new_source, cell['source'])):
                cell['source'] = new_source
                modified = True

        if modified:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f"  ✓ Fixed {path.name}")

if __name__ == '__main__':
    fix_notebooks()