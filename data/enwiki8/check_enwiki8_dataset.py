import os
import numpy as np
import pickle

def load_binary_file(file_path):
    return np.fromfile(file_path, dtype=np.uint8)

def main():
    # Load meta information
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    itos = meta['itos']
    
    # Load and check each dataset
    for split in ['train', 'val', 'test']:
        file_path = f'{split}.bin'
        if os.path.exists(file_path):
            data = load_binary_file(file_path)
            print(f"\n{split.capitalize()} set:")
            print(f"Total tokens: {len(data):,}")
            
            # Decode and show first 100 bytes
            first_100 = bytes([itos[i] for i in data[:100]]).decode('utf-8', errors='replace')
            print("First 100 bytes decoded:")
            print(repr(first_100))
            
            # Decode and show last 100 bytes
            last_100 = bytes([itos[i] for i in data[-100:]]).decode('utf-8', errors='replace')
            print("Last 100 bytes decoded:")
            print(repr(last_100))
        else:
            print(f"\n{file_path} not found.")

if __name__ == "__main__":
    main()


# Train set:
# Total tokens: 90,000,000
# First 100 bytes decoded:
# '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.3/" xmlns:xsi="http://www.w3.org/2001/XMLSch'
# Last 100 bytes decoded:
# 't preserved a unique language, culture and crop system.  The crop system is adapted to the dry north'

# Val set:
# Total tokens: 5,000,000
# First 100 bytes decoded:
# "ern highlands and does not partake of any other area's crops.  The most famous member of this crop s"
# Last 100 bytes decoded:
# '= Pictures ===\n*[http://www.culture.gr/2/21/218/218ab/e218ab00.html Icons of Mount Athos]\n*[http://w'

# Test set:
# Total tokens: 5,000,000
# First 100 bytes decoded:
# 'ww.auburn.edu/academic/liberal_arts/foreign/russian/icons/ Russian Icons from 12th to 18th century]\n'
# Last 100 bytes decoded:
# 'guage, but modern scholars consider them to be separate languages.\n\nRecently, Standard Japanese has '