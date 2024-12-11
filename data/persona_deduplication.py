import os
import argparse
import jsonlines

from typing import List
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

def create_shingles(input_text: str, shingle_size: int) -> List:

    # convert into lowercase string
    input_text = input_text.strip().lower()

    input_text = input_text.split()

    l = 0
    r = shingle_size

    shingles = []
    while r < len(input_text) +1:
        shingles.append(' '.join(input_text[l:r]))
        l+=1
        r+=1
    
    return shingles

def create_hash(shingle_set: List[str]) -> MinHash:
    m = MinHash(num_perm=128)

    for shingle in shingle_set:
        m.update(shingle.encode('utf8'))
    
    return m

def main(args):

    persona_files = [file for file in os.listdir(args.data_directory) if file.endswith(".jsonl")]
    lsh = MinHashLSH(threshold=args.threshold, num_perm=128)

    for file in persona_files:

        num_lines = sum(1 for line in open(os.path.join(args.data_directory, file)))

        with jsonlines.open(os.path.join(args.data_directory, file), "r") as reader,jsonlines.open(os.path.join(args.data_directory, "deduplicated_personas.jsonl"), "a") as writer:
            for row in tqdm(reader, total=num_lines):
                row['shingles'] = create_shingles(row['persona'], args.shingle_size)
                row['minhash'] = create_hash(row['shingles'])

                lsh.insert(row['id'], row['minhash'])

                result = lsh.query(row['minhash'])

                if len(result) == 1:
                    writer.write(row)

    
if __name__ == "__main__":
    args = argparse.ArgumentParser("Deduplicate Generate Personas")
    args.add_argument("--language", type=str, help="Language code")
    args.add_argument("--data_directory", type=str, help="Data directory")
    args.add_argument("--shingle_size", type=int, default=3, help="Shingle size")
    args.add_argument("--threshold", type=float, default=0.8, help="Threshold")
    args = args.parse_args()
    main(args)
