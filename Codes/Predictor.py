#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Predictor
import tensorflow as tf           
import numpy as np                
from Bio import SeqIO             # For reading FASTA files
import argparse                   # To handle command-line inputs
import os                         # To check file existence
import sys
from tensorflow.keras.models import load_model


#CONFIGURATION
MAX_SEQUENCE_LENGTH = 2000 
MODEL_FILE_PATH = "C:/Users/sspat/OneDrive/Desktop/ARG Predictor.h5" # Path to saved model
DNA_ONE_HOT_MAP = {              #Bases converted to numeric arrays
    'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 
    'G': [0, 0, 0, 1], 'N': [0, 0, 0, 0] #N for unknown bases
}

#FUCNTIONS
def encode_dna(sequence, target_length, encoding_map):
    encoded_array = np.zeros((target_length, 4), dtype=np.int8)
    # Iterates up to target_length, encoding each base
    for i, base in enumerate(sequence[:target_length]):
        encoded_array[i] = encoding_map.get(base.upper(), encoding_map['N'])
    return encoded_array

#PREDICTION FUNCTION
def main():
    #SETS UP COMMAND LINE 
    parser = argparse.ArgumentParser(description="Classify DNA sequences as ARGs or Non-ARGs.")
    parser.add_argument("fasta_input_file", help="Path to the FASTA file with sequences.")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Probability threshold (default: 0.5 for ARG classification).") #Probability>0.5, ARG
    args = parser.parse_args()

    #Loading the trained model
    try:
        model = load_model(MODEL_FILE_PATH)
    except Exception: 
        sys.exit(1) # Exits if model cannot be loaded

    #ENCODING THE NEW SEQUENCES
    input_sequences = []
    sequence_ids = []

    try:
        for record in SeqIO.parse(args.fasta_input_file, "fasta"):
            input_sequences.append(str(record.seq))
            sequence_ids.append(record.name)
    except Exception: # Catch any FASTA reading error 
        sys.exit(1)

    if not input_sequences: #If no sequences found, exit
        sys.exit(0)

    # Converts all input sequences to the model's required numerical format
    encoded_input_data = np.array([
        encode_dna(s, MAX_SEQUENCE_LENGTH, DNA_ONE_HOT_MAP) 
        for s in input_sequences
    ])

    #PREDICTIONS
    probabilities = model.predict(encoded_input_data, verbose=0)

    #OUTPUT
    print(f"\nPREDICTION RESULTS (Threshold: {args.threshold:.2f}) ---")
    print(f"{'Sequence ID':<30} | {'Probability':<15} | {'Class'}")
    print("-" * 65)

    for i, seq_id in enumerate(sequence_ids):
        prob = probabilities[i][0]
        prediction_class = "ARG" if prob > args.threshold else "Non-ARG"
        print(f"{seq_id:<30} | {prob:<15.4f} | {prediction_class}")
    print("-" * 65)

#RUN THE MAIN FUNCTION WHEN SCRIPT IS EXECUTED
if __name__ == "__main__":
    main()


# In[ ]:




