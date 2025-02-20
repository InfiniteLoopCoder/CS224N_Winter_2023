#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference script for the NMT model.
"""

import torch
from nmt_model import NMT, Hypothesis
from utils import read_corpus
from vocab import Vocab
import sentencepiece as spm
import argparse

def load_model(model_path, use_cuda=False):
    """ Load the trained model.
    @param model_path (str): Path to trained model
    @param use_cuda (bool): Use GPU or not
    @returns model (NMT): Trained model
    """
    params = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = NMT(vocab=params['vocab'], **params['args'])
    model.load_state_dict(params['state_dict'])
    
    if use_cuda and torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))
    
    return model

def translate_sentence(model, sentence, beam_size=10):
    """ Translate a single sentence.
    @param model (NMT): Trained model
    @param sentence (str): Source language sentence
    @param beam_size (int): Beam size for beam search
    @returns translation (str): Translated sentence
    """
    # Load the source language SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load('src.model')
    
    # Tokenize the input sentence
    tokens = sp.encode_as_pieces(sentence)
    
    # Perform beam search
    hypotheses = model.beam_search(tokens, beam_size=beam_size)
    
    # Get the best hypothesis
    if hypotheses:
        best_hyp = hypotheses[0]
        # Join the tokens and replace the SentencePiece special character
        translation = ''.join(best_hyp.value).replace('▁', ' ').strip()
        return translation
    return ''

def translate_file(model, input_file, output_file, beam_size=10):
    """ Translate all sentences in a file.
    @param model (NMT): Trained model
    @param input_file (str): Path to input file
    @param output_file (str): Path to output file
    @param beam_size (int): Beam size for beam search
    """
    with open(input_file, 'r', encoding='utf8') as f_in, \
         open(output_file, 'w', encoding='utf8') as f_out:
        for line in f_in:
            translation = translate_sentence(model, line.strip(), beam_size)
            f_out.write(translation + '\n')

def interactive_translation(model, beam_size=10):
    """ Interactive translation mode.
    @param model (NMT): Trained model
    @param beam_size (int): Beam size for beam search
    """
    print("Welcome to the interactive translation mode!")
    print("Enter a sentence to translate (or 'q' to quit)")
    
    while True:
        try:
            sentence = input("\nSource sentence: ")
            if sentence.lower() == 'q':
                break
            
            translation = translate_sentence(model, sentence, beam_size)
            print(f"Translation: {translation}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Neural Machine Translation Inference')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU for inference')
    parser.add_argument('--beam-size', type=int, default=10,
                        help='Beam size for beam search')
    parser.add_argument('--input-file', type=str,
                        help='Path to input file (optional)')
    parser.add_argument('--output-file', type=str,
                        help='Path to output file (optional)')
    
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.cuda)
    model.eval()
    
    # File mode or interactive mode
    if args.input_file and args.output_file:
        print(f"Translating file {args.input_file}...")
        translate_file(model, args.input_file, args.output_file, args.beam_size)
        print(f"Translations saved to {args.output_file}")
    else:
        interactive_translation(model, args.beam_size)

if __name__ == '__main__':
    main()
