#!/usr/bin/python3

import argparse
import io
import re
import readline
import os
import enchant


class TextTransform:
  def __init__(self, input_file):
    self.speaker2_dict = self.parse_text_file(input_file)


  def parse_text_file(self, input_file):
       with open(input_file, "r") as f:

        text = f.read()
        text = text.translate({ord(i): None for i in '.,&!?;:()'})
        text = text.lower()

        return text.split()


  def lookup_text(self, text):

    out = []
    words = text.split()
    total = 0
    for word in words:
      indices = [i+1 for i, x in enumerate(self.speaker2_dict) if x == word]
         
      print("\"" + word + "\" is at index " + str(indices))

    return 


  def transform(self, text):
     return self.lookup_text(text)


def main():
  parser = argparse.ArgumentParser(
    description=("Lookup words in the given file"))
  parser.add_argument("--input_file", default="declaration.txt",
                      help="File source for looking words up in")

  options = parser.parse_args()

  if options.input_file:
      transformer = TextTransform(options.input_file)

      user_input = ""
      while user_input != "_quit_":
        user_input = input("Enter input (\'_quit_\' to quit): ").strip()
        if not user_input: break

        transformer.transform(user_input)

      print("Goodbye!")



if __name__ == "__main__":
  main()
