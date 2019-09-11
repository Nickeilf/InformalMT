import argparse
import re
from emoji import UNICODE_EMOJI

def process_noisy_file(input_path, output_path, tag):
	out_file = open(output_path, 'w')
	with open(input_path, 'r') as file:
		for line in file:
			temp = []
			tokens = line.split()
				
			for tok in tokens:
				if is_emoji(tok):
					temp.append('<placeholder>')
				elif tok.isupper():
					temp.append('<U>')
					temp.append(tok.lower())
				elif tok[:1].isupper() and tok[1:].islower():
					temp.append('<T>')
					temp.append(tok.lower())
				else:
					temp.append(tok)
			
			inline_casing = " ".join(temp)

			out_file.write(tag+" "+inline_casing+"\n")

	out_file.close()

def is_emoji(str):
	for char in str:
		if char not in UNICODE_EMOJI:
			return False
	return True



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-output", help="path of the output file")
	parser.add_argument("-input", help="path of the input file")
	parser.add_argument("-tag", help="tags to be added in front of the sentences")
	args = parser.parse_args()

	print("processing:", args.input)
	process_noisy_file(args.input, args.output, args.tag)


