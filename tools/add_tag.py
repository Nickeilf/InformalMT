import argparse
def add_tag(input_path, output_path, tag):
	out_file = open(output_path, 'w')
	
	with open(input_path, 'r') as file:
		for line in file:
			out_file.write(tag+" "+line)

	out_file.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-output", help="path of the output file")
	parser.add_argument("-input", help="path of the input file")
	parser.add_argument("-tag", help="tags to be added in front of the sentences")
	args = parser.parse_args()
	add_tag(args.input, args.output, args.tag)


