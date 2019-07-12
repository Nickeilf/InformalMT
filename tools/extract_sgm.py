from bs4 import BeautifulSoup,SoupStrainer
import argparse

def extract_sentences(inputfile, outputfile):
    f = open(inputfile, 'r')
    data= f.read()
    soup = BeautifulSoup(data, "lxml")
    contents = soup.findAll('seg')

    with open(outputfile, 'w') as f:
        for line in contents:
            f.write(line.text.strip()+"\n")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-src", help="path of the input xml file")
	parser.add_argument("-tgt", help="path of the output txt file")
	args = parser.parse_args()

	extract_sentences(args.src, args.tgt)
