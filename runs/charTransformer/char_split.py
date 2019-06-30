import argparse
import tqdm

def segment_char(sent):
    temp = ""
    beginning_char = True
    for char in sent:
        if char != ' ':
            if beginning_char:
                temp+=char+" "
                beginning_char=False
            else:
                if char == '\n':
                    temp += char
                else:
                    temp+="_"+char+" "
        else:
            beginning_char=True
    temp = temp.strip()
    return temp

def segment_file(input, output):
    result = []
    with open(input, "r") as file:
        lines = file.readlines()
        for line in tqdm.tqdm(lines):
            result.append(segment_char(line))

    with open(output, "w") as file:
        for sent in result:
            file.write(sent+"\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segementing sentences into character-based tokens')
    parser.add_argument('--input', required=True, help='input file path')
    parser.add_argument('--output', required=True, help='segemented file path')
    args = parser.parse_args()
    segment_file(args.input, args.output)
