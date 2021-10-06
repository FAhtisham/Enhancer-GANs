import numpy as np

def load_dataset(max_length, max_n_examples, tokenize=False, max_vocab_size=2048, data_dir=''):
    '''Adapted from https://github.com/igul222/improved_wgan_training/blob/master/language_helpers.py'''
    print ("loading dataset...")

    lines = []

    finished = False

    for i in range(1):
        path = data_dir
        #path = data_dir+("/training-monolingual.tokenized.shuffled/news.en-{}-of-00100".format(str(i+1).zfill(5)))
        with open(path, 'r') as f:
            for line in f:
                line = line[:-1]
                line = tuple(line)
                #print(line)


                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ("P",)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'P':0}
    inv_charmap = ['P']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('P')
        filtered_lines.append(tuple(filtered_line))

    for i in range(1):
        print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap



filtered_lines, charmap, inv_charmap = load_dataset(300, 50, tokenize=False, max_vocab_size=2048, data_dir='long_enhancer.FASTA')