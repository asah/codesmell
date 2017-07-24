from nltk import ngrams
import sys, gzip
fh = gzip.open("githubLabeledTrainData-augmented.tsv.gz")
lines = []
popularity = {}
hashes = []
sentiments = []
contents = []
while True:
    line = fh.readline()
    if line is None: break
    try:
        hash, sentiment, content = line.split("\t", 3)
    except:
        sys.stdout.write(line)
        break
        continue
    hashes.append(hash)
    sentiments.append(sentiment)
    contents.append(content)
    if len(contents) % 10000 == 0:
        if len(contents) % 100000 == 0:
            sys.stdout.write("{}K".format(len(contents)/1000))
        sys.stdout.write(".")
        sys.stdout.flush()
    idx = 0
    idx_end = len(content)-4+1
    while idx < idx_end:
        ngram = content[idx:idx+4]
        if ngram in popularity:
            popularity[ngram] += 1
            idx = idx + 4
        else:
            popularity[ngram] = 1
            idx = idx + 1

print("\n".join(["{}:{}".format(x[0],x[1]) for x in sorted(popularity.items(), key=lambda item: item[1], reverse=True)[0:200]]))
