#!/usr/bin/env python
#
# Usage: codesmell.py [options] <train or predict>   (use -h to get help)
#
# typical training:
#
#    cd <language>/<repo>; ../../get-all-hashes.sh <language>
#    cd ..; cat */git | gzip -9 > githubLabeledTrainData-augmented.tsv.gz
#    python codesmell.py train      (defaults to githubLabeledTrainData-augmented.tsv.gz)
#    ... days pass ...
#    <ctrl-C>
#    ls -l checkpoints/*hdf5       (look for the latest one)
#
# typical single-file predicgion (on codesmell itself!)
# typical single-file predicgion (on codesmell itself!)
#
#
#
# note: for simplicity, codesmell is intentionally a single source file, and will split
# if/as this project grows.  Likewise, code quality will improve - the irony has not been lost.
# Fodder for the Python version of codesmell.
#
# Inspirations:
# https://offbit.github.io/how-to-read/ and https://github.com/offbit/char-models
# https://arxiv.org/pdf/1508.06615.pdf
# https://arxiv.org/pdf/1509.01626.pdf
# https://datapreneurs.co/posts/deep-learning-sentiment-one-character-at-a-t-i-m-e/
# https://github.com/Lab41/sunny-side-up/wiki/Deep-Learning-Techniques-for-Sentiment-Analysis
# https://arxiv.org/pdf/1502.01710.pdf
# https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
#
# Thanks:
# AI guidance: @hannu  https://www.linkedin.com/in/christophersmith1024/  https://github.com/fommil
# github & gitlab - hosting
# hetzner.com - inexpenive GPU hosting
# google - easy hosting/setup and of course, tensorflow
# 

import re
import sys
import os
import datetime
import gzip
import bz2
import argparse
import subprocess
import random
import gc

# NOTE: see below for ternsorflow/keras imports

ARGS = None
LINE_BY_SALIENCY_HASH = {}
LINENUM_BY_SALIENCY_HASH = {}

def strpoem(s):
    return s[0:80]+"..." if len(s) > 80 else s

# courtesy of https://stackoverflow.com/a/20422915/430938
class ActionYesNo(argparse.Action):
    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        if default is None:
            raise ValueError('You must provide a default with Yes/No action')
        if len(option_strings)!=1:
            raise ValueError('Only single argument is allowed with YesNo action')
        opt = option_strings[0]
        if not opt.startswith('--'):
            raise ValueError('Yes/No arguments must be prefixed with --')
        opt = opt[2:]
        opts = ['--' + opt, '--no-' + opt]
        super(ActionYesNo, self).__init__(opts, dest, nargs=0, const=None, 
                                          default=default, required=required, help=help)

    def __call__(self, parser, namespace, values, option_strings=None):
        if option_strings.startswith('--no-'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max-commits", type=int, default=0,
                        help="for testing, choose N commits at random (0=no limit)")
    # here's how we arrived at the default value
    # gunzip -c githubLabeledTrainData-augmented.tsv.gz | \
    # perl -ne '@f1=split("\t");@f2=split("\\\\n", $f1[2]);foreach $f (@f2){print length($f),"\n";}'|sort|uniq -c|sort -k 2 -r -n|less
    parser.add_argument("--max-len-per-line", type=int, default=108,
                        help="trim each line of code at this number of chars - reduces RAM")
    # here's how we arrived at the default value
    # gunzip -c githubLabeledTrainData-augmented.tsv.gz | \
    # perl -ne '@f1=split("\t");@f2=split("\\\\n", $f1[2]);print $#f2;print "\n";'|sort|uniq -c|sort -k 2 -r -n
    parser.add_argument("--max-lines-per-commit", type=int, default=29,
                        help="trim each line of code at this number of chars - reduces RAM")
    parser.add_argument("-t", "--test", action='store_true',
                        help="same as --max-commits=5000")
    parser.add_argument("-f", "--filename", default="githubLabeledTrainData-augmented.tsv.gz",
                        help="what file (gzip) to read from ('-' for stdin)")
    parser.add_argument("-b", "--batch-size", type=int, default=250,
                        help="batch size, in number of commits  (action=train)")
    parser.add_argument("-c", "--use-checkpoint",
                        help="start from this checkpoint file (action=train or predict)")
    # TODO: breaking on challenges with the Lambda
    #parser.add_argument("-m", "--model",
    #help="start from this model file (JSON) (action=train or predict)")
    parser.add_argument("-g", "--gpu-ram-pct", type=float, default=100.0,
                        help="percent of GPU memory to use (action=train)")
    # TODO: support multiple GPUs, though in practice I didn't see a speedup...
    parser.add_argument("--charset",
                        help="what chars to recognize (can be from prev run).  use --charset=detect to auto-detect during training")
    parser.add_argument("--show-charset", action=ActionYesNo, default=False,
                        help="display the charset being used.  Other chars are mapped to backspace.")
    parser.add_argument("--raw", action=ActionYesNo, default=False,
                        help=("(action=predict) assume input is raw text instead of tab-separated commit triplets."+
                              "  sets--max-len-per-line=200 and  --max-lines-per-commit=10000"))
    parser.add_argument("--details", action=ActionYesNo, default=True,
                        help="display per-commit details")
    parser.add_argument("--array", action=ActionYesNo, default=True,
                        help="display array of thresholds")
    parser.add_argument("--confusion", action=ActionYesNo, default=False,
                        help="print a confusion matrix")
    parser.add_argument("--saliency", action=ActionYesNo, default=False,
                        help="auto-generate N examples per commit, with one line missing, e.g. for saliency maps")
    parser.add_argument("action",
                        help="train or predict")
    args = parser.parse_args()

    # batch sizing
    if args.max_commits == 0:
        args.max_commits = sys.maxint
    if args.action == 'train' and args.max_commits < 5000:
        print("for action=train, --max-commits should be 0 or >5000, so we can meaningfully split into test/validation sets")
        sys.exit(1)

    # GPU-related arguments, incl RAM
    if args.batch_size < 1 or args.batch_size > 1000:
        print("batch_size invalid: should be 1 - 1000")
    args.using_gpu = True
    # not an exact science, but let's protect against the stuff we know about
    if args.using_gpu:
        args.gpu_ram = get_gpu_ram_mb()
        # GTX 1080 has ~8GB RAM, where the half K80 (google hosting) has ~11GB RAM
        if (args.gpu_ram * args.gpu_ram_pct / 100.0 < 8200 and args.batch_size > 250) or \
           (args.gpu_ram * args.gpu_ram_pct / 100.0 < 12000 and args.batch_size > 500):
            print("--batch-size ({}) too large for GPU RAM ({} MB) given --gpu-ram-pct ({}%)".format(
                args.batch_size, args.gpu_ram, args.gpu_ram_pct))
            sys.exit(1)
        if args.gpu_ram < 8200:
            args.max_len_per_line = 102
            args.max_lines_per_commit = 26

    try:
        os.mkdir('checkpoints')
    except:
        pass
    if args.use_checkpoint:
        if not os.path.exists(args.use_checkpoint):
            print('--use-checkpoint: no such file {}'.format(args.use_checkpoint))
            sys.exit(1)
        print ("Checkpoint:", args.use_checkpoint)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    elif args.action == 'predict':
        print('action=predict requires --use-checkpoint to load the saved model')
        if args.charset == 'detect':
            print('--charset=detect not allowed during action=predict: the whole point is to match the charset used during training')
        sys.exit(1)
    if args.action != 'predict':
        # technically, this isn't true but I'd rather force myself to explicitly test quality vs during training
        if args.raw:
            print('--raw only applies for action=predict')
            sys.exit(1)
        if args.saliency:
            print('--saliency only applies for action=predict')
            sys.exit(1)
        if args.confusion:
            print('--confusion only applies for action=predict')
            sys.exit(1)
    if args.raw:
        args.max_len_per_line = 200
        args.max_lines_per_commit = 10000
        args.details = False

    # post-processing args, e.g. --test
    if args.test:
        args.max_commits = 5000  # override

    return args

def get_gpu_ram_mb():
    # e.g. 8113 MiB
    try:
        res = subprocess.check_output("nvidia-smi --query-gpu=memory.total --format=csv | tail -1", shell=True)
        ram_mb = int(re.sub(r' MiB', '', re.sub(r' GiB', '000', res)))
        return ram_mb
    except:
        return 0
    
def binarize(x, sz=97):
    import tensorflow as tf2
    return tf2.to_float(tf2.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def binarize_outshape(in_shape, sz=97):
    return in_shape[0], in_shape[1], sz

def is_saliency_hash(hash):
    return bool(re.search(r'fff[0-9][0-9]$', hash))

def load_raw_commits():
    raw_commits = []
    start_ts = current_ts = datetime.datetime.now()

    if ARGS.raw:
        # put into triplet form: <commit hash> <tab> <unused: sentiment> <tab> <text>
        fh = sys.stdin if ARGS.filename == '-' else open(ARGS.filename, 'rb')
        raw_commits.append("ffffff\t0\t" + "\\n".join([re.sub(r'[\r\n]', '', line) for line in fh]))
    else:
        if ARGS.filename is None or ARGS.filename == '-':
            fh = sys.stdin
        elif re.search(r'gz$', ARGS.filename):
            fh = gzip.open(ARGS.filename, 'rb')
        elif re.search(r'bz2$', ARGS.filename):
            fh = bz2.open(ARGS.filename, 'rb')
        else:
            print("-f: unknown input type: use - for stdin, <filename>.gz for gzip or <filename>.bz2 for bzip")
            sys.exit(1)
        print("reading from {}...".format("stdin" if fh == sys.stdin else ARGS.filename))
        for line in fh:
            # ignore header rows, which is esp helpful when concatenating TSV files from multiple repos,
            # e.g. cat c_and_cpp/*/gith*tsv | ../augment-codesmell-data.py | gzip -9 > githubLabeledTrainData-augmented.tsv.gz
	    if re.search(r'id\tsentiment', line): continue
            # ignore tiny commits
	    if len(line) < 20: continue
	    if len(raw_commits) % 1000 == 0 and datetime.datetime.now() - current_ts > datetime.timedelta(seconds=2):
                current_ts = datetime.datetime.now()
                print("{}: {}...".format( (current_ts - start_ts).seconds, len(raw_commits)))
            raw_commits.append(re.sub(r'[\r\n]', '', line))  # strip newlines
    if ARGS.max_commits < sys.maxint:
        # https://stackoverflow.com/a/6482922/430938
        raw_commits = [ raw_commits[i] for i in sorted(random.sample(xrange(len(raw_commits)), ARGS.max_commits)) ]
    return raw_commits

def load_commits():
    global LINE_BY_SALIENCY_HASH
    commits = []
    lines_of_code = []
    commit_hashes = []
    sentiments = []
    total_lines = 0
    chars = set()

    raw_commits = load_raw_commits()
    for line in raw_commits:
	commit_hash, sentiment, content = line.split('\t', 2)
	lines_of_code = re.split(r'\\n', content)
	commits.append(lines_of_code)
        total_lines += len(lines_of_code)
	sentiments.append(sentiment)
        commit_hashes.append(commit_hash)
        if ARGS.charset == 'detect':
            chars.update(set("".join(commit)))
        if ARGS.saliency:
            for lineno in range(len(lines_of_code)):
                saliency_hash = commit_hash+"fff{:02d}".format(lineno)
                LINE_BY_SALIENCY_HASH[saliency_hash] = lines_of_code[lineno]
                LINENUM_BY_SALIENCY_HASH[saliency_hash] = lineno
                commit_hashes.append(saliency_hash)
	        sentiments.append(sentiment)  # ignored for action=predict/--gen-saliency
                commits.append(lines_of_code[:lineno] + lines_of_code[lineno+1:])
	if len(commits) % 25000 == 0: print len(commits), "..."

    # free up RAM
    raw_commits = None
    gc.collect()
        
    # cap validation set to 40000 commits
    num_validation = min(40000, int(len(commits) * 0.05))
    divider = len(commits) - num_validation
    # align to batch size
    divider -= divider % ARGS.batch_size

    if ARGS.charset == 'detect' :
        print('detecting chars...')
    elif ARGS.charset is not None:
        chars = ARGS.charset
    else:
        # moving forward, we want to standardize on a charset... this is what I'm thinking for code...
        # note: \b backspace used for unknown chars - for performance it's hardcoded to be index 0
        chars = '\b \t`~!@#$%^&*()_+-=[]\\{}|;\':",./<>?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    if ARGS.show_charset:
        print('using charset: {} => {}'.format(len(chars), "".join(sorted(list(chars)))))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    example_commit_idx = min(1200, len(commits)-1)
    if not ARGS.use_checkpoint:
        print('total commits:{}, {} training, {} validation, ARGS.batch_size: {}, sample commit: {}'.format(
            len(commits), divider, len(commits)-divider, ARGS.batch_size, repr(commits[example_commit_idx])))
        print('total lines of code:', total_lines, "avg:", 1.0*total_lines/len(commits))
        
    if not ARGS.use_checkpoint:
        print('initializing arrays...')
    x = np.ones((len(commits), ARGS.max_lines_per_commit, ARGS.max_len_per_line), dtype=np.int64) * -1
    y = np.array(sentiments)

    if not ARGS.use_checkpoint:
        print('encoding each character...')
    # TODO: improve performance?
    for i, commit in enumerate(commits):
        if i > 0 and i % 25000 == 0: print ' ',i, "..."
        for j, line_of_code in enumerate(commit):
            if j < ARGS.max_lines_per_commit:
                for t, char in enumerate(line_of_code[-ARGS.max_len_per_line:]):
                    x[i, j, (ARGS.max_len_per_line - 1 - t)] = char_indices.get(char, 0)  # \b for unknown
    if ARGS.action == 'train':
        if not ARGS.use_checkpoint:
            print('sample chars in x:{}'.format(x[example_commit_idx, 2]))
            print('y:{}'.format(y[example_commit_idx]))
            print('randomly shuffling the input records and splitting training vs validation...')
        ids = np.arange(len(x))
        np.random.shuffle(ids)
        x = x[ids]
        y = y[ids]
        x_train = x[:divider]
        x_test = x[divider:]
        y_train = y[:divider]
        y_test = y[divider:]
        return commits, commit_hashes, sentiments, x_train, x_test, y_train, y_test

    if ARGS.action == 'predict':
        return commits, commit_hashes, sentiments, x, None, None, None
    
    return commits, commit_hashes, sentiments, None, None, None, None


def build_model():
    if not ARGS.use_checkpoint:
        print('building network...')
    filter_length = [5, 3, 3]   # adding extra layers didn't help
    nb_filter = [196, 196, 196]  # was: 196, 196, 256
    pool_length = 2
    commit_input = Input(shape=(ARGS.max_lines_per_commit, ARGS.max_len_per_line), dtype='int64')
    line_of_code_input = Input(shape=(ARGS.max_len_per_line,), dtype='int64')
    # char indices to one hot matrix, 1D sequence to 2D 
    embedded = Lambda(binarize, output_shape=binarize_outshape)(line_of_code_input)
    # embedded: encodes line_of_code
    for i in range(len(nb_filter)):
        embedded = Conv1D(filters=nb_filter[i],
                          kernel_size=filter_length[i],
                          padding='valid',
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          strides=1)(embedded)
        # TODO: I tried adding BatchNormalization() but it didn't help... ?
        embedded = Dropout(0.1)(embedded)
        if i >= len(nb_filter) - 3:
            # args.max_len_per_line isn't long enough for more than 3 layers of MaxPooling1D
            embedded = MaxPooling1D(pool_size=pool_length)(embedded)

    # honestly, this part is black magic to me - I understand the theory but couldn't begin to explain why it works.
    # I of course tried various networks with the BLSTM removed
    bi_lstm = Bidirectional(LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(embedded)
    line_of_code_encoder = Dropout(0.3)(bi_lstm)
    encoder = Model(inputs=line_of_code_input, outputs=line_of_code_encoder)
    if not ARGS.use_checkpoint:
        encoder.summary()
    
    encoded = TimeDistributed(encoder)(commit_input)
    #print "encoded.shape:", encoded.shape
    blstm_commit = Bidirectional(LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(encoded)
    output = Dropout(0.3)(blstm_commit)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=commit_input, outputs=output)
    return model

if __name__ == '__main__':
    ARGS = parse_args()
    
    # horrible hack to shutup tensorflow/cuda warnings
    if ARGS.using_gpu:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import pandas as pd, numpy as np, tensorflow as tf
    # horrible hack to shutup Keras telling us about its backend
    sys.stderr = open('/dev/null', 'w'); import keras; sys.stderr = sys.__stderr__
    from keras.models import Model, load_model, model_from_json
    from keras.layers import Lambda, Dense, Input, Dropout, MaxPooling1D, Conv1D
    from keras.layers import LSTM, TimeDistributed, Bidirectional
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam, RMSprop
    from keras.backend.tensorflow_backend import set_session
    import keras.callbacks

    random.seed(1234)  # reproducibility
    np.random.seed(1234)  # reproducibility

    if ARGS.gpu_ram_pct < 0.001:
        print("gpu_ram_pct=0: disabling GPU")
        # TODO: this works but still get CUDA warnings, e.g. failed call to cuInit: CUDA_ERROR_NO_DEVICE
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        ARGS.using_gpu = False
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = ""
        set_session(tf.Session(config=config))
        
    elif ARGS.gpu_ram_pct < 99.99:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = ARGS.gpu_ram_pct / 100.0
        set_session(tf.Session(config=config))

    commits, commit_hashes, sentiments, x_train, x_test, y_train, y_test = load_commits()

    if ARGS.use_checkpoint:
        # load_model(ARGS.use_checkpoint) is failing with NameError: global name 'tf' is not defined
        #if ARGS.model:
        #    model = model_from_json(open(ARGS.model).read())
        #else:
        if True:
            model = build_model()
        model.load_weights(ARGS.use_checkpoint)
    else:
        model = build_model()
        model.summary()

    if ARGS.action == 'train':
        file_name = os.path.basename(sys.argv[0]).split('.')[0]
        mtime = datetime.datetime.fromtimestamp(os.stat(sys.argv[0]).st_mtime).strftime("%Y%m%d-%H%M%s")
        with open('checkpoints/' + file_name + '--' + mtime + '.json', 'w') as outfh:
            outfh.write(model.to_json())
            outfh.close()
    
        check_cb = keras.callbacks.ModelCheckpoint('checkpoints/'+file_name+'--'+mtime+'--'+str(len(commits))+
                                                   '--{epoch:03d}-{val_acc:.4f}.hdf5')
                                                   # not currently used....
                                                   #monitor='val_acc', verbose=0, save_best_only=False, mode='max')
        # asah optimizer = RMSprop(lr=0.175, decay=.01) asah
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=ARGS.batch_size,
                  epochs=250, shuffle=True, callbacks=[check_cb])
    elif ARGS.action == 'predict':
        res = model.predict(x_train, ARGS.batch_size, verbose=1)
        res = (res * 100.0).flatten().tolist()
        threshold = 30.0
        confusion={}
        for i, commit in enumerate(commits):
            hash = commit_hashes[i]
            if ARGS.saliency and is_saliency_hash(hash):
                print("{:3.0f}% smelly, L{}: {}".format(res[i], LINENUM_BY_SALIENCY_HASH[hash], strpoem(LINE_BY_SALIENCY_HASH[hash])))
            elif ARGS.details:
                print("{:3.0f}% smelly, actual={}: {}: {}...".format(res[i], sentiments[i], hash, strpoem("\\n".join(commit))))
            if int(sentiments[i]) > 0:
                if res[i] > threshold:
                    confusion['is_smelly_forecast_smelly'] = confusion.get('is_smelly_forecast_smelly', 0) + 1
                else:
                    confusion['is_clean_forecast_smelly'] = confusion.get('is_clean_forecast_smelly', 0) + 1
            else:
                if res[i] > threshold:
                    confusion['is_smelly_forecast_clean'] = confusion.get('is_smelly_forecast_clean', 0) + 1
                else:
                    confusion['is_clean_forecast_clean'] = confusion.get('is_clean_forecast_clean', 0) + 1
        forecast_smelly = sum([x>threshold for x in res])
        forecast_clean = len(res) - forecast_smelly
        num_smelly = sum(int(sentiments[i])>0 for i in range(len(commits)))
        num_clean = len(res) - num_smelly
        if ARGS.array:
            if len(commits) == 1:
                print("forecast={} ({}%)".format("smelly" if forecast_smelly else "clean", res[0]))
            else:
                print("{} smelly. Forecasts: {}".format(100.0*num_smelly/len(res), " ".join(["{:2.0f}".format(x) for x in res])))
            
        if ARGS.confusion:
            isfs = confusion.get('is_smelly_forecast_smelly', 0)
            isfc = confusion.get('is_smelly_forecast_clean', 0)
            print("is-smelly: {}.  forecast-smelly: {} ({:.1f}%),  forecast-clean: {} ({:.1f}%)".format(
                num_smelly, isfs, 100.0*isfs/num_smelly if num_smelly>0 else 0, isfc, 100.0*isfc/num_smelly if num_smelly>0 else 0))
            icfs = confusion.get('is_clean_forecast_smelly', 0)
            icfc = confusion.get('is_clean_forecast_clean', 0)
            print("is-clean: {}.  forecast-smelly: {} ({:.1f}%),  forecast-clean: {} ({:.1f}%)".format(
                num_clean, icfs, 100.0*icfs/num_clean if num_clean>0 else 0, icfc, 100.0*icfc/num_clean if num_clean>0 else 0))
