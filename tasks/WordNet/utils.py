# Code from https://github.com/pfnet-research/hyperbolic_wrapped_distribution/blob/master/lib/dataset/wordnet.py


import torch
import pathlib
from itertools import count
from collections import defaultdict

import nltk
import numpy as np
from nltk.corpus import wordnet as wn


def generate_dataset(output_dir, with_mammal=False):
    output_path = pathlib.Path(output_dir) / 'noun_closure.tsv'

    # make sure each edge is included only once
    edges = set()
    for synset in wn.all_synsets(pos='n'):
        # write the transitive closure of all hypernyms of a synset to file
        for hyper in synset.closure(lambda s: s.hypernyms()):
            edges.add((synset.name(), hyper.name()))

        # also write transitive closure for all instances of a synset
        for instance in synset.instance_hyponyms():
            for hyper in instance.closure(lambda s: s.instance_hypernyms()):
                edges.add((instance.name(), hyper.name()))
                for h in hyper.closure(lambda s: s.hypernyms()):
                    edges.add((instance.name(), h.name()))

    with output_path.open('w') as fout:
        for i, j in edges:
            fout.write('{}\t{}\n'.format(i, j))

    if with_mammal:
        import subprocess
        mammaltxt_path = pathlib.Path(output_dir).resolve() / 'mammals.txt'
        mammaltxt = mammaltxt_path.open('w')
        mammal = (pathlib.Path(output_dir) / 'mammal_closure.tsv').open('w')
        commands_first = [
            ['cat', '{}'.format(output_path)],
            ['grep', '-e', r'\smammal.n.01'],
            ['cut', '-f1'],
            ['sed', r's/\(.*\)/\^\1/g']
        ]
        commands_second = [
            ['cat', '{}'.format(output_path)],
            ['grep', '-f', '{}'.format(mammaltxt_path)],
            ['grep', '-v', '-f', '{}'.format(
                'mammals_filter.txt'
            )]
        ]
        for writer, commands in zip([mammaltxt, mammal], [commands_first, commands_second]):
            for i, c in enumerate(commands):
                if i == 0:
                    p = subprocess.Popen(c, stdout=subprocess.PIPE)
                elif i == len(commands) - 1:
                    p = subprocess.Popen(c, stdin=p.stdout, stdout=writer)
                else:
                    p = subprocess.Popen(c, stdin=p.stdout, stdout=subprocess.PIPE)
                # prev_p = p
            p.communicate()
        mammaltxt.close()
        mammal.close()


def parse_seperator(line, length, sep='\t'):
    d = line.strip().split(sep)
    if len(d) == length:
        w = 1
    elif len(d) == length + 1:
        w = int(d[-1])
        d = d[:-1]
    else:
        raise RuntimeError('Malformed input ({})'.format(line.strip()))
    return tuple(d) + (w,)


def parse_tsv(line, length=2):
    return parse_seperator(line, length, '\t')


def iter_line(file_name, parse_function, length=2, comment='#'):
    with open(file_name, 'r') as fin:
        for line in fin:
            if line[0] == comment:
                continue
            tpl = parse_function(line, length=length)
            if tpl is not None:
                yield tpl


def intmap_to_list(d):
    arr = [None for _ in range(len(d))]
    for v, i in d.items():
        arr[i] = v
    assert not any(x is None for x in arr)
    return arr


def slurp(file_name, parse_function=parse_tsv, symmetrize=False):
    ecount = count()
    enames = defaultdict(ecount.__next__)

    subs = []
    for i, j, w in iter_line(file_name, parse_function, length=2):
        if i == j:
            continue
        subs.append((enames[i], enames[j], w))
        if symmetrize:
            subs.append((enames[j], enames[i], w))
    idx = np.array(subs, dtype=np.int64)

    # freeze defaultdicts after training data and convert to arrays
    objects = intmap_to_list(dict(enames))
    print('slurp: file_name={}, objects={}, edges={}'.format(
        file_name, len(objects), len(idx)))
    return idx, objects


def create_adjacency(indices):
    adjacency = defaultdict(set)
    for i in range(len(indices)):
        s, o = indices[i]
        adjacency[s].add(o)
    return adjacency


def calculate_energy(model, x, test_samples, batch_size, dist_fn):
    x = torch.tensor(x).cuda()
    kl_target = torch.zeros(x.size(0)).cuda()
    nb_batch = np.ceil(x.size(0) / batch_size).astype(int)

    for i in range(nb_batch):
        idx_start = i * batch_size
        idx_end = (i + 1) * batch_size
        data = x[idx_start:idx_end]

        mean, covar = model(data)
        dist_anchor = dist_fn(mean[:, 0, :], covar[:, 0, :])
        dist_target = dist_fn(mean[:, 1, :], covar[:, 1, :])
         
        z = dist_anchor.rsample(test_samples)
        log_prob_anchor = dist_anchor.log_prob(z)
        log_prob_target = dist_target.log_prob(z)
        kl_target[idx_start:idx_end] = (log_prob_anchor - log_prob_target).mean(dim=0)

    return kl_target


if __name__ == "__main__":
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print('wordnet dataset is not found, start download')
        nltk.download('wordnet')
    print('generate dataset')
    generate_dataset('../../data/', with_mammal=False)
    generate_dataset('../../data/', with_mammal=True)

