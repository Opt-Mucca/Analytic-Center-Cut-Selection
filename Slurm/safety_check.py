#! /usr/bin/env python
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str)
    args = parser.parse_args()

    assert not os.path.isfile(args.outfile)
    assert os.path.isdir(os.path.dirname(args.outfile))

    # The single job of this function is to produce a file with the name outfile. This job will always be made
    # dependent on other slurm jobs. It signals jobs are complete without communicating directly with SLURM controller.
    with open(args.outfile, 'w') as s:
        s.write('True')