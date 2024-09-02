#!/usr/bin/env python3

"""
CBDA validation and training set creation.

This script lists the contents of the Pickle file containing the original
data file information.
"""

import sys
import argparse
import os
import pickle

def define_args(cmd_line_args=None):

    """
    Get command line arguments.
    """

    parser = argparse.ArgumentParser()

    msg = 'The file name of the Pickle file with the original data file'
    msg += ' information.'
    parser.add_argument('--odfi', '--original-data-file-info',
                        dest='originalDataFileInfo', help=msg, type=str, \
                        default=None, required=True)


    cmd_line_args = parser.parse_args()

    # Check the arguments for validity.

    args_ok = True

    if not os.path.isfile(cmd_line_args.originalDataFileInfo):
        msg = '\nValidation ordinal file "{0}" does not exist.\n'
        msg = msg.format(cmd_line_args.originalDataFileInfo)
        print(msg)
        args_ok = False

    if not args_ok:
        sys.exit(1)

    return cmd_line_args

def print_args(cmd_line_args):

    """
    For testing and debugging.
    """

    msg = 'cmd_line_args.originalDataFileInfo: {}'
    print(msg.format(cmd_line_args.originalDataFileInfo))

    # Blank line after all argument lines.
    print('')

args = define_args()
#print_args(args)

with open(args.originalDataFileInfo, 'rb') as odfiFile:
    (originalLineCount, originalColumnCount) = pickle.load(odfiFile)

print(f'originalLineCount: {originalLineCount}')
print(f'originalColumnCount: {originalColumnCount}')
