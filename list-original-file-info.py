#!/usr/bin/env python3

# CBDA validation and training set creation.

# This script lists the contents of the Pickle file containing the original
# data file information.

import sys
import argparse
import os
import pickle

def defineArgs(args=None):

    parser = argparse.ArgumentParser()

    msg = 'The file name of the Pickle file with the original data file'
    msg += ' information.'
    parser.add_argument('--odfi', '--original-data-file-info',
                        dest='originalDataFileInfo', help=msg, type=str, \
                        default=None, required=True)


    args = parser.parse_args()
    
    # Check the arguments for validity.
    
    argsOk = True

    if not os.path.isfile(args.originalDataFileInfo):
        msg = '\nValidation ordinal file "{0}" does not exist.\n'
        msg = msg.format(args.originalDataFileInfo)
        print(msg)
        argsOk = False

    if not argsOk:
        sys.exit(1)

    return args

def printArgs(args):
    
    """
    For testing and debugging.
    """

    print('args.originalDataFileInfo: {0}'.format(args.originalDataFileInfo))

    print

args = defineArgs()
#printArgs(args)

with open(args.originalDataFileInfo, 'rb') as odfiFile:
    (originalLineCount, originalColumnCount) = pickle.load(odfiFile)

print('originalLineCount: {0}'.format(originalLineCount))
print('originalColumnCount: {0}'.format(originalColumnCount))
