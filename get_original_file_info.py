#!/usr/bin/env python3

"""
CBDA validation and training set creation.

This script gets some information about the original data file to be used for
creating validation sets and training sets for a CBDA project.

The purpose is to obtain this information once, since it involves scanning of
the entire original data file. This will avoid unnecessary scans to obtain
this information by subsequent scripts, where those scripts may have to be
run multiple times.

Inputs:
    File name of original data file.
        The following assumptions are made about the contents of the file:
            It is a text file.

            It has a header line with column names.

Outputs:

   A Python Pickle file with a Python set object containing:
       The number of lines in the original data file.

       The number of comma separated columns in the header line of the
       original data file. It is assumed that all other lines in the file
       also have that number of columns.
"""


import sys
import argparse
import os
import subprocess
import pickle
import traceback

def format_cmd_result(cmd_result):

    """
    Format the result from a call to subprocess functions, such as run,
    typically for use in error messages.

    cmd_result is an instance of class subprocess.CompletedProcess.
    """

    msg = '\nReturn code:{0}\nout:\n{1}\nerr:\n{2}'
    msg = msg.format(cmd_result.returncode, cmd_result.stdout, cmd_result.stderr)
    return msg


def define_args(args=None):

    """
    Define, get and check arguments.
    """

    parser = argparse.ArgumentParser()

    msg = 'The file name of the original data set'
    parser.add_argument('-i', '--original-file', dest='original_file_name', \
                        help=msg, type=str, default=None, required=True)

    msg = 'The name for the output file.'
    parser.add_argument('-o', '--output-file', dest='output_file_name',
                        help=msg, type=str, default=None, required=True)

    args = parser.parse_args()

    # Check the arguments for validity.

    args_ok = True

    if not os.path.isfile(args.original_file_name):
        msg = '\nOriginal data set file "{0}" does not exist.\n'
        msg = msg.format(args.original_file_name)
        print(msg)
        args_ok = False

    if not args_ok:
        sys.exit(1)

    return args

def print_args(args):

    """
    For testing and debugging.
    """

    print(f'args.original_file_name: {args.original_file_name}')
    print(f'args.output_file_name: {args.output_file_name}')

    print('')

def get_original_file_line_count(original_file_name):

    """
    Get the line count of the original file using a system command.
    """

    cmd = f'wc -l {original_file_name}' + "| awk '{print $1}'"

    try:
        cmd_result = subprocess.run(cmd, check=False, shell=True, text=True,
                                    capture_output=True)

        if cmd_result.returncode != 0:
            msg = 'Failure getting original file line count'
            msg += '\nCommand: {0}\nResult:{1}'
            formatted_result = format_cmd_result(cmd_result)
            msg = msg.format(cmd, formatted_result)
            print(msg)
            sys.exit(0)

    except (OSError, ValueError):
        msg = 'Failure getting original file line count'
        msg += '\nCommand: {0}\nException:\n{1}'
        msg = msg.format(cmd, traceback.format_exc())
        print(cmd)
        sys.exit(0)

    try:
        original_line_count = int(cmd_result.stdout)
    except ValueError:
        msg = 'Failure getting original file line count'
        msg += 'Count is not an integer'
        msg += '\nCommand: {0}\nResult:{1}'
        formatted_result = format_cmd_result(cmd_result)
        msg = msg.format(cmd, formatted_result)
        print(msg)
        sys.exit(0)

    return original_line_count

def get_original_file_column_count(original_file_name):

    """
    Get the column count of the original file using a system command.
    """
    cmd = "head -n1 {0} | sed 's#,#\\n#g' | wc -l | awk '{{print $1}}'"
    cmd = cmd.format(original_file_name)

    try:
        cmd_result = subprocess.run(cmd, check=False, shell=True, text=True,
                                    capture_output=True)

        if cmd_result.returncode != 0:
            msg = 'Failure getting original file column count'
            msg += '\nCommand: {0}\nResult:{1}'
            formatted_result = format_cmd_result(cmd_result)
            print(msg)
            sys.exit(0)

    except (OSError, ValueError):
        msg = 'Failure getting original file column count'
        msg += '\nCommand: {0}\nException:\n{1}'
        msg = msg.format(cmd, traceback.format_exc())
        print(cmd)
        sys.exit(0)

    try:
        original_column_count = int(cmd_result.stdout)
    except ValueError:
        msg = 'Failure getting original file column count'
        msg += 'Count is not an integer'
        msg += '\nCommand: {0}\nResult:{1}'
        formatted_result = format_cmd_result(cmd_result)
        msg = msg.format(cmd, formatted_result)
        print(msg)
        sys.exit(0)

    return original_column_count

def program_start():
    """
    The main function for the program.

    Putting this code in a function rather than in global scope simplifies
    using pylint. It avoids many pylint complaints such as redfining a variable
    from an outer scope or insisting a variable name refers to a constant
    (pyling considers a constant any variable defined at module level that is
     not bound to a class object).
    """

    args = define_args()
    #print_args(args)

    # Ensure the output file has a specific suffix.
    suffix = '.pickle'
    if not args.output_file_name.endswith(suffix):
        args.output_file_name += suffix

    original_line_count = get_original_file_line_count(args.original_file_name)
    original_column_count = get_original_file_column_count(args.original_file_name)

    # Create a single data structure for writing to the Pickle file.
    save_info = (original_line_count, original_column_count)

    # Write the Pickle file.
    with open(args.output_file_name, 'wb') as output_file:
        pickle.dump(save_info, output_file)

if __name__ == '__main__':
    program_start()
