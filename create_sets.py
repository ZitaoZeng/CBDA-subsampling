#!/usr/bin/env python3

"""
CBDA validation and training set creation.

This script defines validation and training sets for a CBDA project.

Inputs:
    File name of original data file.
        The following assumptions are made about the contents of the file:
            It is a text file.

            It has a header line with column names.

            It is a csv file.

            All lines have the same number of columns, including the header
            line.

    The name of a Python Pickle file containing the following information
    about the original data file:
       The number of lines in the original data file.

       The number of comma separated columns in the header line of the
       original data file. It is assumed that all other lines in the file
       also have that number of columns.

    The number of rows to extract.
        The specific rows to extract are chosen at random from the original
        file. The first row is not included, nor are any rows in the
        validation set.

    The number of columns to extract.
        The specific columns to extract are chosen at random.

    The case number column ordinal.
        To exclude from the selection of data columns for a data set, but to
        be written to each training set in addition to the selected data
        columns.  This is the column whose value corresponds to each patient.

    The output column ordinal.
        To exclude from the selection of data columns for a data set, but to
        be written to each training set in addition to the selected data
        columns.  This is the outcome column whose value is to be predicted
        by the algorithm defined by the machine learning processing of the
        training sets generated here.

    Whether to create one validation set, or to create a validation set for
    each training set. Optional.
        If not present create one validation set. If present create a
        validation set for each training set.

    The number of training sets to create.

    The starting set number. Optional.
        If not present the default value is 1.

        Needed to create unique output file names when this script is run
        more than once for an original data file, to create more training and
        validation sets than the max number of open files allowed.

    The file name of a file containing an optional set of columns to restrict
    the selection to. Optional.
        If not present all columns are available to select from, except the
        case number column and output column.

        If present only the specified columns are available to select from.
        These must not include the case number column or output column.

        This is for a second set of training/validation runs to determine
        which subset of the important columns, identified by the first
        training/validation runs, are most useful.


        This file has 2 numbers per line:

            A column ordinal.

            A value indicating the predictive power ranking of that column.
            The columns should appear in descending order by this ranking.
            The column with the highest ranking should appear first, etc.
            However, this is not checked by this script.

Outputs:
   An output file for each training set.

   Either:

       One validation set file:
           No training set will contain the any of the original data file
           lines as the validation set file.

       A validation set file for each training set file.
           A training set file will not contain any of the original data file
           lines as its corresponding validation set file.
"""

import sys
import argparse
import os
import pickle
import random

class SelectionSet:

    """

    A base class for data sets to be selected from an original data file.

    Each subclass should have the following members:

        row_ordinals: A Python set object that has the row ordinals of the rows
                      (lines) of the original file to be written for this set.

        output_columns: A list of the columns to write from a line of the
                        original file.

        set_file: An open file object to write the selected data to.

    """

    def __init__(self):
        self.row_ordinals = set()
        self.output_columns = None
        self.set_file = None

    def get_random_ordinals(self, count, end):

        """
        Generate the set of validation line ordinals.
        Get a set of count random integers between 2 and end, inclusive.
        2 to avoid the header line.

        count and end should be integers.

        count should be < (end - 1) (-1 to exclude the header line)
        """

        # Make it a set so a training set can efficiently avoid using any
        # validation set line ordinal.
        ordinal_list = random.sample(range(2, end+1), count)
        ordinal_set = set(ordinal_list)
        return ordinal_set

    def get_random_ordinals_exclude(self, count, start, end, exclude):

        """
        Get a set of count random integers between start and end, inclusive,
        but not including any integers in the exclude.

        count, start and end should be integers.

        start should be < end

        count should be < (end - start + 1)

        exclude should be a set of integers. It may be empty.

        """
        ordinals = set()
        while len(ordinals) < count:
            r = random.randint(start, end)
            if not r in exclude:
                ordinals.add(r)

        return ordinals

    def write_ordinals(self, ordinals, file_name):

        """
        Write a set of ordinals to a file, in ascending numerical order.

        These are typically needed by subsequent machine learning steps, not part
        of the data set selection process.
        """

        sorted_ordinals = sorted(ordinals)
        with open(file_name, 'w', encoding='utf_8') as ordinal_file:
            for o in sorted_ordinals:
                ordinal_file.write(str(o) + '\n')

    def check_line(self, ordinal, fields):

        """
        Check a line from the original file, to see if fields from it should be
        written for this training set.

        output_columns is sorted in ascending order by column ordinal, so the
        columns will be written in the same order they are in the original
        file. It also includes the case number column and output column, in
        addition to the selected data columns.
        """

        if ordinal in self.row_ordinals:
            # This line is for this selection set.

            # Get the fields for this selection set.
            if self.output_columns is None:
                # Get all the fields.
                fields_to_write = fields
            else:
                fields_to_write = []
                for o in self.output_columns:

                    # Because we count column ordinals from 1, but list indices
                    # start at 0.
                    o1 = o - 1

                    fields_to_write.append(fields[o1])

            # Write the fields to the training set file.
            field_str = ','.join(fields_to_write) + '\n'
            self.set_file.write(field_str)

# pylint: disable=too-many-instance-attributes
class ValidationSet(SelectionSet):

    """
    A class to represent the information for a validation set to be created.

    Each validation set has an output file name, output file object and a set
    of row ordinals for the lines of the original file for this validation set
    and a set of column ordinals for the columns of the original file for this
    validation set.

    save is true if using a single validation set, in which case we save this
    validation set in a Pickle file.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(self, ordinal, original_line_count, original_column_count, \
                 args, column_set, save=False):

        SelectionSet.__init__(self)

        # Used for output file names and error mesages.
        # If only one validation set is being created this is -1. If there is a
        # validation set for each training set then this is the same ordinal as
        # the associated training set.
        self.ordinal = ordinal

        # 2, to skip the ordinal for the original file header line.
        self.row_ordinals = self.get_random_ordinals(args.validation_row_count,
                                                     original_line_count)

        # Determine the columns to use for this validation set. If a column set
        # was provided, use it. Otherwise if not saving use a random set of
        # columns and if saving use all columns (column_ordinals and
        # output_columns remain None).
        self.column_ordinals = None
        self.output_columns = None
        if column_set is not None:
            self.column_ordinals = column_set
            self.define_output_columns(args)
        elif not save:
            exclude_cols = set([args.case_column, args.outcome_column])
            self.column_ordinals = self.get_random_ordinals_exclude(
                                          args.column_count, 1, \
                                          original_column_count, exclude_cols)
            self.define_output_columns(args)

        file_name_ordinal = ''
        if self.ordinal != -1:
            file_name_ordinal = f'-{ordinal}'

        self.file_name = f'validation-set{file_name_ordinal}'

        f = f'validation-set{file_name_ordinal}-row-ordinals'
        self.row_ordinal_file_name = f
        self.write_ordinals(self.row_ordinals, self.row_ordinal_file_name)

        f = f'validation-set{file_name_ordinal}-column-ordinals'
        self.column_ordinal_file_name = f
        if self.column_ordinals is not None:
            self.write_ordinals(self.column_ordinals, \
                                self.column_ordinal_file_name)

        # This must come before opening the validation set file, since an open
        # file object (or perhaps any file object) can't be pickled.
        if save:
            save_file_name = self.file_name + '.pickle'
            with open(save_file_name, 'wb', encoding='utf_8') as save_file:
                pickle.dump(self, save_file)

        self.open_set_file()

    def define_output_columns(self, args):

        """
        For writing the selected columns in the same order as they are in the
        original file. Also, to include the case number and output columns in
        the output.
        """

        self.output_columns = list(self.column_ordinals)
        self.output_columns.append(args.case_column)
        self.output_columns.append(args.outcome_column)
        self.output_columns.sort()

    def open_set_file(self):
        """
        Open the set's output file.
        """
        # pylint: disable=consider-using-with
        self.set_file = open(self.file_name, 'w', encoding='utf_8')

# pylint: disable=too-many-instance-attributes
class TrainingSet(SelectionSet):

    """
    A class to represent the information for a training set to be created.

    Each training set has an output file name, output file object and a set of
    row ordinals for the lines of the original file for this training set and
    a set of column ordinals for the columns of the original file for this
    training set.

    If the constructor is passed a ValidationSet object, then a single
    validation set is used for all training sets. No training set will choose
    row ordinals that are in that single validation set.

    If the constructor is not passed a ValidationSet object, then each training
    set has its own validation set. In this case a training set will not choose
    row ordinals that are in its associated validation set, but may choose row
    ordinals that are in the validation set of some other training set.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(self, trainingOrdinal, original_line_count, \
                 original_column_count, single_validation_set, args, column_set):

        SelectionSet.__init__(self)

        # Used for output file names and error mesages.
        self.ordinal = trainingOrdinal

        exclude_validation_set = single_validation_set
        self.validation_set = None
        if single_validation_set is None:
            self.validation_set = ValidationSet(self.ordinal, \
                                               original_line_count, \
                                               original_column_count, args, \
                                               column_set)
            exclude_validation_set = self.validation_set

            # When using a separate validation set for each training set, the
            # training set uses the same columns as the validation set.
            self.column_ordinals = self.validation_set.column_ordinals
        else:
            # The columns to be written for each line for this training set,
            # i.e. the columns of the lines specified by self.row_ordinals. This
            # does not include the case column or output column.
            exclude_cols = set([args.case_column, args.outcome_column])
            if column_set is None:
                self.column_ordinals = self.get_random_ordinals_exclude( \
                                             args.column_count, 1, \
                                             original_column_count, exclude_cols)
            else:
                self.column_ordinals = column_set

        # The rows (lines) of the original data file to be written to this
        # training set.
        # 2, to skip the ordinal for the original file header line.
        self.row_ordinals = self.get_random_ordinals_exclude( \
                                args.training_row_count, 2, original_line_count, \
                                exclude_validation_set.row_ordinals)

        # For writing the selected columns in the same order as they are in the
        # original file. Also, to include the case number and output columns in
        # the output.
        self.output_columns = list(self.column_ordinals)
        self.output_columns.append(args.case_column)
        self.output_columns.append(args.outcome_column)
        self.output_columns.sort()

        self.file_name = f'training-set-{trainingOrdinal}'

        f = f'training-set-{trainingOrdinal}-row-ordinals'
        self.row_ordinal_file_name = f
        self.write_ordinals(self.row_ordinals, self.row_ordinal_file_name)

        f = f'training-set-{trainingOrdinal}-column-ordinals'
        self.column_ordinal_file_name = f

        # If using a single validation set then write the column ordinals for
        # this training set to a file.  If not using a single validation set
        # (using a separate validation set for each training set) then don't
        # write its columns, since they are the same as those for this training
        # set's validation set, and the validation set has already written
        # them.
        if single_validation_set is not None:
            self.write_ordinals(self.column_ordinals, \
                                self.column_ordinal_file_name)

        # pylint: disable=consider-using-with
        self.set_file = open(self.file_name, 'w', encoding='utf_8')

    def check_line(self, ordinal, fields):

        """
        If doing a validation set for each training set, then check this
        training set's validation set if it should write the line.

        In either case check this training set if it should write the line.
        """

        if self.validation_set is not None:
            self.validation_set.check_line(ordinal, fields)

        super().check_line(ordinal, fields)

def define_and_get_args(cmd_args=None):

    """
    Define and get the command line options.
    """

    parser = argparse.ArgumentParser()

    msg = 'The file name of the original data set'
    parser.add_argument('-i', '--original-file', dest='original_file_name', \
                        help=msg, type=str, default=None, required=True)

    msg = 'The file name of the Pickle file with the original data file'
    msg += ' information.'
    parser.add_argument('--odfi', '--original-data-file-info',
                        dest='original_data_file_info', help=msg, type=str, \
                        default=None, required=True)

    msg = 'The number of rows to extract for each training set.'
    parser.add_argument('--trc', '--training-row-count', \
                        dest='training_row_count', help=msg, \
                        type=int, required=True)

    msg = 'The number of rows to extract for each validation set.'
    parser.add_argument('--vrc', '--validation-row-count', \
                        dest='validation_row_count', help=msg, \
                        type=int, required=True)

    msg = 'The number of columns to extract for each validation'
    msg += ' and training set.'
    parser.add_argument('--cc', '--column-count', dest='column_count', \
                        help=msg, type=int, required=True)

    msg = 'The case number column ordinal'
    parser.add_argument('--cn', '--case-column', dest='case_column', \
                        help=msg, type=int, required=True)

    msg = 'The outcome column ordinal'
    parser.add_argument('--oc', '--outcome-column', dest='outcome_column', \
                        help=msg, type=int, required=True)

    msg = 'If present create a validation set for each training set.'
    msg += ' Otherwise create only one validation set.'
    parser.add_argument('--mvs', '--multiple-validation-set', \
                        dest='multiple_validation_set', help=msg, \
                        action='store_true', default=False)

    msg = 'The number of training sets to create'
    parser.add_argument('--tsc', '--training-set-count', \
                        dest='training_set_count', help=msg, type=int, \
                        required=True)

    msg = 'The starting set number'
    parser.add_argument('-s', '--starting-set-number', \
                        dest='starting_set_number', help=msg, type=int, \
                        default=1)

    msg = 'The file name of a file with a resticted set of column ordinals'
    msg += ' to use'
    parser.add_argument('--cs', '--column-set', dest='column_set_file_name', \
                        help=msg, type=str, default=None, required=False)

    cmd_args = parser.parse_args()
    return cmd_args

# pylint: disable-next=too-many-statements
def check_args(cmd_args=None):

    """
    Perform validity checks on the command line arguments.
    """

    args_ok = True

    if not os.path.isfile(cmd_args.original_file_name):
        msg = '\nOriginal data set file "{0}" does not exist.\n'
        msg = msg.format(cmd_args.original_file_name)
        print(msg)
        args_ok = False

    if not os.path.isfile(cmd_args.original_data_file_info):
        msg = '\nValidation ordinal file "{0}" does not exist.\n'
        msg = msg.format(cmd_args.original_data_file_info)
        print(msg)
        args_ok = False

    if cmd_args.training_row_count < 1:
        msg = 'The training row count, {0}, is less than 1.'
        msg = msg.format(cmd_args.training_row_count)
        print(msg)
        args_ok = False

    if cmd_args.validation_row_count < 1:
        msg = 'The validation row count, {0}, is less than 1.'
        msg = msg.format(cmd_args.validation_row_count)
        print(msg)
        args_ok = False

    if cmd_args.column_count < 1:
        msg = 'The column count, {0}, is less than 1.'
        msg = msg.format(cmd_args.column_count)
        print(msg)
        args_ok = False

    if cmd_args.case_column < 1:
        msg = 'The case number column ordinal, {0}, is less than 1.'
        msg = msg.format(cmd_args.case_column)
        print(msg)
        args_ok = False

    if cmd_args.outcome_column < 1:
        msg = 'The outcome column ordinal, {0}, is less than 1.'
        msg = msg.format(cmd_args.outcome_column)
        print(msg)
        args_ok = False

    if cmd_args.case_column == cmd_args.outcome_column:
        msg = 'The case number column ordinal and outcome column are the'
        msg += ' same, {0}'
        msg = msg.format(cmd_args.case_column)
        print(msg)
        args_ok = False

    if cmd_args.starting_set_number < 1:
        msg = 'The starting_set_number, {0}, is less than 1.'
        msg = msg.format(cmd_args.starting_set_number)
        print(msg)
        args_ok = False

    if cmd_args.column_set_file_name is not None and \
       not os.path.isfile(cmd_args.column_set_file_name):
        msg = '\nColumn set file "{0}" does not exist.\n'
        msg = msg.format(cmd_args.column_set_file_name)
        print(msg)
        args_ok = False

    if not args_ok:
        sys.exit(1)

    return cmd_args

def define_and_check_args(cmd_args=None):

    """
    Define, get and check the command line options.
    """
    args = define_and_get_args(cmd_args)
    check_args(args)
    return args

def print_args(cmd_args):

    """
    For testing and debugging.
    """

    print(f'cmd_args.original_file_name: {cmd_args.original_file_name}')

    msg = 'cmd_args.original_data_file_info: {0}'
    print(msg.format(cmd_args.original_data_file_info))

    print(f'cmd_args.training_row_count: {cmd_args.training_row_count}')

    msg = 'cmd_args.validation_row_count: {0}'
    print(msg.format(cmd_args.validation_row_count))

    print(f'cmd_args.column_count: {cmd_args.column_count}')
    print(f'cmd_args.case_column: {cmd_args.case_column}')
    print(f'cmd_args.outcome_column: {cmd_args.outcome_column}')

    msg = 'cmd_args.multiple_validation_set: {0}'
    print(msg.format(cmd_args.multiple_validation_set))

    print(f'cmd_args.training_set_count: {cmd_args.training_set_count}')
    print(f'cmd_args.starting_set_number: {cmd_args.starting_set_number}')
    print(f'cmd_args.column_set_file_name: {cmd_args.column_set_file_name}')

    print('')

def check_args_additional(original_line_count, original_column_count, args):

    """
    Perform argument checks that require information read from files.
    """

    args_ok = True

    # -1 to exclude the header line.
    # - args.validation_row_count to exclude the validation rows.
    available_training_rows = original_line_count - 1 - args.validation_row_count
    if args.training_row_count > available_training_rows:
        msg = 'The training row count, {0}, is greater than the available'
        msg += ' training rows:'
        msg += '\n{1} = {2}(original file rows) - 1(exclude header line) -'
        msg += ' {3}(exclude validation rows).'
        msg = msg.format(args.training_row_count, available_training_rows, \
                         original_line_count, args.validation_row_count)
        print(msg)
        args_ok = False

    if args.case_column > original_column_count:
        msg = 'The case number column ordinal, {0}, is greater than the number'
        msg += ' of columns in the original file, {1}.'
        msg = msg.format(args.case_column, original_column_count)
        print(msg)
        args_ok = False

    if args.outcome_column > original_column_count:
        msg = 'The output column ordinal, {0}, is greater than the number'
        msg += ' of columns in the original file, {1}.'
        msg = msg.format(args.outcome_column, original_column_count)
        print(msg)
        args_ok = False

    if not args_ok:
        sys.exit(1)

def get_column_set(original_column_count, args):

    """
    Get a restricted set of columns to use from a specified file.

    The file has one line per column, with 2 values:

        A column ordinal.

        A value indicating the predictive power ranking of that column.
        The columns should appear in descending order by this ranking.
        The column with the highest ranking should appear first, etc.
        However, this is not checked by this script.

    Only the requested number of columns to use (args.column_count) are read.
    """

    column_set = set()
    with open(args.column_set_file_name, 'r', encoding='utf_8') as input_file:
        ordinal_base = 1
        for (ordinal, line) in enumerate(input_file, ordinal_base):

            if ordinal > args.column_count:
                break

            # Delete trailing newline so it isn't treated as part of the values
            # read.
            fields = line.rstrip('\n').split(',')

            try:
                column = int(fields[1])
            except ValueError:
                msg = 'Column "{0}" from line {1} of file {2} is not an'
                msg += ' integer.'
                msg = msg.format(fields[0], ordinal, args.column_set_file_name)
                print(msg)
                sys.exit(1)

            if column < 1 or column > original_column_count:
                msg = 'Column "{0}" from line {1} of file {2} is out of range.'
                msg += ' It must be between 1 and {3}.'
                msg = msg.format(fields[0], ordinal, args.column_set_file_name, \
                                 original_column_count)
                print(msg)
                sys.exit(1)

            if column == args.case_column:
                msg = 'Column "{0}" from line {1} of file {2} is the same as'
                msg += ' the case column {3}.'
                msg = msg.format(fields[0], ordinal, args.column_set_file_name, \
                                 args.case_column)
                print(msg)
                sys.exit(1)

            if column == args.outcome_column:
                msg = 'Column "{0}" from line {1} of file {2} is the same'
                msg += ' as the output column {3}.'
                msg = msg.format(fields[0], ordinal, args.column_set_file_name, \
                                 args.outcome_column)
                print(msg)
                sys.exit(1)

            column_set.add(column)

    return column_set

def create_selection_sets(original_line_count, original_column_count, args, \
                          column_set):

    """
    Create the validation set and training set objects.
    """

    print('Creating SelectionSet objects...', end='')
    selection_sets = []

    # Get a single validation set if requested.
    #
    # If using a single validation set, only create it (define rows and columns
    # to use) if the starting set number is 1. This avoids creating it multiple
    # times if this script is run multiple times when creating a large number
    # of training sets and only 1 validation set. Multiple runs of this script
    # might be needed to avoid the system max open file limit.
    #
    # Otherwise load a previously created one from a file.
    single_validation_set = None
    if not args.multiple_validation_set:

        if args.starting_set_number == 1:
            # -1 for the set ordinal because there is only one validation set, not
            # one per training set.
            try:
                single_validation_set = ValidationSet(-1, original_line_count, \
                                                   original_column_count, args, \
                                                   column_set, save=True)
                selection_sets.append(single_validation_set)
            # pylint: disable=broad-exception-caught
            except (Exception) as e:
                msg = 'An exception occured creating the single validation set:'
                msg += '\n{0}'
                msg = msg.format(e)
                print(msg)
                sys.exit(1)
        else:
            # Load the validation set created on a prior run of this script,
            # when the starting set number was 1. Needed so the training sets
            # know not to create their own validation sets.
            load_file_name = 'validation-set.pickle'
            with open(load_file_name, 'rb') as load_file:
                single_validation_set = pickle.load(load_file)


    # Create the training sets.
    ending_set_number = args.starting_set_number - 1 + args.training_set_count
    for i in range(args.starting_set_number, ending_set_number + 1):
        try:
            tr_set = TrainingSet(i, original_line_count, original_column_count, \
                                single_validation_set, args, column_set)
            selection_sets.append(tr_set)
        # pylint: disable=broad-exception-caught
        except (Exception) as e:
            msg = 'An exception occured creating TrainingSet {0} of {1}:\n{2}'
            msg = msg.format(i, args.training_set_count, e)
            print(msg)
            sys.exit(1)

    print('...Done')
    return selection_sets

def program_start():
    """
    The main function for the program.

    Putting this code in a function rather than in global scope simplifies
    using pylint. It avoids many pylint complaints such as redfining a variable
    from an outer scope or insisting a variable name refers to a constant
    (pyling considers a constant any variable defined at module level that is
     not bound to a class object).
    """

    cmd_args = define_and_get_args()
    #print_args(cmd_args)

    # Load the original file info.
    with open(cmd_args.original_data_file_info, 'rb') as odi_file:
        (original_line_count, original_column_count) = pickle.load(odi_file)

    check_args_additional(original_line_count, original_column_count, cmd_args)

    # If requested get a restricted set of column ordinals to use.
    column_set = None
    if cmd_args.column_set_file_name is not None:
        column_set = get_column_set(original_column_count, cmd_args)

    # Create SelectionSet objects.
    selection_sets = create_selection_sets(original_line_count, \
                                           original_column_count, cmd_args, \
                                           column_set)

    print('Creating SelectionSet files...', end='')
    with open(cmd_args.original_file_name, 'r', encoding='utf_8') as input_file:
        # Count the line ordinals starting from 1, not the default of 0.
        option_base = 1
        for (ordinal, line) in enumerate(input_file, option_base):

            # Used for debugging.
            #if ordinal == 1:
                #header = line

            # Delete trailing newline from last column, otherwise, if the last
            # column is written to a training set then that training set will have
            # extra blank lines.
            line_fields = line.rstrip('\n').split(',')
            #if len(line_fields) != original_column_count:
                #msg = 'Line {0} has {1} columns, which doesn''t match'
                #msg += ' the header line, which has {2} columns.'
                #msg += '\nHeader line:\n{3}'
                #msg += '\nline {0}:\n{4}'
                #msg = msg.format(ordinal, len(line_fields), original_column_count, \
                                 #header, line)
                #print(msg)
                #sys.exit(0)

            for sel_set in selection_sets:
                sel_set.check_line(ordinal, line_fields)
    print('...Done')

if __name__ == '__main__':
    program_start()
