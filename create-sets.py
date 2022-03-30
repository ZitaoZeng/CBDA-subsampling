#!/usr/bin/env python3

# CBDA validation and training set creation.

# This script defines validation and training sets for a CBDA project.
#
# Inputs:
#     File name of original data file.
#         The following assumptions are made about the contents of the file:
#             It is a text file.
#
#             It has a header line with column names.
#
#             It is a csv file.
#
#             All lines have the same number of columns, including the header
#             line.
#
#     The name of a Python Pickle file containing the following information
#     about the original data file:
#        The number of lines in the original data file.
#
#        The number of comma separated columns in the header line of the
#        original data file. It is assumed that all other lines in the file
#        also have that number of columns.
#
#     The number of rows to extract.
#         The specific rows to extract are chosen at random from the original
#         file. The first row is not included, nor are any rows in the
#         validation set.
#
#     The number of columns to extract.
#         The specific columns to extract are chosen at random.
#
#     The case number column ordinal.
#         To exclude from the selection of data columns for a data set, but to
#         be written to each training set in addition to the selected data
#         columns.  This is the column whose value corresponds to each patient.
#
#     The output column ordinal.
#         To exclude from the selection of data columns for a data set, but to
#         be written to each training set in addition to the selected data
#         columns.  This is the outcome column whose value is to be predicted
#         by the algorithm defined by the machine learning processing of the
#         training sets generated here.
#
#     Whether to create one validation set, or to create a validation set for
#     each training set. Optional.
#         If not present create one validation set. If present create a
#         validation set for each training set.
#
#     The number of training sets to create.
#
#     The starting set number. Optional.
#         If not present the default value is 1.
#
#         Needed to create unique output file names when this script is run
#         more than once for an original data file, to create more training and
#         validation sets than the max number of open files allowed.
#
#     The file name of a file containing an optional set of columns to restrict
#     the selection to. Optional.
#         If not present all columns are available to select from, except the
#         case number column and output column.
#
#         If present only the specified columns are available to select from.
#         These must not include the case number column or output column. 
#
#         This is for a second set of training/validation runs to determine
#         which subset of the important columns, identified by the first
#         training/validation runs, are most useful.
#

#         This file has 2 numbers per line:
#
#             A column ordinal.
#
#             A value indicating the predictive power ranking of that column.
#             The columns should appear in descending order by this ranking.
#             The column with the highest ranking should appear first, etc.
#             However, this is not checked by this script.
#
# Outputs:
#    An output file for each training set.
#
#    Either:
#
#        One validation set file:
#            No training set will contain the any of the original data file
#            lines as the validation set file.
#
#        A validation set file for each training set file.
#            A training set file will not contain any of the original data file
#            lines as its corresponding validation set file. 

import sys
import argparse
import os
import pickle
import random

class SelectionSet:

    """

    A base class for data sets to be selected from an original data file.

    Each subclass should have the following members:

		rowOrdinals: A Python set object that has the row ordinals of the rows
                     (lines) of the original file to be written for this set.

		outputColumns: A list of the columns to write from a line of the
                       original file.

        setFile: An open file object to write the selected data to.

    """

    def __init__(self):
        self.rowOrdinals = None
        self.outputColumns = None
        self.setFile = None

    def getRandomOrdinals(self, count, end):

        """
        Generate the set of validation line ordinals.
        Get a set of count random integers between 2 and end, inclusive.
        2 to avoid the header line.

        count and end should be integers.
      
        count should be < (end - 1) (-1 to exclude the header line)
        """

		# Make it a set so a training set can efficiently avoid using any
		# validation set line ordinal.
        ordinalList = random.sample(range(2, end+1), count)
        ordinalSet = set(ordinalList)
        return ordinalSet

    def getRandomOrdinalsExclude(self, count, start, end, exclude):
    
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
    
    def writeOrdinals(self, ordinals, fileName):
    
    	"""
    	Write a set of ordinals to a file, in ascending numerical order.
    
    	These are typically needed by subsequent machine learning steps, not part
        of the data set selection process.
    	"""
    
    	sortedOrdinals = sorted(ordinals)
    	with open(fileName, 'w') as ordinalFile:
    		for o in sortedOrdinals:
    			ordinalFile.write(str(o) + '\n')

    def checkLine(self, ordinal, fields):

        """
        Check a line from the original file, to see if fields from it should be
        written for this training set.

        outputColumns is sorted in ascending order by column ordinal, so the
        columns will be written in the same order they are in the original
        file. It also includes the case number column and output column, in
        addition to the selected data columns.
        """

        if ordinal in self.rowOrdinals:
            # This line is for this selection set.

            # Get the fields for this selection set.
            if self.outputColumns == None:
                # Get all the fields.
                fieldsToWrite = fields
            else:
                fieldsToWrite = []
                for o in self.outputColumns:

                    # Because we count column ordinals from 1, but list indices
                    # start at 0.
                    o1 = o - 1

                    fieldsToWrite.append(fields[o1]) 

            # Write the fields to the training set file.
            fieldStr = ','.join(fieldsToWrite) + '\n'
            self.setFile.write(fieldStr)

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

    def __init__(self, ordinal, originalLineCount, originalColumnCount, \
                 args, columnSet, save=False):

        SelectionSet.__init__(self)

        # Used for output file names and error mesages.
		# If only one validation set is being created this is -1. If there is a
		# validation set for each training set then this is the same ordinal as
		# the associated training set.
        self.ordinal = ordinal

        # 2, to skip the ordinal for the original file header line.
        self.rowOrdinals = self.getRandomOrdinals(args.validationRowCount, 
                                                  originalLineCount)

        # Determine the columns to use for this validation set. If a column set
        # was provided, use it. Otherwise if not saving use a random set of
        # columns and if saving use all columns (columnOrdinals and
        # outputColumns remain None).
        self.columnOrdinals = None
        self.outputColumns = None
        if columnSet != None:
            self.columnOrdinals = columnSet
            self.defineOutputColumns(args)
        elif not save:
            excludeCols = set([args.caseColumn, args.outcomeColumn])
            self.columnOrdinals = self.getRandomOrdinalsExclude(
                                          args.columnCount, 1, \
                                          originalColumnCount, excludeCols)
            self.defineOutputColumns(args)

        fileNameOrdinal = ''
        if self.ordinal != -1:
            fileNameOrdinal = '-{0}'.format(ordinal)

        self.fileName = 'validation-set{0}'.format(fileNameOrdinal)

        f = 'validation-set{0}-row-ordinals'.format(fileNameOrdinal)
        self.rowOrdinalFileName = f
        self.writeOrdinals(self.rowOrdinals, self.rowOrdinalFileName)

        f = 'validation-set{0}-column-ordinals'.format(fileNameOrdinal)
        self.columnOrdinalFileName = f
        if self.columnOrdinals != None: 
            self.writeOrdinals(self.columnOrdinals, self.columnOrdinalFileName)

        # This must come before opening the validation set file, since an open
        # file object (or perhaps any file object) can't be pickled.
        if save:
            saveFileName = self.fileName + '.pickle'
            with open(saveFileName, 'wb') as saveFile:
                pickle.dump(self, saveFile)

        self.openSetFile()

    def defineOutputColumns(self, args):

        """
        For writing the selected columns in the same order as they are in the
        original file. Also, to include the case number and output columns in
        the output.
        """

        self.outputColumns = list(self.columnOrdinals)
        self.outputColumns.append(args.caseColumn)
        self.outputColumns.append(args.outcomeColumn)
        self.outputColumns.sort()

    def openSetFile(self):
        self.setFile = open(self.fileName, 'w')

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

    def __init__(self, trainingOrdinal, originalLineCount, \
                 originalColumnCount, singleValidationSet, args, columnSet):

        SelectionSet.__init__(self)

        # Used for output file names and error mesages.
        self.ordinal = trainingOrdinal

        excludeValidationSet = singleValidationSet
        self.validationSet = None
        if singleValidationSet == None:
            self.validationSet = ValidationSet(self.ordinal, \
                                               originalLineCount, \
                                               originalColumnCount, args, \
                                               columnSet)
            excludeValidationSet = self.validationSet

            # When using a separate validation set for each training set, the
            # training set uses the same columns as the validation set.
            self.columnOrdinals = self.validationSet.columnOrdinals
        else:
            # The columns to be written for each line for this training set,
            # i.e. the columns of the lines specified by self.rowOrdinals. This
            # does not include the case column or output column. 
            excludeCols = set([args.caseColumn, args.outcomeColumn])
            if columnSet == None:
                self.columnOrdinals = self.getRandomOrdinalsExclude( \
                                             args.columnCount, 1, \
                                             originalColumnCount, excludeCols)
            else:
                self.columnOrdinals = columnSet

        # The rows (lines) of the original data file to be written to this
        # training set.
        # 2, to skip the ordinal for the original file header line.
        self.rowOrdinals = self.getRandomOrdinalsExclude(args.trainingRowCount, \
                                                        2, originalLineCount, \
                                              excludeValidationSet.rowOrdinals)

        # For writing the selected columns in the same order as they are in the
        # original file. Also, to include the case number and output columns in
        # the output.
        self.outputColumns = list(self.columnOrdinals)
        self.outputColumns.append(args.caseColumn)
        self.outputColumns.append(args.outcomeColumn)
        self.outputColumns.sort()

        self.fileName = 'training-set-{0}'.format(trainingOrdinal)

        f = 'training-set-{0}-row-ordinals'.format(trainingOrdinal)
        self.rowOrdinalFileName = f
        self.writeOrdinals(self.rowOrdinals, self.rowOrdinalFileName)

        f = 'training-set-{0}-column-ordinals'.format(trainingOrdinal)
        self.columnOrdinalFileName = f

        # If using a single validation set then write the column ordinals for
        # this training set to a file.  If not using a single validation set
        # (using a separate validation set for each training set) then don't
        # write its columns, since they are the same as those for this training
        # set's validation set, and the validation set has already written
        # them.
        if singleValidationSet != None:
            self.writeOrdinals(self.columnOrdinals, self.columnOrdinalFileName)

        self.setFile = open(self.fileName, 'w')

    def checkLine(self, ordinal, fields):

        """
		If doing a validation set for each training set, then check this
        training set's validation set if it should write the line.

        In either case check this training set if it should write the line.
        """

        if self.validationSet != None:
            self.validationSet.checkLine(ordinal, fields)

        super().checkLine(ordinal, fields)

def defineArgs(args=None):

    parser = argparse.ArgumentParser()

    msg = 'The file name of the original data set'
    parser.add_argument('-i', '--original-file', dest='originalFileName', \
                        help=msg, type=str, default=None, required=True)

    msg = 'The file name of the Pickle file with the original data file'
    msg += ' information.'
    parser.add_argument('--odfi', '--original-data-file-info',
                        dest='originalDataFileInfo', help=msg, type=str, \
                        default=None, required=True)

    msg = 'The number of rows to extract for each training set.'
    parser.add_argument('--trc', '--training-row-count', \
                        dest='trainingRowCount', help=msg, \
                        type=int, required=True)

    msg = 'The number of rows to extract for each validation set.'
    parser.add_argument('--vrc', '--validation-row-count', \
                        dest='validationRowCount', help=msg, \
                        type=int, required=True)

    msg = 'The number of columns to extract for each validation'
    msg += ' and training set.'
    parser.add_argument('--cc', '--column-count', dest='columnCount', \
                        help=msg, type=int, required=True)

    msg = 'The case number column ordinal'
    parser.add_argument('--cn', '--case-column', dest='caseColumn', \
                        help=msg, type=int, required=True)

    msg = 'The outcome column ordinal'
    parser.add_argument('--oc', '--outcome-column', dest='outcomeColumn', \
                        help=msg, type=int, required=True)

    msg = 'If present create a validation set for each training set.'
    msg += ' Otherwise create only one validation set.'
    parser.add_argument('--mvs', '--multiple-validation-set', \
                        dest='multipleValidationSet', help=msg, \
                        action='store_true', default=False)

    msg = 'The number of training sets to create'
    parser.add_argument('--tsc', '--training-set-count', \
                        dest='trainingSetCount', help=msg, type=int, \
                        required=True)

    msg = 'The starting set number'
    parser.add_argument('-s', '--starting-set-number', \
                        dest='startingSetNumber', help=msg, type=int, \
                        default=1)

    msg = 'The file name of a file with a resticted set of column ordinals'
    msg += ' to use'
    parser.add_argument('--cs', '--column-set', dest='columnSetFileName', \
                        help=msg, type=str, default=None, required=False)

    args = parser.parse_args()
    
    argsOk = True

    if not os.path.isfile(args.originalFileName):
        msg = '\nOriginal data set file "{0}" does not exist.\n'
        msg = msg.format(args.originalFileName)
        print(msg)
        argsOk = False

    if not os.path.isfile(args.originalDataFileInfo):
        msg = '\nValidation ordinal file "{0}" does not exist.\n'
        msg = msg.format(args.originalDataFileInfo)
        print(msg)
        argsOk = False

    if args.trainingRowCount < 1:
        msg = 'The training row count, {0}, is less than 1.'
        msg = msg.format(args.trainingRowCount)
        print(msg)
        argsOk = False

    if args.validationRowCount < 1:
        msg = 'The validation row count, {0}, is less than 1.'
        msg = msg.format(args.validationRowCount)
        print(msg)
        argsOk = False

    if args.columnCount < 1:
        msg = 'The column count, {0}, is less than 1.'
        msg = msg.format(args.columnCount)
        print(msg)
        argsOk = False

    if args.caseColumn < 1:
        msg = 'The case number column ordinal, {0}, is less than 1.'
        msg = msg.format(args.caseColumn)
        print(msg)
        argsOk = False

    if args.outcomeColumn < 1:
        msg = 'The outcome column ordinal, {0}, is less than 1.'
        msg = msg.format(args.outcomeColumn)
        print(msg)
        argsOk = False

    if args.caseColumn == args.outcomeColumn:
        msg = 'The case number column ordinal and outcome column are the'
        msg += ' same, {0}'
        msg = msg.format(args.caseColumn)
        print(msg)
        argsOk = False

    if args.startingSetNumber < 1:
        msg = 'The startingSetNumber, {0}, is less than 1.'
        msg = msg.format(args.startingSetNumber)
        print(msg)
        argsOk = False

    if args.columnSetFileName != None and \
       not os.path.isfile(args.columnSetFileName):
        msg = '\nColumn set file "{0}" does not exist.\n'
        msg = msg.format(args.columnSetFileName)
        print(msg)
        argsOk = False

    if not argsOk:
        sys.exit(1)

    return args

def printArgs(args):
    
    """
    For testing and debugging.
    """

    print('args.originalFileName: {0}'.format(args.originalFileName))
    print('args.originalDataFileInfo: {0}'.format(args.originalDataFileInfo))
    print('args.trainingRowCount: {0}'.format(args.trainingRowCount))
    print('args.validationRowCount: {0}'.format(args.validationRowCount))
    print('args.columnCount: {0}'.format(args.columnCount))
    print('args.caseColumn: {0}'.format(args.caseColumn))
    print('args.outcomeColumn: {0}'.format(args.outcomeColumn))
    print('args.multipleValidationSet: {0}'.format(args.multipleValidationSet))
    print('args.trainingSetCount: {0}'.format(args.trainingSetCount))
    print('args.startingSetNumber: {0}'.format(args.startingSetNumber))
    print('args.columnSetFileName: {0}'.format(args.columnSetFileName))

    print

def checkArgs(originalLineCount, originalColumnCount, args):

    """
    Perform argument checks that require information read from files.
    """
    
    argsOk = True

    # -1 to exclude the header line.
    # - args.validationRowCount to exclude the validation rows.
    availableTrainingRows = originalLineCount - 1 - args.validationRowCount
    if args.trainingRowCount > availableTrainingRows:
        msg = 'The training row count, {0}, is greater than the available'
        msg += ' training rows:'
        msg += '\n{1} = {2}(original file rows) - 1(exclude header line) -'
        msg += ' {3}(exclude validation rows).'
        msg = msg.format(args.trainingRowCount, availableTrainingRows, \
                         originalLineCount, args.validationRowCount)
        print(msg)
        argsOk = False

    if args.caseColumn > originalColumnCount:
        msg = 'The case number column ordinal, {0}, is greater than the number'
        msg += ' of columns in the original file, {1}.'
        msg = msg.format(args.caseColumn, originalColumnCount)
        print(msg)
        argsOk = False

    if args.outcomeColumn > originalColumnCount:
        msg = 'The output column ordinal, {0}, is greater than the number'
        msg += ' of columns in the original file, {1}.'
        msg = msg.format(args.outcomeColumn, originalColumnCount)
        print(msg)
        argsOk = False

    if not argsOk:
        sys.exit(1)

def getColumnSet(originalColumnCount, args):

    """
    Get a restricted set of columns to use from a specified file.

    The file has one line per column, with 2 values:

        A column ordinal.
        
        A value indicating the predictive power ranking of that column.
        The columns should appear in descending order by this ranking.
        The column with the highest ranking should appear first, etc.
        However, this is not checked by this script.

    Only the requested number of columns to use (args.columnCount) are read.
    """

    columnSet = set()
    with open(args.columnSetFileName, 'r') as inputFile:
        ordinalBase = 1
        for (ordinal, line) in enumerate(inputFile, ordinalBase):

            if ordinal > args.columnCount:
                break;
    
			# Delete trailing newline so it isn't treated as part of the values
			# read.
            fields = line.rstrip('\n').split(',')

            try:
                column = int(fields[1])
            except ValueError as e:
                msg = 'Column "{0}" from line {1} of file {2} is not an'
                msg += ' integer.'
                msg = msg.format(fields[0], ordinal, args.columnSetFileName)
                print(msg)
                sys.exit(1)

            if column < 1 or column > originalColumnCount:
                msg = 'Column "{0}" from line {1} of file {2} is out of range.'
                msg += ' It must be between 1 and {3}.'
                msg = msg.format(fields[0], ordinal, args.columnSetFileName, \
                                 originalColumnCount)
                print(msg)
                sys.exit(1)

            if column == args.caseColumn:
                msg = 'Column "{0}" from line {1} of file {2} is the same as'
                msg += ' the case column {3}.'
                msg = msg.format(fields[0], ordinal, args.columnSetFileName, \
                                 args.caseColumn)
                print(msg)
                sys.exit(1)

            if column == args.outcomeColumn:
                msg = 'Column "{0}" from line {1} of file {2} is the same'
                msg += ' as the output column {3}.'
                msg = msg.format(fields[0], ordinal, args.columnSetFileName, \
                                 args.outcomeColumn)
                print(msg)
                sys.exit(1)

            columnSet.add(column)

    return columnSet

def createSelectionSets(originalLineCount, originalColumnCount, args, \
                        columnSet):

    """
    Create the validation set and training set objects.
    """

    print('Creating SelectionSet objects...', end='')
    selectionSets = []

    # Get a single validation set if requested.
    #
    # If using a single validation set, only create it (define rows and columns
    # to use) if the starting set number is 1. This avoids creating it multiple
    # times if this script is run multiple times when creating a large number
    # of training sets and only 1 validation set. Multiple runs of this script
    # might be needed to avoid the system max open file limit.
    #
    # Otherwise load a previously created one from a file.
    singleValidationSet = None
    if not args.multipleValidationSet:

        if args.startingSetNumber == 1:
    		# -1 for the set ordinal because there is only one validation set, not
    		# one per training set.
            try:
                singleValidationSet = ValidationSet(-1, originalLineCount, \
                                                   originalColumnCount, args, \
                                                   columnSet, save=True)
                selectionSets.append(singleValidationSet)
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
            loadFileName = 'validation-set.pickle'
            with open(loadFileName, 'rb') as loadFile:
                singleValidationSet = pickle.load(loadFile)
            

    # Create the training sets.
    endingSetNumber = args.startingSetNumber - 1 + args.trainingSetCount
    for i in range(args.startingSetNumber, endingSetNumber + 1):
        try:
            trSet = TrainingSet(i, originalLineCount, originalColumnCount, \
                                singleValidationSet, args, columnSet)
            selectionSets.append(trSet)
        except (Exception) as e:
            msg = 'An exception occured creating TrainingSet {0} of {1}:\n{2}'
            msg = msg.format(i, args.trainingSetCount, e)
            print(msg)
            sys.exit(1)

    print('...Done')
    return selectionSets



args = defineArgs()
#printArgs(args)

# Load the original file info.
with open(args.originalDataFileInfo, 'rb') as odfiFile:
    (originalLineCount, originalColumnCount) = pickle.load(odfiFile)

checkArgs(originalLineCount, originalColumnCount, args)

# If requested get a restricted set of column ordinals to use.
columnSet = None
if args.columnSetFileName != None:
    columnSet = getColumnSet(originalColumnCount, args)

# Create SelectionSet objects.
selectionSets = createSelectionSets(originalLineCount, originalColumnCount, \
                                    args, columnSet)

print('Creating SelectionSet files...', end='')
with open(args.originalFileName, 'r') as inputFile:
    # Count the line ordinals starting from 1, not the default of 0.
    ordinalbase = 1
    for (ordinal, line) in enumerate(inputFile, ordinalbase):

        if ordinal == 1:
            header = line

        # Delete trailing newline from last column, otherwise, if the last
        # column is written to a training set then that training set will have
        # extra blank lines.
        fields = line.rstrip('\n').split(',')
        #if len(fields) != originalColumnCount:
            #msg = 'Line {0} has {1} columns, which doesn''t match'
            #msg += ' the header line, which has {2} columns.'
            #msg += '\nHeader line:\n{3}'
            #msg += '\nline {0}:\n{4}'
            #msg = msg.format(ordinal, len(fields), originalColumnCount, \
                             #header, line)
            #print(msg)
            #sys.exit(0)

        for selSet in selectionSets:
            selSet.checkLine(ordinal, fields)
print('...Done')
