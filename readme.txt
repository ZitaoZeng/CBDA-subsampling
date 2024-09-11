
This project is a set of scripts for creating validation and training sets for
the CBDA machine learning project.

See the section "Programming Notes" at the end of this document on some
important notes about making changes to the project source files.

For a quick start see the examples at the end of this file.

The idea is to use machine learning to examine patient data, including patient
outcomes, using machine learning algorithms, to determine those patient
attributes that best determine patient outcomes, and how those attributes map
to outcomes.

The intent is that once the mapping is determined, it can be used in a clinical
setting to map a particular patient's attributes to the most likely outcome. In
effect, to improve the diagnoses of clinical conditions.

The machine learning process is to take an actual clinical data set (typically
a large one) of known patient attributes and outcomes and divide it into
various subsets.

There is a separate validation set for each training set.

The rows (lines) of the original data file are partitioned into two disjoint
subsets - a subset for creating the training set samples and a subset for
creating the validation set samples. This ensures that no row in a validation
set appears in any training set.

The number of original data file rows to use is specified as a percentage (a
value between 0 and 1), by a command line option (--tp) to script
create_sets.py. The remaining row if the original file are used for creating
the validation sets.

There are 4 scripts in this part of the CBDA project, 3 Python3 scripts and 1
bash script. For the Python scripts use the "-h" command line option to see the
command line options for running the script.

These scripts assume the original data file from which validation and training
sets are to be produced conforms to the following:

Is a csv file with a header line that contains text labels for each column.

There is a column that contains a case id value.

There is a column that contains the outcome.

All other columns are patient attributes.

The original file may be placed in a zip file archive. The scripts that read
the original data file can read either a plain text version of the file or
the zip file.

If a zip file is specified, the following assumptions are made:
   There is only one file in the file archive.

   The name of the file in the file archive is the same as the
   name of the zip file, except it has no path and it ends with
   '.csv' instead of '.zip'.

The scripts are:

*************************
get_original_file_info.py
*************************

This script reads the original data file and writes a Python Pickle file that
contains the number of lines in the file and the number of columns.  The number
of columns is determined from the first line.

This requires one complete pass of the original data file, to determine the
number of lines in the file.

**************************
list_original_file_info.py
**************************

Lists the contents of a Pickle file created by get_original_file_info.py to
standard output. Useful for testing and debugging the Python scripts.

**************
create_sets.py
**************

This creates the validation set(s) and training sets. It reads the Pickle file
produced by get_original_file_info.py and the original data file.  It uses
command line options for the number of rows and columns to include in each
validation and training set, the column ordinals for the case number and
outcome columns, the number of training sets to produce and the percentage (a
value betwen 0 and 1) of the original data file lines to use for training sets
(with the remainder of the original data file lines used for validation sets).

It excludes the case number and outcome column from the randomly selected
attribute column ordinals to include for a training or validation set, but it
does write those columns to each training set and each validation set, in
addition to the randomly selected attribute columns.

It makes as few passes as possible of the original data set to create all the
validation and training sets. The number of passes of the original data set is
determined by the system limit on the maximum number of open files.

Suppose the system file open limit is 1024 (a common value on Linux systems).
There will already be stdin, stdout and stderr, leaving 1021 open files
available.  If creating 500 training sets, which also means creating 500
validation sets (one for each training set), so 1000 files in total, it takes
one pass of the original data set, not 1,000 passes. If creating 1000 training
sets, which also means creating 1000 validation sets, so 2000 files in total,
it takes two passes, 510 training sets and 510 validation sets on the first
pass, and 490 of each on the second pass.

This script produces the following files.

3 files for each validation set:

validation-set-i.csv:

	The validation set data, the randomly selected rows and columns (plus the case
	number and outcome column). Ex. validation-set-1.csv, validation-set-2.csv, etc.

    The first column is the case number column.
    The second column is the outcome column.
    The remaining columns are the randomly selected columns, in the order they
    appear in the original file.

validation-set-i-row-ordinals:

	The	row ordinals in the original data file for the rows in the validation
	set data.  This may be needed by the machine learning step or the
	validation step.

validation-set-i-column-ordinals:

	The	column ordinals in the original data file for the columns in the
	validation set data, not including the case number or outcome column
	ordinals.  This may be needed by the machine learning step or the
	validation step.

2 files for each training set:

training-set-i.csv:

	The training set data, the randomly selected row and columns (plus the case
	number and outcome column). Ex. training-set-1.csv, training-set-2.csv, etc.

    The first column is the case number column.
    The second column is the outcome column.
    The remaining columns are the randomly selected columns, in the order they
    appear in the original file.

training-set-i-row-ordinals:

	The	row ordinals in the original data file for the rows in the training
	set data.  This may be needed by the machine learning step or the training
	step.

There is no file for training set column ordinals, because a training set uses
the same columns as its validation set.

********************
create_test_data_set
********************

This creates a test data set, including a header line, with command line
options for specifying the number of rows and number of columns.

Each value in a field contains the row number and column number (except for the
header, which just has the column number).

This makes it easy to determine if the training sets are correct - include the
correct rows and columns for each training set, exclude the rows for the
validation set and the rows and columns of a training set are not exactly the
same as those of another (in the vast majority of cases).

An example test data set with 10 rows and 10 columns:

h1,h2,h3,h4,h5,h6,h7,h8,h9,h10
2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,2.10
3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,3.10
4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,4.10
5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,5.10
6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,6.10
7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,7.10
8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,8.10
9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,9.10
10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,10.10
11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.8,11.9,11.10

********
Examples
********

These use the above test file.

They assume the scripts are in the command search path.

See the description above of script create_sets.py for a description of the
contents of the files produced.

get_original_file_info.py -i test-dataset.csv -o odfi
 
This produces file:

odfi.pickle: Has the line count and column count of test-dataset.csv.

create_sets.py -i test-dataset.csv --odfi odfi.pickle --trc 2 --vrc 4 --cc 4 --cn 1 --oc 2 --tsc 4 --tp 0.8

This produces the following files:

training-set-1.csv
training-set-1-row-ordinals

training-set-2.csv
training-set-2-row-ordinals

training-set-3.csv
training-set-3-row-ordinals

training-set-4.csv
training-set-4-row-ordinals

validation-set-1.csv
validation-set-1-column-ordinals
validation-set-1-row-ordinals

validation-set-2.csv
validation-set-2-column-ordinals
validation-set-2-row-ordinals

validation-set-3.csv
validation-set-3-column-ordinals
validation-set-3-row-ordinals

validation-set-4.csv
validation-set-4-column-ordinals
validation-set-4-row-ordinals

*****************
Programming Notes
*****************

This project has had a git pre-commit hook added. The files which implement
this are in subdirectory hooks.

Script hooks/pre-commit runs hooks/pre-commit-sample-modified and then runs
hooks/pre-commit-python.

hooks/pre-commit-sample-modified is a modified copy of the sample pre-commit
hook that git put into .git/hooks. The modifications are noted in this file.
It checks the files being committed for names that have non-ascii characters and
for whitespace issues - trailing whitespace on a line or blank lines at the end
of a file. It aborts the commit if there are any violations of these
conditions.

hooks/pre-commit-python runs each file being committed that is a Python script
through the pylint program, which checks a file for conformance to the Python
Pep8 code standard. Any violations abort the commit.

To enable the pre-commit hook, after cloning the repository, run the command
hooks/setup-hooks. This will update .git/config to use directory hooks as the
location for hooks to run, instead of .git/hooks. This allows versioning the
hooks along with the main code and for easily sharing them with anyone that
clones the repository.
