
This project is a set of scripts for creating validation and training sets for
the CBDA machine learning project.

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

There are 2 approaches to this, which we will call Option 1 and Option 2.

********
Option 1
********

One subset is a validation set - a (typically small) subset of patients from
the data set set aside to test the machine learning derived mapping between
attributes and outcomes. This is chosen randomly from the original data set.
For example, 1,000 patient records from a file of 100,000 patient records.

A collection of training subsets is also created, used as inputs to the machine
learning algorithm. Each training subset is a randomly selected set of patient
records, but also only a randomly selected subset of attributes (data columns)
from those records. For example, from a file of 100,000 patient records, 1,000
training sets might be created each consisting of 100 patient records with 10
randomly selected attributes out of 1,000 attributes per patient record.

A training set might (and typically will) have some of the same patient records
as one or more other training subsets, but it is unlikely two training subsets
will have the exact same set of records. Similarly for the patient attributes -
there might, and typicall will be, some overlap among different training sets,
but it is unlikely any 2 will have exactly the same attributes.

However, it is desirable for none of the validation set records to be included
in any of the training sets, so the validation step operates on records not
included in the machine learning step.

The training sets are used as input to the machine learning algorithm,
producing a mapping between attributes and outcomes. Then a validation step is
performed which uses the mapping to map attributes in the records of the
validation set to expected outcomes. The expected outcomes are compared to the
actual outcomes for the validation set as a test of the accuracy of the
attribute to outcome mapping.

********
Option 2
********

In option 2, instead of one validation set, there is a separate validation set
for each training set.

The training set and its associated validation do not share any rows in common.
However, both have the same randomly selected subset of attributes.

The randomly selected rows from the original file for a training set are
different from the randomly selected rows from the original file for the
validation set associated with that training set.  However some of the rows for
a training set might be the same as some of the rows of a validation set
associated with another training set.


There are 4 scripts in this part of the CBDA project, 3 Python3 scripts and 1
bash script. For the Python scripts use the "-h" command line option to see the
command line options for running the script.

These scripts assume the original data file from which validation and training
sets are to be produced conforms to the following:

Is a csv file with a header line that contains text labels for each column.

There is a column that contains a case id value.

There is a column that contains the outcome.

All other columns are patient attributes.

The scripts are:

*************************
get-original-file-info.py
*************************

This script reads the original data file and writes a Python Pickle file that
contains the number of lines in the file and the number of columns.  The number
of columns is determined from the first line.

This requires one complete pass of the original data file, to determine the
number of lines in the file.

**************************
list-original-file-info.py
**************************

Lists the contents of a Pickle file created by get-original-file-info.py to
standard output. Useful for testing and debugging the Python scripts.

**************
create-sets.py
**************

This creates the validation set(s) and training sets. It reads the Pickle file
prodcued by get-original-file-info.py and the original data file.  It uses
command line options for the number of rows and columns to include in each
validation and training set, the column ordinals for the case number and
outcome columns and the number of training sets to produce.

It also has a command line option for choosing between Option 1 and Option 2 -
between a single validation set or a separate validation set for each training
set.

It excludes the case number and outcome column from the randomly selected
attribute column ordinals to include for a training set, but it does write
those columns to each training set, in addition to the randomly selected
attribute columns.

It makes a single pass of the original data set to create all the validation
and training sets. If creating a single validation set and 1,000 training sets,
it takes one pass of the original data set, not 1,001 passes.

This is limited by the maximum number of open files a user or user process can
have on a system, which is usually around 1,020 on a Linux system.

This script produces the following files, which is different for Option 1 and
Option 2.

Option 1 files:

3 files for the single validation set:

validation-set:

	The validation set data, the randomly selected rows for the validation set.
    All columns from the original data file are included.

validation-set-row-ordinals:

	The	row ordinals for the rows in the validation set data.  This may be needed
	by the machine learning step or the validation step.

validation-set.pickle:

	A Python Pickle file with the ValidationSet object used to create files
	validation-set and validation-set-row-ordinals. This is needed if script
	create-sets.py needs to be run more than once if the number of training
	sets to create exceeds the maximum number of open files allowed.

	The first run of create-sets.py will create the validation files.
	Subsequent runs for a given original file will read this Pickle file to
	avoid recreating them again. This is not only, or primarily, an effciency
	issue. This is to use a single set of validation row ordinals that are not
	to be used for any training set created from the same original data set.

3 files for each training set:

training-set-i:

	The training set data, the randomly selected row and columns (plus the case
	number and outcome column). Ex. training-set-1, training-set-2, etc.

training-set-i-row-ordinals:

	The	row ordinals for the rows in the training set data.  This may be needed
	by the machine learning step or the validation step.

training-set-i-column-ordinals:

	The	column ordinals for the columns in the training set data, not including
	the case number or outcome column ordinals.  This may be needed by the
	machine learning step or the validation step.

Option 2 files:

3 files for each validation set:

validation-set-i:

	The validation set data, the randomly selected row and columns (plus the case
	number and outcome column). Ex. validation-set-1, validation-set-2, etc.

validation-set-i-row-ordinals:

	The	row ordinals for the rows in the validation set data.  This may be needed
	by the machine learning step or the validation step.

validation-set-i-column-ordinals:

	The	column ordinals for the columns in the validation set data, not including
	the case number or outcome column ordinals.  This may be needed by the
	machine learning step or the validation step.

2 files for each training set:

training-set-i:

	The training set data, the randomly selected row and columns (plus the case
	number and outcome column). Ex. training-set-1, training-set-2, etc.

training-set-i-row-ordinals:

	The	row ordinals for the rows in the training set data.  This may be needed
	by the machine learning step or the training step.

There is no file for training set column ordinals, because for Option 2 a
training set uses the same columns as its validation set.

********************
create-test-data-set
********************

This created a test data set, including a header line, with command line
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

See the description above of script create-sets.py for a description of the
contents of the files produced.

****************
Option 1 example:
****************

original-file-info.py -i test-dataset.csv -o odfi
 
This produces file:

odfi.pickle: Has the line count and column count of test-dataset.csv.

create-sets.py -i test-dataset.csv --odfi odfi.pickle --trc 2 --vrc 4 --cc 4 --cn 1 --oc 2 --tsc 4

training-set-1
training-set-1-column-ordinals
training-set-1-row-ordinals

training-set-2
training-set-2-column-ordinals
training-set-2-row-ordinals

training-set-3
training-set-3-column-ordinals
training-set-3-row-ordinals

training-set-4
training-set-4-column-ordinals
training-set-4-row-ordinals

validation-set
validation-set.pickle
validation-set-row-ordinals

****************
Option 2 example:
****************

original-file-info.py -i test-dataset.csv -o odfi
 
This produces file:

odfi.pickle: Has the line count and column count of test-dataset.csv.

create-sets.py -i test-dataset.csv --odfi odfi.pickle --trc 2 --vrc 4 --cc 4 --cn 1 --oc 2 --tsc 4 --mvs

--mvs means multiple validation sets (i.e. Option 2).

This prodcues the following files:

training-set-1
training-set-1-row-ordinals

training-set-2
training-set-2-row-ordinals

training-set-3
training-set-3-row-ordinals

training-set-4
training-set-4-row-ordinals

validation-set-1
validation-set-1-column-ordinals
validation-set-1-row-ordinals

validation-set-2
validation-set-2-column-ordinals
validation-set-2-row-ordinals

validation-set-3
validation-set-3-column-ordinals
validation-set-3-row-ordinals

validation-set-4
validation-set-4-column-ordinals
validation-set-4-row-ordinals

