# If you are getting errors or not getting the output in PART 1 then try PART 2

# -------------PART 1-------------
> pip install -r requirements.txt
## Run the file with .uai and .evid files as arguments
## for e.g.
> python bucket_elimination.py input_files/1.uai input_files/1.uai.evid


# -------------PART 2-------------
# Steps to run the code... commands are tested in linux.. you can apply alternative commands for windows/MacOS
## Step 1 creating a virtual environment to run the code so that it does not conflicts with other instaled packages on the machine
> python3 -m venv my_env
## Step 2 if the above gives error then make sure your python version is 3.6 or above and install the venv package. If no error move to Step 3
	### for linux and MacOS
	> python3 -m pip install --user virtualenv
	### for windows
	> py -m pip install --user virtualenv

## Step 3 activate the environment
> source my_env/bin/activate


## Step 2 use requirements.txt file to install required packages
> pip install -r requirements.txt

## Run the file with .uai and .evid files as arguments
## for e.g.
> python bucket_elimination.py input_files/1.uai input_files/1.uai.evid

## after running the above you'll see the output printed on the screen and also written to output.txt in the current working folder

### once done with grading of the code you can deactivate the environment and delete it
> deactivate
> rm -r my_env

## Above code is tested on linux. If there are any issues which can't be resolved feel free to email me at zxd200000@utdallas.edu
