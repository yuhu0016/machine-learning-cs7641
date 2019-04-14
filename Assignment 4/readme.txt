
Github link: 




CS-7641 Machine Learning Project 4 - Markov Decision Processes

Navigate to the root folder, then install requirements by entering the command below: 

pip install -r requirements.txt

Next, run the python script by entering the following command: 

python run_experiment.py --all --plot --verbose

If running experiments by argument, entering the following for a list of commands:
 python run_experiment.py --h 

 List of arguments below: 

        --threads     Number of threads (defaults to -1 for auto)
        --seed        A random seed to set, if desired
        --policy      Run the Policy Iteration (PI) experiment
        --value       Run the Value Iteration (VI) experiment
        --ql          Run the Q-Learner (QL) experiment
        --all         Run all experiments (policy, value, ql)
        --plot        Plot data results
        --verbose     If true, provide verbose output (defaults to false)
        --help        Display the help message and exit


Citations: 

All code was adopted from Michael Mallo's CS 7641 repository:

https://github.gatech.edu/mmallo3/CS7641_Project4

