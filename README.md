# MCR-Agent
<a href="https://bhkim94.github.io/projects/MCR-Agent"> <b> Multi-Level Compositional Reasoning for Interactive Instruction Following </b> </a>
<br>
<a href="https://www.linkedin.com/in/suvaansh-bhambri-1784bab7/"> Suvaansh Bhambri* </a>,
<a href="https://bhkim94.github.io/"> Byeonghwi Kim* </a>,
<a href="http://ppolon.github.io/"> Jonghyun Choi </a>
<br>
<a href="https://aaai.org/Conferences/AAAI-23/"> AAAI 2023 </a>

<b> MCR-Agent </b> (<b>M</b>ulti-Level <b>C</b>ompositional <b>R</b>easoning Agent) is a multi-level compositional approach that learns to navigate and manipulate objects in a divide-and-conquer manner for the diverse nature of the entailing task.
MCR-Agent addresses long-horizon instruction following tasks based on egocentric RGB observations and natural language instructions on the <a href="https://github.com/askforalfred/alfred">ALFRED</a> benchmark.
<br>

<img src="mcr-agent.png" alt="MCR-Agent">

## Code

### Training
To train MCR-Agent, run `train.sh` with hyper-parameters below. <br>

**Note**: As mentioned in the repository of <a href="https://github.com/askforalfred/alfred/tree/master/models">ALFRED</a>, run with `--preprocess` only once for preprocessed json files. <br>


### Evaluation
#### Task Evaluation
First we need to evaluate the individual modules using 'test_unseen.sh' in each module folder. <br>

To evaluate MCR-Agent on ALFRED validation set, input the best model paths in `test_unseen.sh` for unseen fold and `test_seen.sh` for seen fold <br>

**Note**: All hyperparameters used for the experiments in the paper are set as default. <br>
