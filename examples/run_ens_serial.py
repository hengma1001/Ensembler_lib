import sys, os

sys.path.append('../')

import ensembler

from ensembler.build_env import Ensembler_proj_init, GatherTargetsFromUniprot, gather_templates_from_uniprot 
from ensembler.modeling import model_template_loops, align_targets_and_templates
from ensembler.core import default_project_dirnames, mpistate 

search_string = 'accession:p42336 taxonomy:"Homo sapiens (Human) [9606]"'
loglevel = 'debug'

# # Initializaion of the project
ensembler.build_env.Ensembler_proj_init()


# # Download target from Uniprot
ensembler.build_env.GatherTargetsFromUniprot(search_string)


# # Download templates from Uniprot
ensembler.build_env.gather_templates_from_uniprot(search_string, loglevel=loglevel)

# # Loop model
model_template_loops(loglevel=loglevel)

# # Align the sequence
align_targets_and_templates(loglevel=loglevel)
