import sys, os

sys.path.append('../')

import ensembler

# from ensembler.build_env import Ensembler_proj_init, GatherTargetsFromUniprot, gather_templates_from_uniprot 
# from ensembler.modeling import model_template_loops
from ensembler.core import default_project_dirnames, mpistate 


loglevel='debug'
search_string = 'accession:p42336 taxonomy:"Homo sapiens (Human) [9606]"'


# # Initializaion of the project
if mpistate.rank == 0:
    ensembler.build_env.Ensembler_proj_init()


# # Download target from Uniprot
if mpistate.rank == 0:    
    ensembler.build_env.GatherTargetsFromUniprot(search_string, loglevel=loglevel)


# # Download templates from Uniprot
ensembler.build_env.gather_templates_from_uniprot(search_string, loglevel=loglevel)


mpistate.comm.Barrier()
# # Loop model
ensembler.modeling.model_template_loops(loglevel=loglevel)


# # alignment 
ensembler.modeling.align_targets_and_templates(loglevel=loglevel)

# # Modeling
ensembler.modeling.build_models(loglevel=loglevel) 
