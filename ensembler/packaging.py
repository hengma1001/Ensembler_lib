import os, sys, warnings, datetime
import shutil, glob

from . import core, utils 
from .core import mpistate, logger 

@utils.notify_when_done 
@utils.mpirank0only 
def package_for_md(
    process_only_these_targets=None, 
    loglevel=None
    ): 
    """
    Collecting results for further MD simulation
    """ 
    utils.set_loglevel(loglevel) 
    targets, templates_resolved_seq = core.get_targets_and_templates() 
    
    packages_dir = os.path.abspath(core.default_project_dirnames.packaged_models)
    simulations_dir = os.path.abspath(core.default_project_dirnames.simulations) 
    
    for target in targets: 
        if (process_only_these_targets is not None) and (target.id not in process_only_these_targets):
            logger.info('Skipping because %s is not in process_only_these_targets' % target.id)
            logger.info(process_only_these_targets)
            continue
        
        simulations_target_dir = os.path.join(simulations_dir, target.id) 
        packages_target_dir = os.path.join(packages_dir, target.id) 
        utils.create_dir(packages_target_dir) 
        
        simulations_target_model_dirs = sorted(glob.glob(os.path.join(simulations_target_dir, target.id + '*'))) 
        
        for simulation_target_model_dir in simulations_target_model_dirs: 
            model_file = os.path.join(simulation_target_model_dir, 'explicit-refined.pdb') 
            simulation_model_name = os.path.join(os.path.basename(os.path.dirname(model_file)), os.path.basename(model_file))[:-4].replace('/', '_') +'.pdb'
            packaged_model = os.path.join(packages_target_dir, simulation_model_name) 
            shutil.copy2(model_file, packaged_model)
        
        