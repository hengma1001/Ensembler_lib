import os, sys, warnings, datetime
import shutil, glob
import traceback 

from . import core, utils 
from .core import mpistate, logger 


import simtk.unit as unit
import simtk.openmm as openmm
import simtk.openmm.app as app
import simtk.openmm.version


def refine_implicit_md(
        openmm_platform=None, gpupn=1, process_only_these_targets=None,
        write_trajectory=False,
#         include_disulfide_bonds=False,
#         custom_residue_variants=None,
        ff='amber99sbildn',
        implicit_water_model='amber99_obc',
        sim_length=100.0 * unit.picoseconds,          
        timestep=2.0 * unit.femtoseconds,             # timestep
        temperature=300.0 * unit.kelvin,              # simulation temperature
        collision_rate=20.0 / unit.picoseconds,       # Langevin collision rate
        cutoff=15 * unit.angstrom,                    # nonbonded cutoff
        minimization_tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        minimization_steps=200,
        nsteps_per_iteration=500,
        ph=7,
        retry_failed_runs=False,
        cpu_platform_threads=1,
        loglevel=None):
    """
    Run MD simulation with implicit water model
    
    MPI-enabled 
    """
    
    utils.set_loglevel(loglevel) 
    gpuid = mpistate.rank % gpupn 
    
    models_dir = os.path.abspath(core.default_project_dirnames.models)
    simulations_dir = os.path.abspath(core.default_project_dirnames.simulations)

    targets, templates_resolved_seq = core.get_targets_and_templates() 
    
#     if process_only_these_templates:
#         selected_template_indices = [i for i, seq in enumerate(templates_resolved_seq) if seq.id in process_only_these_templates]
#     else:
#         selected_template_indices = range(len(templates_resolved_seq))
    
    if not openmm_platform:
        openmm_platform = auto_select_openmm_platform()
        
    if openmm_platform == 'CPU':
        platform_properties = {'CpuThreads': str(cpu_platform_threads)}
    else:
        platform_properties = {}
        
    ff_files = [ff+'.xml', implicit_water_model+'.xml']
    forcefield = app.ForceField(*ff_files)

    def simulate_implicit_md(): 
        logger.debug("Reading model...")
        pdb = app.PDBFile(model_file)

        # Set up Platform 
        platform = openmm.Platform.getPlatformByName(openmm_platform)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            if openmm_platform == 'CUDA': 
                properties = {'DeviceIndex': str(gpuid), 'CudaPrecision': 'mixed'} 
            if openmm_platform == 'OpenCL': 
                properties = {'DeviceIndex': str(gpuid)}

        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield, pH=ph) 
        topology = modeller.getTopology()
        positions = modeller.getPositions() 

        logger.debug("Constructing System object...")
        if cutoff is None: 
            logger.debug("Using NoCutoff for the MD simulation...")
            system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        else:
            system = forcefield.createSystem(topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=cutoff, constraints=app.HBonds)

        logger.debug("Creating Context...") 
        integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
        #         context = openmm.Context(system, integrator, platform, platform_properties)
        simulation = app.Simulation(topology, system, integrator, platform, properties)
        simulation.context.setPositions(positions) 

        logger.debug("Minimizing structure...")
        simulation.minimizeEnergy(tolerance=minimization_tolerance, maxIterations=minimization_steps)

        if write_trajectory: 
            trajectory_filename = os.path.join(simulation_model_dir, 'implicit-trajectory.dcd')
            simulation.reporters.append(app.DCDReporter(trajectory_filename, nsteps_per_iteration)) 
        energy_log_filename = os.path.join(simulation_model_dir, 'implicit-energies.txt') 
        simulation.reporters.append(app.StateDataReporter(energy_log_filename, 
                nsteps_per_iteration, step=True, time=True, speed=True, 
                potentialEnergy=True, temperature=True, totalEnergy=True)) 

        if loglevel == "debug": 
            simulation.reporters.append(app.StateDataReporter(sys.stdout, 
                nsteps_per_iteration*10, step=True, time=True, speed=True, 
                potentialEnergy=True, temperature=True, totalEnergy=True)) 
        logger.debug("Running dynamics...")
        simulation.step(int((sim_length / timestep))) 

        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions().value_in_unit(unit.angstrom) 
        app.PDBFile.writeFile(topology, positions, open(pdb_filename, 'w')) 
        
    
    logger.info("Processing targets...") 
    for target in targets: 
        if (process_only_these_targets is not None) and (target.id not in process_only_these_targets):
            logger.info('Skipping because %s is not in process_only_these_targets' % target.id)
            logger.info(process_only_these_targets)
            continue
        if mpistate.rank == 0: 
            logger.info('Processing %s' % target.id) 
        
        models_target_dir = os.path.join(models_dir, target.id) 
        simulations_target_dir = os.path.join(simulations_dir, target.id) 
        utils.create_dir(simulations_target_dir) 
        
        if mpistate.rank == 0:
            target_starttime = datetime.datetime.now() 
            logger.info("Starting %s at %s" % (target.id, str(target_starttime)))
            if not os.path.exists(models_target_dir):
                logger.info('%s does not exist, skipping' % models_target_dir)
                continue 
                
        mpistate.comm.Barrier()
        model_pdbs = open(os.path.join(models_target_dir, 'unique-models.txt'), 'r').read().split() 
        
        n_model_pdb = len(model_pdbs) 
        
        for model_index in range(mpistate.rank, n_model_pdb, mpistate.size): 
            gpuid = model_index % gpupn 
            simulation_model_dir = os.path.join(simulations_target_dir, model_pdbs[model_index][19:-4].replace('/', '_')) 
            utils.create_dir(simulation_model_dir) 
            model_file = os.path.join(simulation_model_dir, 'model.pdb') 
            if os.path.exists(model_pdbs[model_index]): 
                shutil.copy2(model_pdbs[model_index], model_file) 
            else: 
                logger.debug("%s not found: target %s rank %d gpuid %d" % (model_pdbs[model_index], target.id, mpistate.rank, gpuid))
            
            pdb_filename = os.path.join(simulation_model_dir, 'implicit-refined.pdb') 
            if os.path.exists(pdb_filename): 
                logger.info("%s already exists, continuing..." % pdb_filename) 
                continue 
            logger.info("-------------------------------------------------------------------------")
            logger.info("Simulating %s from %s in implicit solvent for %.1f ps (MPI rank: %d, GPU ID: %d)" % (target.id, model_pdbs[model_index][19:-4].replace('/', '_'), sim_length/unit.picoseconds, mpistate.rank, gpuid))
            logger.info("-------------------------------------------------------------------------") 
            
            simulate_implicit_md()
            # print(model_pdbs[model_index], simulation_model_dir) 
    
    mpistate.comm.Barrier() 
    if mpistate.rank == 0:
        logger.info('Done.')
         
                
def auto_select_openmm_platform(available_platform_names=None):
    if available_platform_names is None:
        available_platform_names = ['CUDA', 'OpenCL', 'CPU', 'Reference']
    for platform_name in available_platform_names:
        try:
            platform = openmm.Platform.getPlatformByName(platform_name)
            if type(platform) == openmm.Platform:
                logger.info('Auto-selected OpenMM platform: %s' % platform_name)
                return platform_name
        except Exception:
            continue
    raise Exception('No OpenMM platform found') 
    
    
   
def refine_explicit_md(
        openmm_platform=None, gpupn=1, process_only_these_targets=None,
        write_trajectory=False,
        ff='amber99sbildn',
        water_model='tip3p', 
        padding = 10.0 * unit.angstroms, 
        nonbondedMethod = app.PME, # nonbonded method
        cutoff = 0.9*unit.nanometers, # nonbonded cutoff
        constraints = app.HBonds, # bond constrains
        rigidWater = True, # rigid water 
        ionic_strength = 0.0 * unit.molar, # ionic_strength
        removeCMMotion = False, # remove center-of-mass motion
        sim_length=1000.0 * unit.picoseconds,
        timestep=2.0 * unit.femtoseconds,   # timestep
        temperature=300.0 * unit.kelvin,   # simulation temperature
        pressure=1.0 * unit.atmospheres,   # simulation pressure
        collision_rate=20.0 / unit.picoseconds,   # Langevin collision rate
#         barostat_period=50,
        minimization_tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        minimization_steps=20,
        nsteps_per_iteration=500,
        write_solvated_model=False,
        cpu_platform_threads=1,
        retry_failed_runs=False,
        serialize_at_start_of_each_sim=False, 
        loglevel = None):

    """
    Run MD simulation in explicit solvent environment. 

    MPI-enabled. 
    """ 
    utils.set_loglevel(loglevel)  
    gpuid = mpistate.rank % gpupn 
    
    simulations_dir = os.path.abspath(core.default_project_dirnames.simulations)

    targets, templates_resolved_seq = core.get_targets_and_templates() 
    
    if not openmm_platform:
        openmm_platform = auto_select_openmm_platform()
        
    if openmm_platform == 'CPU':
        platform_properties = {'CpuThreads': str(cpu_platform_threads)}
    else:
        platform_properties = {}
        
    ff_files = [ff+'.xml', water_model+'.xml']
    forcefield = app.ForceField(*ff_files)

    def simulate_explicit_md(): 
        logger.debug("Reading model...")
        pdb = app.PDBFile(model_file)
        # Set up Platform 
        platform = openmm.Platform.getPlatformByName(openmm_platform)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            if openmm_platform == 'CUDA': 
                properties = {'DeviceIndex': str(gpuid), 'CudaPrecision': 'mixed'} 
            if openmm_platform == 'OpenCL': 
                properties = {'DeviceIndex': str(gpuid)}

        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addSolvent(forcefield, model='tip3p', padding=padding, ionicStrength=ionic_strength) 
        topology = modeller.getTopology()
        positions = modeller.getPositions() 
        app.PDBFile.writeFile(topology, positions, open(pdb_filename_solvated, 'w')) 

        logger.debug("Constructing System object...")
        system = forcefield.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=cutoff, constraints=app.HBonds)
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        system.addForce(barostat)

        logger.debug("Creating Context...") 
        integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
        #         context = openmm.Context(system, integrator, platform, platform_properties)
        simulation = app.Simulation(topology, system, integrator, platform, properties)
        simulation.context.setPositions(positions) 

        logger.debug("Minimizing structure...")
        simulation.minimizeEnergy(tolerance=minimization_tolerance, maxIterations=minimization_steps)

        if write_trajectory: 
            trajectory_filename = os.path.join(simulation_model_dir, 'explicit-trajectory.dcd')
            simulation.reporters.append(app.DCDReporter(trajectory_filename, nsteps_per_iteration)) 
        energy_log_filename = os.path.join(simulation_model_dir, 'explicit-energies.txt') 
        simulation.reporters.append(app.StateDataReporter(energy_log_filename, 
                nsteps_per_iteration, step=True, time=True, speed=True, 
                potentialEnergy=True, temperature=True, totalEnergy=True)) 

        if loglevel == "debug": 
            simulation.reporters.append(app.StateDataReporter(sys.stdout, 
                nsteps_per_iteration*100, step=True, time=True, speed=True, 
                potentialEnergy=True, temperature=True, totalEnergy=True)) 
        logger.debug("Running dynamics...")
        simulation.step(int((sim_length / timestep))) 

        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions().value_in_unit(unit.angstrom) 
        app.PDBFile.writeFile(topology, positions, open(pdb_filename, 'w')) 
        
    
    logger.info("Processing targets...") 
    for target in targets: 
        if (process_only_these_targets is not None) and (target.id not in process_only_these_targets):
            logger.info('Skipping because %s is not in process_only_these_targets' % target.id)
            logger.info(process_only_these_targets)
            continue
        if mpistate.rank == 0: 
            logger.info('Processing %s' % target.id) 
        
        simulations_target_dir = os.path.join(simulations_dir, target.id) 
        
        if mpistate.rank == 0:
            target_starttime = datetime.datetime.now() 
            logger.info("Starting %s at %s" % (target.id, str(target_starttime)))
                
        mpistate.comm.Barrier()
        
        simulations_target_model_dirs = glob.glob(os.path.join(simulations_target_dir, target.id + '*')) 
        n_model_pdb = len(simulations_target_model_dirs) 
        
        for model_index in range(mpistate.rank, n_model_pdb, mpistate.size): 
            gpuid = model_index % gpupn 
            simulation_model_dir = simulations_target_model_dirs[model_index]
            model_file = os.path.join(simulation_model_dir, 'implicit-refined.pdb') 
            if not os.path.exists(model_file):  
                logger.debug("%s not found: target %s rank %d gpuid %d" % (model_file, target.id, mpistate.rank, gpuid))
            
            pdb_filename_solvated = os.path.join(simulation_model_dir, 'implicit-solvated.pdb')
            pdb_filename = os.path.join(simulation_model_dir, 'explicit-refined.pdb') 
            if os.path.exists(pdb_filename): 
                logger.info("%s already exists, continuing..." % pdb_filename) 
                continue 
            logger.info("-------------------------------------------------------------------------")
            logger.info("Simulating %s from %s in implicit solvent for %.1f ps (MPI rank: %d, GPU ID: %d)" % (target.id, model_file, sim_length/unit.picoseconds, mpistate.rank, gpuid))
            logger.info("-------------------------------------------------------------------------") 
            
            simulate_explicit_md()
    
    mpistate.comm.Barrier() 
    if mpistate.rank == 0:
        logger.info('Done.')
    