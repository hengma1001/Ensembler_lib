import sys
import os
import re
import datetime
import shutil
import gzip
import glob
import warnings
import traceback 
import tempfile

import Bio
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from . import core, utils 
from .utils import notify_when_done 
from .core import mpistate, logger 
import subprocess
loopmodel_subprocess_kwargs = {'timeout': 10800}   # 3 hour timeout - used for loopmodel call
    
class LoopmodelOutput:
    def __init__(self, output_text=None, loopmodel_exception=None, exception=None, trbk=None, successful=False, no_missing_residues=False):
        self.output_text = output_text
        self.exception = exception
        self.loopmodel_exception = loopmodel_exception
        self.traceback = trbk
        self.successful = successful
        self.no_missing_residues = no_missing_residues

    
@notify_when_done
def model_template_loops(process_only_these_templates=None, overwrite_structures=False, loglevel=None):
    """
    Use Rosetta loopmodel to model missing loops in template structures.
    Completed templates are stored in templates/structures-modeled-loops

    :param process_only_these_templates: list of str
    :param loglevel: str
    :return:
    """
    utils.set_loglevel(loglevel) 
    targets, templates_resolved_seq = core.get_targets_and_templates()
    templates_full_seq = core.get_templates_full_seq()
    missing_residues_list = pdbfix_templates(templates_full_seq, process_only_these_templates=process_only_these_templates, overwrite_structures=overwrite_structures)
    loopmodel_templates(templates_resolved_seq, missing_residues_list, process_only_these_templates=process_only_these_templates, overwrite_structures=overwrite_structures) 
    
    
def pdbfix_templates(templates_full_seq, process_only_these_templates=None, overwrite_structures=False):
    """
    Parameters
    ----------
    templates_full_seq: list of BioPython SeqRecord
        full UniProt sequence for span of the template (including unresolved residues)
    process_only_these_templates: list of str
    overwrite_structures: bool
    Returns
    -------
    missing_residues_list: list of list of OpenMM Residue
    """
    missing_residues_sublist = []
    ntemplates = len(templates_full_seq)
    for template_index in range(mpistate.rank, ntemplates, mpistate.size):
        template_full_seq = templates_full_seq[template_index]
        if process_only_these_templates and template_full_seq.id not in process_only_these_templates:
            missing_residues_sublist.append(None)
            continue
        missing_residues_sublist.append(pdbfix_template(template_full_seq, overwrite_structures=overwrite_structures))

    missing_residues_gathered = mpistate.comm.gather(missing_residues_sublist, root=0)

    missing_residues_list = []
    if mpistate.rank == 0:
        missing_residues_list = [None] * ntemplates
        for template_index in range(ntemplates):
            missing_residues_list[template_index] = missing_residues_gathered[template_index % mpistate.size][template_index // mpistate.size]

    missing_residues_list = mpistate.comm.bcast(missing_residues_list, root=0)

    return missing_residues_list 


def pdbfix_template(template_full_seq, overwrite_structures=False):
    """
    Parameters
    ----------
    template_full_seq: BioPython SeqRecord
        full UniProt sequence for span of the template (including unresolved residues)
    overwrite_structures: bool
    Returns
    -------
    fixer.missingResidues
    """
    try:
        template_pdbfixed_filepath = os.path.join(
            core.default_project_dirnames.templates_structures_modeled_loops,
            template_full_seq.id + '-pdbfixed.pdb'
        )
        seq_pdbfixed_filepath = os.path.join(
            core.default_project_dirnames.templates_structures_modeled_loops,
            template_full_seq.id + '-pdbfixed.fasta'
        )
        import pdbfixer
        import simtk.openmm.app
        template_filepath = os.path.join(
            core.default_project_dirnames.templates_structures_resolved,
            template_full_seq.id + '.pdb'
        )
        fixer = pdbfixer.PDBFixer(filename=template_filepath)
        chainid = next(fixer.topology.chains()).id
        sequence = [ Bio.SeqUtils.seq3(r).upper() for r in template_full_seq.seq ]
        seq_obj = pdbfixer.pdbfixer.Sequence(chainid, sequence)
        fixer.sequences.append(seq_obj)
        fixer.findMissingResidues()
        remove_missing_residues_at_termini(fixer, len_full_seq=len(template_full_seq.seq))
        if not overwrite_structures and os.path.exists(template_pdbfixed_filepath):
            return fixer.missingResidues
        fixer.findMissingAtoms()
        (newTopology, newPositions, newAtoms, existingAtomMap) = fixer._addAtomsToTopology(True, True)
        fixer.topology = newTopology
        fixer.positions = newPositions
        with open(template_pdbfixed_filepath, 'w') as template_pdbfixed_file:
            simtk.openmm.app.PDBFile.writeFile(
                fixer.topology, fixer.positions, file=template_pdbfixed_file
            )

        # Write sequence to file
        seq_pdbfixed = ''.join([Bio.SeqUtils.seq1(r.name) for r in fixer.topology.residues()])
        seq_record_pdbfixed = SeqRecord(Seq(seq_pdbfixed), id=template_full_seq.id, description=template_full_seq.id)
        Bio.SeqIO.write([seq_record_pdbfixed], seq_pdbfixed_filepath, 'fasta')

        return fixer.missingResidues
    except (KeyboardInterrupt, ImportError):
        raise
    except Exception as e:
        trbk = traceback.format_exc()
#         log_filepath = os.path.abspath(os.path.join(
#             core.default_project_dirnames.templates_structures_modeled_loops,
#             template_full_seq.id + '-pdbfixer-log.yaml'
#         ))
#         logfile = core.LogFile(log_filepath)
#         logfile.log({
#             'templateid': str(template_full_seq.id),
#             'exception': e,
#             'traceback': core.literal_str(trbk),
#             'mpi_rank': mpistate.rank,
#         })
        logger.error(
            'MPI rank %d pdbfixer error for template %s - see logfile' %
            (mpistate.rank, template_full_seq.id)
        )
        logger.debug(e)
        logger.debug(trbk)

        
def remove_missing_residues_at_termini(fixer, len_full_seq):
    # remove C-terminal missing residues
    if len(fixer.missingResidues) == 0:
        return None
    sorted_missing_residues_keys = sorted(fixer.missingResidues, key=lambda x: x[1])
    last_missing_residues_key = sorted_missing_residues_keys[-1]
    last_missing_residues_start_index = last_missing_residues_key[1]
    last_missing_residues = fixer.missingResidues[last_missing_residues_key]
    nmissing_residues_up_to_last = sum([len(fixer.missingResidues[key]) for key in sorted_missing_residues_keys[:-1]])

    if last_missing_residues_start_index + nmissing_residues_up_to_last + len(last_missing_residues) == len_full_seq:
        fixer.missingResidues.pop(last_missing_residues_key)

    # remove N-terminal missing residues
    fixer.missingResidues.pop((0, 0), None)

    
def loopmodel_templates(templates, missing_residues, process_only_these_templates=None, overwrite_structures=False):
    """
    Parameters
    ----------
    templates:  list of BioPython SeqRecord
        only the id is used
    missing_residues: list of list of OpenMM Residue
    process_only_these_templates: bool
    overwrite_structures: bool
    """
    for template_index in range(mpistate.rank, len(templates), mpistate.size):
        template = templates[template_index]
        if process_only_these_templates and template.id not in process_only_these_templates:
            continue
        if mpistate.size > 1:
            logger.info('MPI rank %d modeling missing loops for template %s' % (mpistate.rank, template.id))
        else:
            logger.info('Modeling missing loops for template %s' % template.id)
        loopmodel_template(template, missing_residues[template_index], overwrite_structures=overwrite_structures)
        

def loopmodel_template(template, missing_residues, overwrite_structures=False):
    template_filepath = os.path.abspath(os.path.join(core.default_project_dirnames.templates_structures_modeled_loops, template.id + '-pdbfixed.pdb'))
    output_pdb_filepath = os.path.abspath(os.path.join(core.default_project_dirnames.templates_structures_modeled_loops, template.id + '.pdb'))
    loop_filepath = os.path.abspath(os.path.join(core.default_project_dirnames.templates_structures_modeled_loops, template.id + '.loop'))
    output_score_filepath = os.path.abspath(os.path.join(core.default_project_dirnames.templates_structures_modeled_loops, template.id + '-loopmodel-score.sc'))
    log_filepath = os.path.abspath(os.path.join(core.default_project_dirnames.templates_structures_modeled_loops, template.id + '-loopmodel-log.yaml'))
    if not overwrite_structures:
        if os.path.exists(log_filepath):
            return
#     logfile = core.LogFile(log_filepath)
    write_loop_file(template, missing_residues)
    starttime = datetime.datetime.utcnow()
    if len(missing_residues) == 0:
        loopmodel_output = LoopmodelOutput(successful=True, no_missing_residues=True)
    else:
        loopmodel_output = run_loopmodel(template_filepath, loop_filepath, output_pdb_filepath, output_score_filepath)
    if not loopmodel_output.successful:
        logger.error('MPI rank %d Loopmodel error for template %s - see logfile' % (mpistate.rank, template.id))
    timedelta = datetime.datetime.utcnow() - starttime
#     logfile.log({
#         'templateid': str(template.id),
#         'no_missing_residues': loopmodel_output.no_missing_residues,
#         'loopmodel_output': loopmodel_output.output_text,
#         'mpi_rank': mpistate.rank,
#         'successful': loopmodel_output.successful,
#         'exception': loopmodel_output.exception,
#         'loopmodel_exception': loopmodel_output.loopmodel_exception,
#         'traceback': loopmodel_output.traceback,
#         'timing': core.strf_timedelta(timedelta),
#         })


def write_loop_file(template, missing_residues):
    loop_file_text = ''
    loop_residues_added = 0
    loop_residues_data = [(key[1], len(residues)) for key, residues in missing_residues.items()]
    loop_residues_data = sorted(loop_residues_data, key=lambda x: x[0])
    for loop_residue_data in loop_residues_data:
        residue_number, nresidues = loop_residue_data
        loop_begin = residue_number + loop_residues_added   # 1-based, one residue before the loop
        loop_end = residue_number + nresidues + loop_residues_added + 1   # 1-based, one residue after the loop
        loop_residues_added += nresidues
        # Note that missing residues at termini (which cannot be modeled by Rosetta loopmodel) have already been removed from the PDBFixer.missingResidues dictionary
        loop_file_text += 'LOOP {0} {1} - - 1\n'.format(loop_begin, loop_end)
    loop_filepath = os.path.join(core.default_project_dirnames.templates_structures_modeled_loops, template.id + '.loop')
    with open(loop_filepath, 'w') as loop_file:
        loop_file.write(loop_file_text)


def run_loopmodel(input_template_pdb_filepath, loop_filepath, output_pdb_filepath, output_score_filepath, loopmodel_executable_filepath=None, nmodels_to_build=1):
    if loopmodel_executable_filepath is None:
        loopmodel_executable_filepath = core.find_loopmodel_executable()

    temp_dir = tempfile.mkdtemp()
    temp_template_filepath = os.path.join(temp_dir, 'template.pdb')
    temp_loop_filepath = os.path.join(temp_dir, 'template.loop')
    temp_output_model_filepath = os.path.join(temp_dir, 'template_0001.pdb')
    temp_output_score_filepath = os.path.join(temp_dir, 'score.sc')
    minirosetta_database_path = os.environ.get('MINIROSETTA_DATABASE')
    shutil.copy(input_template_pdb_filepath, temp_template_filepath)
    shutil.copy(loop_filepath, temp_loop_filepath)
    try:
        output_text = subprocess.check_output(
            [
                loopmodel_executable_filepath,
                '-database', minirosetta_database_path,
                '-in::file::s', temp_template_filepath,
                '-loops:loop_file', temp_loop_filepath,
                '-out:path:all', temp_dir,
                '-loops:remodel', 'perturb_kic',
                '-loops:refine', 'refine_kic',
                '-ex1',
                '-ex2',
                '-nstruct', '%d' % nmodels_to_build,
                '-loops:max_kic_build_attempts', '100',
                '-in:file:fullatom',
                '-overwrite',
                ],
            stderr=subprocess.STDOUT,
            **loopmodel_subprocess_kwargs
            )
        if os.path.exists(temp_output_model_filepath):
            shutil.copy(temp_output_model_filepath, output_pdb_filepath)
            shutil.copy(temp_output_score_filepath, output_score_filepath)
            shutil.rmtree(temp_dir)
            return LoopmodelOutput(output_text=output_text, successful=True)
        else:
            shutil.rmtree(temp_dir)
            return LoopmodelOutput(output_text=output_text, successful=False)
    except KeyboardInterrupt:
        shutil.rmtree(temp_dir)
        raise
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir)
        return LoopmodelOutput(loopmodel_exception=e.output, trbk=traceback.format_exc(), successful=False)
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(temp_dir)
        return LoopmodelOutput(output_text=e.output, exception=e, trbk=traceback.format_exc(), successful=False)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return LoopmodelOutput(output_text=output_text, exception=e, trbk=traceback.format_exc(), successful=False)

