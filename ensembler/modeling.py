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
from collections import namedtuple
import mdtraj

import Bio
import Bio.SeqIO
import Bio.pairwise2
import Bio.SubsMat.MatrixInfo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from . import core, utils 
from .utils import notify_when_done 
from .core import mpistate, logger 
import subprocess
loopmodel_subprocess_kwargs = {'timeout': 10800}   # 3 hour timeout - used for loopmodel call
    
TargetSetupData = namedtuple(
    'TargetSetupData',
    ['target_starttime', 'models_target_dir']
)

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



@utils.notify_when_done
def align_targets_and_templates(process_only_these_targets=None,
                                process_only_these_templates=None,
                                substitution_matrix='gonnet',
                                gap_open=-10,
                                gap_extend=-0.5,
                                loglevel=None
                                ):
    """
    Conducts pairwise alignments of target sequences against template sequences.
    Stores Modeller-compatible 'alignment.pir' files in each model directory,
    and also outputs a table of model IDs, sorted by sequence identity.

    Parameters
    ----------
    process_only_these_targets:
    process_only_these_templates:
    substitution_matrix: str
        Specify an amino acid substitution matrix available from Bio.SubsMat.MatrixInfo
    """
    utils.set_loglevel(loglevel)
    targets, templates_resolved_seq = core.get_targets_and_templates()
    ntemplates = len(templates_resolved_seq)
    nselected_templates = len(process_only_these_templates) if process_only_these_templates else ntemplates
    for target in targets:
        if process_only_these_targets and target.id not in process_only_these_targets: continue

        if mpistate.rank == 0:
            logger.info('Working on target %s...' % target.id)

        models_target_dir = os.path.join(core.default_project_dirnames.models, target.id)
        utils.create_dir(models_target_dir)

        seq_identity_data_sublist = []

        for template_index in range(mpistate.rank, ntemplates, mpistate.size):
            template_id = templates_resolved_seq[template_index].id
            if os.path.exists(os.path.join(core.default_project_dirnames.templates_structures_modeled_loops, template_id + '.pdb')):
                remodeled_seq_filepath = os.path.join(core.default_project_dirnames.templates_structures_modeled_loops, template_id + '-pdbfixed.fasta')
                template = list(Bio.SeqIO.parse(remodeled_seq_filepath, 'fasta'))[0]
            else:
                template = templates_resolved_seq[template_index]

            if process_only_these_templates and template_id not in process_only_these_templates: continue

            model_dir = os.path.abspath(os.path.join(core.default_project_dirnames.models, target.id, template_id))
            utils.create_dir(model_dir)
            aln = align_target_template(
                target,
                template,
                substitution_matrix=substitution_matrix,
                gap_open=gap_open,
                gap_extend=gap_extend
            )
#             aln_filepath = os.path.join(model_dir, 'alignment.pir')
#             write_modeller_pir_aln_file(aln, target, template, pir_aln_filepath=aln_filepath)
            aln_filepath = os.path.abspath(os.path.join(model_dir, 'alignment.ros'))
            write_rosetta_grishin_aln_file(aln, target, template, pir_aln_filepath=aln_filepath)
            seq_identity_data_sublist.append({
                'templateid': template_id,
                'seq_identity': calculate_seq_identity(aln),
            })

        seq_identity_data_gathered = mpistate.comm.gather(seq_identity_data_sublist, root=0)

        seq_identity_data = []
        if mpistate.rank == 0:
            seq_identity_data = [None] * nselected_templates
            for i in range(nselected_templates):
                seq_identity_data[i] = seq_identity_data_gathered[i % mpistate.size][i // mpistate.size]

        seq_identity_data = mpistate.comm.bcast(seq_identity_data, root=0)

        seq_identity_data = sorted(seq_identity_data, key=lambda x: x['seq_identity'], reverse=True)
        write_sorted_seq_identities(target, seq_identity_data)


def align_target_template(target,
                          template,
                          substitution_matrix='gonnet',
                          gap_open=-10,
                          gap_extend=-0.5
                          ):
    """
    Parameters
    ----------
    target: BioPython SeqRecord
    template: BioPython SeqRecord
    substitution_matrix: str
        Specify an amino acid substitution matrix available from Bio.SubsMat.MatrixInfo
    gap_open: float or int
    gap_extend: float or int

    Returns
    -------
    alignment: list
    """
    matrix = getattr(Bio.SubsMat.MatrixInfo, substitution_matrix)
    aln = Bio.pairwise2.align.globalds(str(target.seq), str(template.seq), matrix, gap_open, gap_extend)
    return aln


def calculate_seq_identity(aln):
    len_shorter_seq = min([len(aln[0][0].replace('-', '')), len(aln[0][1].replace('-', ''))])
    seq_id = 0
    for r in range(len(aln[0][0])):
        if aln[0][0][r] == aln[0][1][r]:
            seq_id += 1
    seq_id = 100 * float(seq_id) / float(len_shorter_seq)
    return seq_id


def write_sorted_seq_identities(target, seq_identity_data):
    seq_identity_file_str = ''
    for seq_identity_dict in seq_identity_data:
        seq_identity_file_str += '%-30s %.1f\n' % (seq_identity_dict['templateid'], seq_identity_dict['seq_identity'])
    seq_identity_filepath = os.path.join(core.default_project_dirnames.models, target.id, 'sequence-identities.txt')
    with open(seq_identity_filepath, 'w') as seq_identity_file:
        seq_identity_file.write(seq_identity_file_str)


def write_rosetta_grishin_aln_file(aln, target, template, pir_aln_filepath='alignment.ros'):
    contents = "## {0} {1} \n# \nscores_from_program: 0 \n".format(target.id, template.id)
    contents += '0 ' + aln[0][0] + '\n' + '0 ' + aln[0][1] + '\n--\n' 
    with open(pir_aln_filepath, 'w') as outfile:
        outfile.write(contents)


@utils.notify_when_done
def build_models(process_only_these_targets=None, process_only_these_templates=None,
                 model_seqid_cutoff=None, write_modeller_restraints_file=False, 
                 number_of_models=1, loglevel=None):
    """Uses the build_model method to build homology models for a given set of
    targets and templates.

    MPI-enabled.
    """
    # Note that this code uses an os.chdir call to switch into a temp directory before running Modeller.
    # This is because Modeller writes various output files in the current directory, and there is NO WAY
    # to define where these files are written, other than to chdir beforehand. If running this routine
    # in parallel, it is likely that occasional exceptions will occur, due to concurrent processes
    # making os.chdir calls.
    utils.set_loglevel(loglevel)
    targets, templates_resolved_seq = core.get_targets_and_templates()

    if process_only_these_templates:
        selected_template_indices = [i for i, seq in enumerate(templates_resolved_seq) if seq.id in process_only_these_templates]
    else:
        selected_template_indices = range(len(templates_resolved_seq))

    for target in targets:
        if process_only_these_targets and target.id not in process_only_these_targets: continue
        target_setup_data = build_models_target_setup(target)

        if model_seqid_cutoff:
            process_only_these_templates = core.select_templates_by_seqid_cutoff(target.id, seqid_cutoff=model_seqid_cutoff)
            selected_template_indices = [i for i, seq in enumerate(templates_resolved_seq) if seq.id in process_only_these_templates]

        ntemplates_selected = len(selected_template_indices)

        for template_index in range(mpistate.rank, ntemplates_selected, mpistate.size):
            template_resolved_seq = templates_resolved_seq[selected_template_indices[template_index]]
            if process_only_these_templates and template_resolved_seq.id not in process_only_these_templates: continue
            build_model(target, template_resolved_seq, target_setup_data,
                        write_modeller_restraints_file=write_modeller_restraints_file, 
                        number_of_models=number_of_models, 
                        loglevel=loglevel)

def build_model(target, template_resolved_seq, target_setup_data,
                write_modeller_restraints_file=False, number_of_models=1, 
		loglevel=None):
    """Uses Rosetta to build a homology model for a given target and
    template.

    Will not run Rosetta if the output files already exist.

    Parameters
    ----------
    target : BioPython SeqRecord
    template_resolved_seq : BioPython SeqRecord
        Must be a corresponding .pdb template file with the same ID in the
        templates/structures directory.
    template_resolved_seq : BioPython SeqRecord
        Must be a corresponding .pdb template file with the same ID in the
        templates/structures directory.
    target_setup_data : TargetSetupData obj
    write_modeller_restraints_file : bool
        Write file containing restraints used by Modeller - note that this file can be relatively
        large, e.g. ~300KB per model for a protein kinase domain target.
    loglevel : bool
    """
    utils.set_loglevel(loglevel)

    template_structure_dir = os.path.abspath(
        core.default_project_dirnames.templates_structures_modeled_loops
    )

    if os.path.exists(os.path.join(template_structure_dir, template_resolved_seq.id + '.pdb')):
        remodeled_seq_filepath = os.path.join(
            core.default_project_dirnames.templates_structures_modeled_loops,
            template_resolved_seq.id + '-pdbfixed.fasta'
        )
        template = list(Bio.SeqIO.parse(remodeled_seq_filepath, 'fasta'))[0]
    else:
        template = template_resolved_seq
        template_structure_dir = os.path.abspath(
            core.default_project_dirnames.templates_structures_resolved
        )

    model_dir = os.path.abspath(os.path.join(target_setup_data.models_target_dir, template.id))
    if not os.path.exists(model_dir):
        utils.create_dir(model_dir)
    model_pdbfilepath = os.path.abspath(os.path.join(model_dir, 'model.pdb.gz'))
    modeling_log_filepath = os.path.abspath(os.path.join(model_dir, 'modeling-log.yaml'))

    check_model_pdbfilepath_ends_in_pdbgz(model_pdbfilepath)
    model_pdbfilepath_uncompressed = model_pdbfilepath[:-3]

    if check_all_model_files_present(model_dir):
        logger.debug(
            "Output files already exist for target '%s' // template '%s'; files were not overwritten." %
            (target.id, template.id)
        )
        return

    logger.info(
        '-------------------------------------------------------------------------\n'
        'Modelling "%s" => "%s"\n' 
        '   Generating %d model(s)\n'
        '-------------------------------------------------------------------------'
        % (target.id, template.id, number_of_models) 
    )

    aln_filepath = os.path.abspath(os.path.join(model_dir, 'alignment.ros'))
    run_rosettaCM(target, template, model_dir, model_pdbfilepath, model_pdbfilepath_uncompressed,
                             template_structure_dir, number_of_models = number_of_models)
    start = datetime.datetime.utcnow()


def build_models_target_setup(target):
    target_setup_data = None
    if mpistate.rank == 0:
        models_target_dir = os.path.join(core.default_project_dirnames.models, target.id)
        target_starttime = datetime.datetime.utcnow()
        logger.info(
            '=========================================================================\n'
            'Working on target "%s"\n'
            '========================================================================='
            % target.id
        )
        target_setup_data = TargetSetupData(
            target_starttime=target_starttime,
            models_target_dir=models_target_dir
        )
    target_setup_data = mpistate.comm.bcast(target_setup_data, root=0)
    return target_setup_data

def check_model_pdbfilepath_ends_in_pdbgz(model_pdbfilepath):
    if model_pdbfilepath[-7:] != '.pdb.gz':
        raise Exception('model_pdbfilepath (%s) must end in .pdb.gz' % model_pdbfilepath)


def check_all_model_files_present(model_dir):
    seqid_filepath = os.path.abspath(os.path.join(model_dir, 'sequence-identity.txt'))
    model_pdbfilepath = os.path.abspath(os.path.join(model_dir, 'model.pdb.gz'))
    aln_filepath = os.path.abspath(os.path.join(model_dir, 'alignment.pir'))
    files_to_check = [model_pdbfilepath, seqid_filepath, aln_filepath]
    files_present = [os.path.exists(filename) for filename in files_to_check]
    return all(files_present)

def run_rosettaCM(target, template, model_dir, model_pdbfilepath, model_pdbfilepath_uncompressed,
                 template_structure_dir, number_of_models = 1):
    rosetta_output = open(os.path.abspath(os.path.join(model_dir, 'rosetta_cm.out')), 'w')
    partial_thread_excutable = core.find_partial_thread_executable()
    target_fasta_filepath = os.path.abspath(os.path.join(core.default_project_dirnames.targets, 'targets.fa'))
    aln_filepath = os.path.abspath(os.path.join(model_dir, 'alignment.ros'))
    minirosetta_database_path = os.environ.get('MINIROSETTA_DATABASE')
    template_filepath = os.path.abspath(os.path.join(template_structure_dir, template.id+'.pdb'))
    cwd = os.getcwd()
    os.chdir(model_dir)
    command = "%s -database %s -mute all -in:file:fasta %s -in:file:alignment %s -in:file:template_pdb %s -ignore_unrecognized_res"%(partial_thread_excutable, minirosetta_database_path, target_fasta_filepath, aln_filepath, template_filepath)
    rosetta_output.write(command + '\n')
    partial_threading = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    partial_threading.wait()
    thread_filepath = os.path.abspath(os.path.join(model_dir, template.id+'.pdb'))
    if os.path.exists(thread_filepath):
        thread_fullnames = [ thread_filepath ]
    else:
        logger.error('Recheck the partial threading process in Rosetta.')
    rosetta_output.write('done partial threading.\n')
    thread_fullnames = [ thread_filepath ]
    flag_fn = os.path.abspath(os.path.join(model_dir, 'flags'))
    xml_fn = os.path.abspath(os.path.join(model_dir, 'rosetta_cm.xml'))
    silent_out = os.path.abspath(os.path.join(model_dir, 'rosetta_cm.silent'))
    write_rosettaCM_flags(flag_fn, target_fasta_filepath, xml_fn, silent_out)
    write_resettaCM_xml(xml_fn, thread_fullnames)
    rosetta_script_excutable = core.find_rosetta_scripts_executable()
# -nstruct controls how many output structures 
    command="%s @%s -database %s -nstruct %d"%(rosetta_script_excutable, flag_fn, minirosetta_database_path, number_of_models)
    rosetta_output.write(command + '\n')
    rosetta_script = subprocess.Popen(command, stdout=rosetta_output, stderr=subprocess.STDOUT, shell=True)
#    for line in iter(rosetta_script.stdout.readline, b''): 
#        print(">>> " + line.rstrip())
    rosetta_script.wait()
    rosetta_output.write('\ndone rosetta scripts--hybridize mover.\n')
    rosetta_model_output = os.path.join(model_dir, 'S_0001.pdb')
    if os.path.exists(rosetta_model_output):
        model_pdbfilepath_uncompressed = os.path.join(model_dir, 'model.pdb')
        traj = mdtraj.load(rosetta_model_output)
        selection_noH = traj.topology.select('not element H')
        traj_noH = traj.atom_slice(selection_noH)
        traj_noH.save_pdb(model_pdbfilepath_uncompressed)
        model_pdbfilepath_compressed = os.path.join(model_dir, 'model.pdb.gz')
        with open(model_pdbfilepath_uncompressed) as model_pdbfile:
            with gzip.open(model_pdbfilepath, 'w') as model_pdbfilegz:
                model_pdbfilegz.write(model_pdbfile.read().encode())
    else:
        warnings.warn('Job failed to generate pdb for %s template, check your log file. '% template.id)
    os.chdir(cwd)


def write_rosettaCM_flags(flag_fn, fasta_fn, xml_fn, silent_fn):
    flag_file=open(flag_fn,'w')
    flag_file.write("# i/o\n")
    flag_file.write("-in:file:fasta %s\n"%fasta_fn)
    flag_file.write("-nstruct 20\n")
    flag_file.write("-parser:protocol %s\n\n"%xml_fn)
    flag_file.write("# relax options\n")
    flag_file.write("-relax:dualspace\n")
    flag_file.write("-relax:jump_move true\n")
    flag_file.write("-default_max_cycles 200\n")
    flag_file.write("-beta_cart\n")
    flag_file.write("-hybridize:stage1_probability 1.0\n")

def write_resettaCM_xml(fn, template_filenames):
    xml_file=open(fn,'w')
    xml_file.write("<ROSETTASCRIPTS>\n")
    xml_file.write("    <TASKOPERATIONS>\n")
    xml_file.write("    </TASKOPERATIONS>\n")
    xml_file.write("    <SCOREFXNS>\n")
    xml_file.write("        <ScoreFunction name=\"stage1\" weights=\"score3\" symmetric=\"0\">\n")
    xml_file.write("            <Reweight scoretype=\"atom_pair_constraint\" weight=\"0.1\"/>\n")
    xml_file.write("        </ScoreFunction>\n")
    xml_file.write("        <ScoreFunction name=\"stage2\" weights=\"score4_smooth_cart\" symmetric=\"0\">\n")
    xml_file.write("            <Reweight scoretype=\"atom_pair_constraint\" weight=\"0.1\"/>\n")
    xml_file.write("        </ScoreFunction>\n")
    xml_file.write("        <ScoreFunction name=\"fullatom\" weights=\"beta_cart\" symmetric=\"0\">\n")
    xml_file.write("            <Reweight scoretype=\"atom_pair_constraint\" weight=\"0.1\"/>\n")
    xml_file.write("        </ScoreFunction>\n")
    xml_file.write("    </SCOREFXNS>\n")
    xml_file.write("    <FILTERS>\n")
    xml_file.write("    </FILTERS>\n")
    xml_file.write("    <MOVERS>\n")
    xml_file.write("        <Hybridize name=\"hybridize\" stage1_scorefxn=\"stage1\" stage2_scorefxn=\"stage2\" fa_scorefxn=\"fullatom\" batch=\"1\" stage1_increase_cycles=\"1.0\" stage2_increase_cycles=\"1.0\">\n")
    for tmpl in template_filenames:
        xml_file.write("            <Template pdb=\"%s\" cst_file=\"AUTO\" weight=\"1.0\" />\n"%(tmpl))
    xml_file.write("        </Hybridize>\n")
    xml_file.write("    </MOVERS>\n")
    xml_file.write("    <APPLY_TO_POSE>\n")
    xml_file.write("    </APPLY_TO_POSE>\n")
    xml_file.write("    <PROTOCOLS>\n")
    xml_file.write("        <Add mover=\"hybridize\"/>\n")
    xml_file.write("    </PROTOCOLS>\n")
    xml_file.write("</ROSETTASCRIPTS>\n")
    xml_file.close()
