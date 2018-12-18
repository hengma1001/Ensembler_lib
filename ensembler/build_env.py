import os, sys, re
import Bio.SeqUtils
import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from . import utils, uniprot, core, pdb
from .utils import notify_when_done, file_exists_and_not_empty


structure_type_file_extension_mapper = {'pdb': '.pdb.gz', 'sifts': '.xml.gz'}

class Ensembler_proj_init(object): 
    def __init__(self, project_toplevel_dir='./', run_main=True): 
        self.project_toplevel_dir = project_toplevel_dir
        if run_main: 
            self._init_project()
    
    @notify_when_done
    def _init_project(self): 
        self._create_dir()

    def _create_dir(self): 
        utils.create_dir(os.path.join(self.project_toplevel_dir, core.default_project_dirnames.targets))
        utils.create_dir(os.path.join(self.project_toplevel_dir, core.default_project_dirnames.templates))
        utils.create_dir(os.path.join(self.project_toplevel_dir, core.default_project_dirnames.structures))
        utils.create_dir(os.path.join(self.project_toplevel_dir, core.default_project_dirnames.models))
        utils.create_dir(os.path.join(self.project_toplevel_dir, core.default_project_dirnames.packaged_models))
        utils.create_dir(os.path.join(self.project_toplevel_dir, core.default_project_dirnames.structures_pdb))
        utils.create_dir(os.path.join(self.project_toplevel_dir, core.default_project_dirnames.structures_sifts))
        utils.create_dir(os.path.join(self.project_toplevel_dir, core.default_project_dirnames.templates_structures_resolved))
        utils.create_dir(os.path.join(self.project_toplevel_dir, core.default_project_dirnames.templates_structures_modeled_loops))


class GatherTargetsFromUniprot(): 
    """
    Gather target protein data from Uniprot
    """
    def __init__(self, uniprot_query_string, uniprot_domain_regex=None, save_uniprot_xml=False, loglevel=None, run_main=True): 
        self.uniprot_query_string = uniprot_query_string
        self.uniprot_domain_regex = uniprot_domain_regex
        self._save_uniprot_xml = save_uniprot_xml
        if run_main:
            self._gather_targets()
    
    @notify_when_done
    def _gather_targets(self, write_output_files = True): 
        print('Querying Uniprot web server...')
        get_uniprot_xml_args = {}
        if self._save_uniprot_xml:
            get_uniprot_xml_args['write_to_filepath'] = 'targets-uniprot.xml'

        self.uniprotxml = uniprot.get_uniprot_xml(self.uniprot_query_string, **get_uniprot_xml_args)
        
        print('Number of entries returned from initial UniProt search: %r\n' % len(self.uniprotxml)) 
        
        log_unique_domain_names(self.uniprot_query_string, self.uniprotxml)
        if self.uniprot_domain_regex:
            log_unique_domain_names_selected_by_regex(self.uniprot_domain_regex, self.uniprotxml)
        fasta_ofilepath = os.path.join(core.default_project_dirnames.targets, 'targets.fa')
        self._extract_targets_from_uniprot_xml() 
        if write_output_files:
            Bio.SeqIO.write(self.targets, fasta_ofilepath, 'fasta')

            
    def _extract_targets_from_uniprot_xml(self):
        targets = []
        residue_spans = []
        domain_descriptions = []
        for entry in self.uniprotxml.findall('entry'):
            entry_name = entry.find('name').text
            fullseq = core.sequnwrap(entry.find('sequence').text)
            if self.uniprot_domain_regex:
                selected_domains = entry.xpath(
                    'feature[@type="domain"][match_regex(@description, "%s")]' % self.uniprot_domain_regex,
                    extensions={(None, 'match_regex'): ensembler.core.xpath_match_regex_case_sensitive}
                )

                domain_iter = 0
                for domain in selected_domains:
                    targetid = '%s_D%d' % (entry_name, domain_iter)
                    # domain span override
                    if targetid in self.manual_overrides.target.domain_spans:
                        start, end = [int(x) - 1 for x in self.manual_overrides.target.domain_spans[targetid].split('-')]
                    else:
                        start, end = [int(domain.find('location/begin').get('position')) - 1,
                                      int(domain.find('location/end').get('position')) - 1]
                    targetseq = fullseq[start:end + 1]
                    targets.append(SeqRecord(Seq(targetseq), id=targetid, description=targetid))
                    residue_spans.append([start, end])
                    domain_descriptions.append(domain.get('description'))
                    domain_iter += 1

            else:
                targetid = entry_name
                targets.append(SeqRecord(Seq(fullseq), id=targetid, description=targetid))
                residue_spans.append([0, len(fullseq)-1])

        self.targets = targets
        self.residue_spans = residue_spans
        self.domain_descriptions = domain_descriptions
        


@notify_when_done
def gather_templates_from_uniprot(uniprot_query_string, uniprot_domain_regex=None, structure_dirs=None, pdbids=None, chainids=None, loglevel=None):
    """# Searches UniProt for a set of template proteins with a user-defined
    query string, then saves IDs, sequences and structures."""
    manual_overrides = core.ManualOverrides()
    selected_pdbchains = None
    
    uniprotxml = uniprot.get_uniprot_xml(uniprot_query_string)
    log_unique_domain_names(uniprot_query_string, uniprotxml)
    if uniprot_domain_regex is not None:
        log_unique_domain_names_selected_by_regex(uniprot_domain_regex, uniprotxml)

    selected_pdbchains = extract_template_pdbchains_from_uniprot_xml(uniprotxml, uniprot_domain_regex=uniprot_domain_regex, manual_overrides=manual_overrides, specified_pdbids=pdbids, specified_chainids=chainids)
    get_structure_files(selected_pdbchains, structure_dirs)

    print('Selected PDB chains: {0}'.format([pdbchain['templateid'] for pdbchain in selected_pdbchains]))

    selected_templates = extract_template_pdb_chain_residues(selected_pdbchains)
    write_template_seqs_to_fasta_file(selected_templates)
    extract_template_structures_from_pdb_files(selected_templates)
    write_gather_templates_from_uniprot_metadata(uniprot_query_string, uniprot_domain_regex, len(selected_templates), structure_dirs)
    

def log_unique_domain_names(uniprot_query_string, uniprotxml):
    # Example query string: 'domain:"Protein kinase" AND reviewed:yes'
    domain_match = re.search('domain:([\"\'].*[\"\'])', uniprot_query_string)
    if domain_match and len(domain_match.groups()) > 0:
        query_string_domain_selection = domain_match.groups()[0].replace('\'', '').replace('\"', '')
        uniprot_query_string_domains = uniprotxml.xpath(
            'entry/feature[@type="domain"][match_regex(@description, "%s")]' % query_string_domain_selection,
            extensions={
                (None, 'match_regex'): ensembler.core.xpath_match_regex_case_insensitive
            }
        )
        uniprot_unique_domain_names = set([domain.get('description') for domain in uniprot_query_string_domains])
        print('Set of unique domain names selected by the domain selector \'%s\' during the initial UniProt search:\n%s\n'
                    % (query_string_domain_selection, uniprot_unique_domain_names))

    else:
        uniprot_domains = uniprotxml.xpath('entry/feature[@type="domain"]')
        uniprot_unique_domain_names = set([domain.get('description') for domain in uniprot_domains])
        print('Set of unique domain names returned from the initial UniProt search using the query string \'%s\':\n%s\n'
                    % (uniprot_query_string, uniprot_unique_domain_names))


def log_unique_domain_names_selected_by_regex(uniprot_domain_regex, uniprotxml):
    regex_matched_domains = uniprotxml.xpath(
        'entry/feature[@type="domain"][match_regex(@description, "%s")]' % uniprot_domain_regex,
        extensions={(None, 'match_regex'): ensembler.core.xpath_match_regex_case_sensitive}
    )
    regex_matched_domains_unique_names = set([domain.get('description') for domain in regex_matched_domains])
    print('Unique domain names selected after searching with the case-sensitive regex string \'%s\':\n%s\n'
        % (uniprot_domain_regex, regex_matched_domains_unique_names))
    
    
def extract_template_pdbchains_from_uniprot_xml(uniprotxml, uniprot_domain_regex=None, manual_overrides=None, specified_pdbids=None, specified_chainids=None):
    """
    Parameters
    ----------
    uniprotxml: lxml.etree.Element
    uniprot_domain_regex: str
    manual_overrides: ensembler.core.ManualOverrides
    specified_pdbids: list of str
        ['2QR8', '4GU9']
    specified_chainids: dict of list of str
        {'2QR8': ['A'], '4GU9': ['A', 'B']}

    Returns
    -------
    selected_pdbchains: list of dict
        [
            {
                'templateid': str,
                'pdbid': str,
                'chainid': str,
                'residue_span': [
                    start (int),   # 1-based inclusive
                    end (int)      # 1-based inclusive
                ]
            }
        ]
    """
    selected_pdbchains = []
    all_uniprot_entries = uniprotxml.findall('entry')
    for entry in all_uniprot_entries:
        entry_name = entry.find('name').text
        if uniprot_domain_regex:
            selected_domains = entry.xpath(
                'feature[@type="domain"][match_regex(@description, "%s")]' % uniprot_domain_regex,
                extensions={(None, 'match_regex'): ensembler.core.xpath_match_regex_case_sensitive}
            )

            domain_iter = 0
            for domain in selected_domains:
                domain_id = '%s_D%d' % (entry_name, domain_iter)
                domain_span = [int(domain.find('location/begin').get('position')), int(domain.find('location/end').get('position'))]
                if manual_overrides and domain_id in manual_overrides.template.domain_spans:
                    domain_span = [int(x) for x in manual_overrides.template.domain_spans[domain_id].split('-')]
                domain_len = domain_span[1] - domain_span[0] + 1
                if manual_overrides and manual_overrides.template.min_domain_len is not None and domain_len < manual_overrides.template.min_domain_len:
                    continue
                if manual_overrides and manual_overrides.template.max_domain_len is not None and domain_len > manual_overrides.template.max_domain_len:
                    continue

                domain_iter += 1
                pdbs = domain.getparent().xpath(
                    'dbReference[@type="PDB"]/property[@type="method"][@value="X-ray" or @value="NMR"]/..'
                )

                for pdb in pdbs:
                    pdbid = pdb.get('id')
                    if manual_overrides and pdbid in manual_overrides.template.skip_pdbs:
                        continue
                    if specified_pdbids and pdbid not in specified_pdbids:
                        continue
                    pdb_chain_span_nodes = pdb.findall('property[@type="chains"]')

                    for pdb_chain_span_node in pdb_chain_span_nodes:
                        chain_span_string = pdb_chain_span_node.get('value')
                        chain_spans = ensembler.uniprot.parse_uniprot_pdbref_chains(chain_span_string)

                        for chainid in chain_spans.keys():
                            if specified_chainids and len(specified_chainids[pdbid]) > 0 and chainid not in specified_chainids[pdbid]:
                                continue
                            span = chain_spans[chainid]
                            if (span[0] < domain_span[0] + 30) & (span[1] > domain_span[1] - 30):
                                templateid = '%s_%s_%s' % (domain_id, pdbid, chainid)
                                data = {
                                    'templateid': templateid,
                                    'pdbid': pdbid,
                                    'chainid': chainid,
                                    'residue_span': domain_span
                                }
                                selected_pdbchains.append(data)

        else:
            pdbs = entry.xpath(
                'dbReference[@type="PDB"]/property[@type="method"][@value="X-ray" or @value="NMR"]/..'
            )

            for pdb in pdbs:
                pdbid = pdb.get('id')
                if manual_overrides and pdbid in manual_overrides.template.skip_pdbs:
                    continue
                if specified_pdbids and pdbid not in specified_pdbids:
                    continue
                pdb_chain_span_nodes = pdb.findall('property[@type="chains"]')

                for pdb_chain_span_node in pdb_chain_span_nodes:
                    chain_span_string = pdb_chain_span_node.get('value')
                    chain_spans = uniprot.parse_uniprot_pdbref_chains(chain_span_string)

                    for chainid in chain_spans.keys():
                        if specified_chainids and len(specified_chainids[pdbid]) > 0 and chainid not in specified_chainids[pdbid]:
                            continue
                        span = chain_spans[chainid]
                        templateid = '%s_%s_%s' % (entry_name, pdbid, chainid)
                        data = {
                            'templateid': templateid,
                            'pdbid': pdbid,
                            'chainid': chainid,
                            'residue_span': span
                        }
                        selected_pdbchains.append(data)

    print('%d PDB chains selected.' % len(selected_pdbchains))
    return selected_pdbchains


def get_structure_files(selected_pdbchains, structure_dirs=None):
    if structure_dirs:
        for structure_dir in structure_dirs:
            if not os.path.exists(structure_dir):
                logger.warn('Structure directory {0} not found'.format(structure_dir))
    for pdbchain in selected_pdbchains:
        get_structure_files_for_single_pdbchain(pdbchain['pdbid'], structure_dirs)
        
        
def get_structure_files_for_single_pdbchain(pdbid, structure_dirs=None):
    if type(structure_dirs) != list:
        structure_dirs = []
    project_structures_dir = 'structures'
    for structure_type in ['pdb', 'sifts']:
        project_structure_filepath = os.path.join(
            project_structures_dir,
            structure_type,
            pdbid + structure_type_file_extension_mapper[structure_type]
        )
        if not file_exists_and_not_empty(project_structure_filepath):
            attempt_symlink_structure_files(
                pdbid, project_structures_dir, structure_dirs, structure_type=structure_type
            )
            if not os.path.exists(project_structure_filepath):
                download_structure_file(
                    pdbid, project_structure_filepath, structure_type=structure_type
                )

                
def attempt_symlink_structure_files(pdbid, project_structures_dir, structure_dirs, structure_type='pdb'):
    project_structure_filepath = os.path.join(project_structures_dir, structure_type, pdbid + structure_type_file_extension_mapper[structure_type])
    for structure_dir in structure_dirs:
        structure_filepath = os.path.join(structure_dir, pdbid + structure_type_file_extension_mapper[structure_type])
        if os.path.exists(structure_filepath):
            if file_exists_and_not_empty(structure_filepath) > 0:
                if os.path.exists(project_structure_filepath):
                    os.remove(project_structure_filepath)
                os.symlink(structure_filepath, project_structure_filepath)
                break

def download_structure_file(pdbid, project_structure_filepath, structure_type='pdb'):
    if structure_type == 'pdb':
        download_pdb_file(pdbid, project_structure_filepath)
    elif structure_type == 'sifts':
        download_sifts_file(pdbid, project_structure_filepath)


def download_pdb_file(pdbid, project_pdb_filepath):
    print('Downloading PDB file for: %s' % pdbid)
    pdbgz_page = pdb.retrieve_pdb(pdbid, compressed='yes').decode("utf-8") 
    with open(project_pdb_filepath, 'w') as pdbgz_file:
        pdbgz_file.write(pdbgz_page)


def download_sifts_file(pdbid, project_sifts_filepath):
    print('Downloading sifts file for: %s', pdbid)
    sifts_page = pdb.retrieve_sifts(pdbid).decode("utf-8") 
    with gzip.open(project_sifts_filepath, 'wb') as project_sifts_file:
        project_sifts_file.write(sifts_page)
