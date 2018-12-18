import os 
from collections import namedtuple

manual_overrides_filename = 'manual-overrides.yaml'

project_dirtypes = [
    'targets',
    'templates',
    'structures', 'models',
    'packaged_models',
    'structures_pdb',
    'structures_sifts',
    'templates_structures_resolved',
    'templates_structures_modeled_loops',
]
ProjectDirNames = namedtuple('ProjectDirNames', project_dirtypes)
default_project_dirnames = ProjectDirNames(
    targets='targets',
    templates='templates',
    structures='structures',
    models='models',
    packaged_models='packaged_models',
    structures_pdb=os.path.join('structures', 'pdb'),
    structures_sifts=os.path.join('structures', 'sifts'),
    templates_structures_resolved=os.path.join('templates', 'structures-resolved'),
    templates_structures_modeled_loops=os.path.join('templates', 'structures-modeled-loops'),
)


def encode_url_query(uniprot_query):
    def replace_all(text, replace_dict):
        for i, j in replace_dict.items():
            text = text.replace(i, j)
        return text

    encoding_dict = {
        ' ': '+',
        ':': '%3A',
        '(': '%28',
        ')': '%29',
        '"': '%22',
        '=': '%3D',
    }
    return replace_all(uniprot_query, encoding_dict)


def sequnwrap(sequence):
    """Unwraps a wrapped sequence string
    """
    unwrapped = sequence.strip()
    unwrapped = ''.join(unwrapped.split('\n'))
    return unwrapped 

class ManualOverrides:
    """
    Reads in user-defined override data from a YAML file named "manual-overrides.yaml"

    Parameters
    ----------
    manual_overrides_filepath: str
        In normal use, this should not need to be set. Defaults to 'manual-overrides.yaml'

    Example file contents
    ---------------------

    target-selection:
        domain-spans:
        ABL1_HUMAN_D0: 242-513
    template-selection:
        min-domain-len: 0
        max-domain-len: 350
        domain-spans:
            ABL1_HUMAN_D0: 242-513
        skip-pdbs:
            - 4CYJ
            - 4P41
            - 4P2W
            - 4QTD
            - 4Q2A
            - 4CTB
            - 4QOX
    refinement:
        ph: 8.0
        custom_residue_variants:
            DDR1_HUMAN_D0_PROTONATED:
                # keyed by 0-based residue index
                35: ASH

    Or see `ensembler/tests/example_project/manual_overrides.yaml` for an example file.
    """
    def __init__(self, manual_overrides_filepath=None):
        if not manual_overrides_filepath:
            manual_overrides_filepath = manual_overrides_filename
        if os.path.exists(manual_overrides_filepath):
            with open(manual_overrides_filepath, 'r') as manual_overrides_file:
                manual_overrides_yaml = yaml.load(manual_overrides_file, Loader=YamlLoader)
        else:
            manual_overrides_yaml = {}

        if type(manual_overrides_yaml) is not dict:
            manual_overrides_yaml = {}

        self.target = TargetManualOverrides(manual_overrides_yaml)
        self.template = TemplateManualOverrides(manual_overrides_yaml)
        self.refinement = RefinementManualOverrides(manual_overrides_yaml) 
        

class TargetManualOverrides:
    """
    Parameters
    ----------
    manual_overrides_yaml: dict

    Attributes
    ----------
    domain_spans: dict
        dict with structure {`targetid`: `domain_span`, ...} where
        `domain_span` is a str e.g. '242-513'
    """
    def __init__(self, manual_overrides_yaml):
        target_dict = manual_overrides_yaml.get('target-selection')
        if target_dict is not None:
            self.domain_spans = target_dict.get('domain-spans')
        else:
            self.domain_spans = {}


class TemplateManualOverrides:
    """
    Parameters
    ----------
    manual_overrides_yaml: dict

    Attributes
    ----------
    min_domain_len: int or NoneType
    max_domain_len: int or NoneType
    domain_spans: dict
        dict with structure {`targetid`: `domain_span`, ...} where
        `domain_span` is a str e.g. '242-513'
    skip_pdbs: list
        list of PDB IDs to skip
    """
    def __init__(self, manual_overrides_yaml):
        template_dict = manual_overrides_yaml.get('template-selection')
        if template_dict is not None:
            self.min_domain_len = template_dict.get('min-domain-len')
            self.max_domain_len = template_dict.get('max-domain-len')
            self.domain_spans = template_dict.get('domain-spans')
            self.skip_pdbs = template_dict.get('skip-pdbs') if template_dict.get('skip-pdbs') is not None else []
        else:
            self.min_domain_len = None
            self.max_domain_len = None
            self.domain_spans = {}
            self.skip_pdbs = []


class RefinementManualOverrides:
    """
    Parameters
    ----------
    manual_overrides_yaml: dict

    Attributes
    ----------
    ph: float or NoneType
    custom_residue_variants_by_targetid: dict or NoneType
        dict with structure {`targetid`: {residue_index: residue_name}, ...} where
        e.g. {'DDR1_HUMAN_D0_PROTONATED': {35: 'ASH'}}
    """
    def __init__(self, manual_overrides_yaml):
        refinement_dict = manual_overrides_yaml.get('refinement')
        if refinement_dict is not None:
            self.ph = refinement_dict.get('ph')
            self.custom_residue_variants_by_targetid = refinement_dict.get('custom_residue_variants')
        else:
            self.ph = None
            self.custom_residue_variants_by_targetid = {}