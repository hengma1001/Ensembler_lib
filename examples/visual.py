import __main__
__main__.pymol_argv = [ 'pymol', '-qc']
import pymol
pymol.finish_launching()


pdbs = open('./models/PK3CA_HUMAN/unique-models.txt', 'r').read().split() 

pymol_names = []
for i, pdb in enumerate(pdbs): 
    pymol_name = pdb[-21:-17] + pdb[-8:-4]
    pymol.cmd.load(pdb, pymol_name) 
    pymol_names.append(pymol_name)


for pymol_name in pymol_names[1:]: 
    pymol.cmd.align(pymol_name, pymol_names[0])


pymol.cmd.zoom()
