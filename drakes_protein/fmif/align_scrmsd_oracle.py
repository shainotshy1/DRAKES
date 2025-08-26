from Bio import PDB
import pyrosetta
import torch

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

# def get_structure_matrix(pdb_file):
#     parser = PDB.PDBParser(QUIET=True)
#     structure = parser.get_structure('protein', pdb_file)
#     coordinates = []
#     for model in structure:
#         for chain in model:
#             for residue in chain:
#                 for atom in residue:
#                     coordinates.append(atom.get_coord())    
#     structure_matrix = torch.asarray(coordinates)
#     return structure_matrix

def gen_scRMSD_reward(esm_model, true_seq, mask_for_loss, name):
    def get_pose(gen_folded_pdb_path):
        pose = pyrosetta.pose_from_file(gen_folded_pdb_path)
        
        # uncommented additional transformations for better runtime; still has effective alignment

        # scorefxn = pyrosetta.create_score_function("ref2015_cart")
        # tf = TaskFactory()
        # tf.push_back(RestrictToRepacking())
        # packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(pose))
        # packer.apply(pose)
        # relax = FastRelax()
        # relax.set_scorefxn(scorefxn)
        # relax.apply(pose)

        return pose
    
    true_seq = "".join([ALPHABET[x] for _ix, x in enumerate(true_seq[0]) if mask_for_loss[0][_ix] == 1])
    with torch.no_grad():
        true_output = esm_model.infer_pdb(true_seq)
    with open(f"temp_true_result_{name}.pdb", "w") as f:
        f.write(true_output)
    true_pose = get_pose(f"temp_true_result_{name}.pdb")

    def reward_oracle(ssps):
        res = torch.zeros((ssps.shape[0]), device=ssps.device)
        for i, ssp in enumerate(ssps):
            ssp_str = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[0][_ix] == 1])
            with torch.no_grad():
                output = esm_model.infer_pdb(ssp_str)
            with open(f"temp_result_{name}.pdb", "w") as f:
                f.write(output)
            pose = get_pose(f"temp_result_{name}.pdb")
            res[i] = -1 * pyrosetta.rosetta.core.scoring.bb_rmsd(true_pose, pose)
        return res
        
    return reward_oracle

