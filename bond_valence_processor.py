from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
import numpy as np
import json
from mp_api.client import MPRester
from BVparams_search import TheoreticalBondValenceSolver, BVParamSolver

@dataclass
class MaterialData:
    material_id: str
    possible_species: List[str]
    structure_graph: Optional[object] = None
    formula_pretty: Optional[str] = None

class BondValenceProcessor:
    def __init__(self, api_key: str, algos: List[str], cations: List[str], anion: str):
        self.api_key = api_key
        self.algos = algos
        self.cations = cations
        self.anion = anion
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist for all cations"""
        Path("res").mkdir(exist_ok=True)
        for cation in self.cations:
            cation_dir = Path(f"res/{cation}{self.anion}")
            cation_dir.mkdir(exist_ok=True)
            (cation_dir / "params").mkdir(exist_ok=True)
            (cation_dir / "R0Bs").mkdir(exist_ok=True)
            (cation_dir / "no_solu").mkdir(exist_ok=True)
            for algo in self.algos:
                (cation_dir / "R0Bs" / algo).mkdir(exist_ok=True)

    def get_possible_species(self, save_dir: str, docs: List[MaterialData]) -> List[str]:
        """Extract possible species from materials documents"""
        species_data = {
            doc.material_id: doc.possible_species
            for doc in tqdm(docs, desc='Getting possible species')
            if doc.possible_species
        }
        
        output_file = Path(save_dir) / "params" / "dict_matID_possible_species.json"
        with output_file.open('w') as f:
            json.dump(species_data, f)
            
        return list(species_data.keys())

    def process_cation_system(self, cation: str, anion: str) -> None:
        """Process a single cation-oxygen system"""
        print(f'Processing {cation}-{anion} system...')
        
        # Setup processing pipeline
        docs = self._download_materials_data(cation)
        res_dir = f'res/{cation}{anion}'
        mids = self.get_possible_species(res_dir, docs)
        if not mids:
            print(f"No materials with possible species found for {cation}-O system, skipping...")
            return
        bonds_docs = self._download_bonding_data(mids)
        
        # Initialize data structures
        results = {
            'sij': {},
            'charges': {},
            'solved': set(),
            'no_solution': []
        }
        
        # Load previous results if they exist
        results.update(self._load_previous_results(res_dir))
        
        # Process materials
        self._process_materials(
            bonds_docs=bonds_docs,
            results=results,
            res_dir=res_dir,
            cation=cation
        )
        
        # Save final results
        self._save_results(res_dir, results['sij'], results['charges'])

    def _download_materials_data(self, cation: str, anion: str) -> List[MaterialData]:
        """Download materials data from Materials Project"""
        with MPRester(api_key=self.api_key) as mpr:
            return mpr.materials.summary.search(
                elements=[cation, anion],
                energy_above_hull=(0.000, 0.05),
                fields=['material_id', 'possible_species']
            )

    def _download_bonding_data(self, material_ids: List[str]) -> List[MaterialData]:
        """Download bonding data from Materials Project"""
        with MPRester(api_key=self.api_key) as mpr:
            return mpr.materials.bonds.search(
                material_ids=material_ids,
                fields=['material_id', 'structure_graph', 'formula_pretty']
            )

    def _load_previous_results(self, res_dir: str) -> Dict:
        """Load previously computed results"""
        results = {
            'solved': set(),
            'no_solution': []
        }
        
        # Load solved materials and find common solved files across all algorithms
        solved_sets = []
        for alg in self.algos:
            alg_dir = Path(res_dir) / "R0Bs" / alg
            if alg_dir.exists():
                solved_files = {f.stem for f in alg_dir.glob("*.txt")}
                solved_sets.append(solved_files)
        
        # Find intersection of all solved files across algorithms
        if solved_sets:
            common_solved = set.intersection(*solved_sets)
            results['solved'].update(common_solved)
        
        # Load no-solution cases
        for alg in self.algos:
            no_solu_file = Path(res_dir) / "no_solu" / f"{alg}.txt"
            if no_solu_file.exists():
                alg_no_solu = np.loadtxt(no_solu_file, dtype=str).tolist()
                results['no_solution'].extend(alg_no_solu)
        
        # Deduplicate no-solution cases
        if results['no_solution']:
            results['no_solution'] = list({tuple(item) for item in results['no_solution']})
            
        return results

    def _process_materials(self, bonds_docs: List[MaterialData], results: Dict, 
                         res_dir: str, cation: str, anion: str) -> None:
        """Process each material in the dataset"""
        solver = TheoreticalBondValenceSolver(
            species_matID_path=str(Path(res_dir) / "params" / "dict_matID_possible_species.json")
        )
        
        for material in tqdm(bonds_docs, desc=f'Processing {cation} materials'):
            if material.material_id in results['solved']:
                continue
                
            # Compute Sij values
            sij_data = solver.get_sij(
                material.material_id,
                material.structure_graph.structure,
                material.structure_graph
            )
            
            # Store results
            results['sij'][material.material_id] = sij_data[0]
            results['charges'][material.material_id] = sij_data[3]
            
            # Process with algorithms
            self._run_algorithms(
                sij_data=sij_data,
                material=material,
                results=results,
                res_dir=res_dir,
                cation=cation,
                anion=anion
            )

    def _run_algorithms(self, sij_data: Tuple, material: MaterialData, 
                       results: Dict, res_dir: str, cation: str, anion: str) -> None:
        """Run all algorithms on the material"""
        network_valence, bond_types, bond_lengths, _ = sij_data
        
        if not network_valence:
            no_solution_case = (
                material.material_id, cation, anion, 
                material.formula_pretty, 'no_network_sol'
            )
            results['no_solution'].append(no_solution_case)
            self._save_no_solution(res_dir, results['no_solution'])
            return
            
        for algorithm in self.algos:
            solver = BVParamSolver(
                save_dir=res_dir,
                algo=algorithm,
                no_sol=results['no_solution']
            )
            
            solution = solver.solve_R0Bs(
                cation=cation,
                anion=anion,
                bond_type_list=bond_types,
                networkValence_dict=network_valence,
                bondLen_dict=bond_lengths,
                materID=material.material_id,
                chem_formula=material.formula_pretty,
                R0_bounds=(0, 5),
            )
            
            if solution:
                output_file = Path(res_dir) / "R0Bs" / algorithm / f"{material.material_id}.txt"
                np.savetxt(output_file, solution)

    def _save_no_solution(self, res_dir: str, no_solution: List) -> None:
        """Save no-solution cases for all algorithms"""
        for algorithm in self.algos:
            output_file = Path(res_dir) / "no_solu" / f"{algorithm}.txt"
            np.savetxt(output_file, no_solution, fmt='%s')

    def _save_results(self, res_dir: str, sij_data: Dict, charges: Dict) -> None:
        """Save final results to JSON files"""
        with open(Path(res_dir) / "dict_sijs.json", 'w') as f:
            json.dump(sij_data, f)
        
        with open(Path(res_dir) / "dict_charges.json", 'w') as f:
            json.dump(charges, f)


if __name__ == "__main__":
    # User-defined parameters
    user_cations = ['Li', 'Na', 'K', 'Rb', 'Cs']  # Can be modified by user
    user_anion = 'O'
    user_algos = ['shgo', 'brute', 'diff', 'dual_annealing', 'direct']
    api_key = "your_api_key"  # Should be provided by user
    
    processor = BondValenceProcessor(
        api_key=api_key,
        algos=user_algos,
        cations=user_cations
    )
    
    for cation in user_cations:
        processor.process_cation_system(cation, user_anion)