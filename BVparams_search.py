"""
This module implements bond valence parameter optimization using theoretical network valence analysis.
The main classes are:
1. TheoreticalBondValenceSolver - Solves for theoretical bond valences using network equations
2. BVParamSolver - Optimizes bond valence parameters (R0, B) using different optimization algorithms

Key concepts:
- Uses crystal structure and bonding information to set up network equations
- Solves equations to get theoretical bond valences (Sij)
- Optimizes bond valence parameters R0 and B to match theoretical and observed bond lengths
"""

# Import required libraries
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN
import networkx as nx
import sympy as sp
from scipy.optimize import brute, shgo, differential_evolution, dual_annealing, direct
import numpy as np
import json
import re
import os

class TheoreticalBondValenceSolver:
    """
    Solves for theoretical bond valences using network equations derived from:
    1. Valence sum rule: Sum(Sij) = Vi
    2. Kirchoff's laws: Sum(Sij)_loop = 0
    """
    
    def __init__(self, element2charge_path='element2charge.json',
                 species_matID_path='params/dict_matID_possible_species.json'):
        """
        Initialize the TheoreticalBondValenceSolver object
        
        Args:
            element2charge_path (str): Path to JSON file mapping elements to common charges
            species_matID_path (str): Path to JSON file mapping material IDs to possible species
        """
        # Load element to charge mapping
        with open(element2charge_path, 'r') as f:
            self.dict_ele2charge = json.load(f)
            
        # Load material ID to species mapping
        with open(species_matID_path, 'r') as f:
            self.dict_species_matID = json.load(f)

    def get_element(self, label):
        """
        Extract element from site label
        
        Args:
            label (str): Site label (e.g. 'Li1', 'O2')
            
        Returns:
            str: Element symbol
        """
        return re.split(r'[^a-zA-Z]', label)[0]

    def idx2label(self, struct):
        """
        Create index to label mapping for structure sites
        
        Args:
            struct (Structure): Pymatgen Structure object
            
        Returns:
            dict: Mapping of site indices to labels
        """
        return {i: site.label for i, site in enumerate(struct.sites)}

    def relabel_site_labels_sym(self, struct):
        """
        Relabel sites using symmetry-equivalent labels
        
        Args:
            struct (Structure): Pymatgen Structure object to relabel
        """
        sga = SpacegroupAnalyzer(struct)
        sym_struct = sga.get_symmetrized_structure()
        
        new_labels = {}
        processed_element = {}
        
        # Generate new labels for symmetry-equivalent sites
        for idx, group in enumerate(sym_struct.equivalent_indices):
            species = self.get_element(struct[group[0]].species_string)
            if species not in processed_element:
                processed_element[species] = 0
            processed_element[species] += 1
            for site in group:
                new_labels[site] = f'{species}{processed_element[species]}'
                
        # Apply new labels
        for i, site in enumerate(struct):
            site.label = f"{new_labels[i]}"

    def get_element2charge(self, list_of_possible_species):
        """
        Create element to charge mapping from list of possible species
        
        Args:
            list_of_possible_species (list): List of species strings (e.g. ['Li+', 'O2-'])
            
        Returns:
            dict: Element to charge mapping
        """
        ele2char = {}
        for spec in list_of_possible_species:
            element = re.split(r'[^a-zA-Z]', spec)[0]
            charge = spec.split(element)[1]
            chargeval = re.split(r'[+,-]', charge)[0] or '1'
            sign = charge[-1]
            charge = float(f'{sign}{chargeval}')
            ele2char[element] = charge
        return ele2char

    def find_cycles(self, edges):
        """
        Find cycles in bonding graph using Depth-First Search (DFS).

        The algorithm works as follows:
        1. Create an adjacency list representation of the graph from the edges.
        2. Iterate over all nodes in the graph.
        3. For each node, perform a DFS to detect cycles.
        4. If a cycle is detected, add it to the set of cycles.

        Args:
            edges (list): List of edges in bonding graph

        Returns:
            list: List of cycles (each cycle is a tuple of node indices)
        """
        graph = {}
        for node1, node2 in edges:
            graph.setdefault(node1, set()).add(node2)
            graph.setdefault(node2, set()).add(node1)

        cycles = set()
        
        def dfs(node, visited, path):
            """
            Perform DFS starting from the given node to detect a cycle.

            Args:
                node (int): Node index
                visited (set): Set of visited nodes
                path (list): Path of nodes visited in the current DFS

            Returns:
                None
            """
            # Mark the current node as visited and add it to the path
            visited.add(node)
            path.append(node)
            
            # Iterate over all neighbors of the current node
            for neighbor in graph[node]:
                # If the neighbor is already in the path, a cycle is detected
                if neighbor in path:
                    idx = path.index(neighbor)
                    cycles.add(tuple(path[idx:]))
                # If the neighbor is not visited, continue the DFS
                elif neighbor not in visited:
                    dfs(neighbor, visited, path)
                    
            # Remove the current node from the path
            path.pop()
            
        visited = set()
        for node in graph:
            if node not in visited:
                dfs(node, visited, [])
                
        return [cycle for cycle in cycles if len(cycle) >= 4]

    def get_cycle_path(self, graph, idx2label):
        """
        Convert cycles from indices to site labels
        
        Args:
            graph (nx.Graph): Bonding graph
            idx2label (dict): Index to label mapping
            
        Returns:
            list: List of cycles with site labels
        """
        cycles = self.find_cycles(list(graph.edges()))
        return [[idx2label[idx] for idx in cycle] for cycle in cycles]

    def get_eq_cycle(self, cur_cycle, dict_element_charge):
        """
        Generate network equation from cycle
        
        Args:
            cur_cycle (list): Cycle of site labels
            dict_element_charge (dict): Element to charge mapping
            
        Returns:
            str: Network equation string
        """
        eq_temp = ''
        for i, e in enumerate(cur_cycle):
            atom_element = self.get_element(e)
            next_e = cur_cycle[0] if i == len(cur_cycle)-1 else cur_cycle[i+1]
            
            if dict_element_charge[atom_element] > 0:
                eq_temp = eq_temp[:-1]
                temp_bond = f'-{e}{next_e}+'
            else:
                temp_bond = f'{next_e}{e}+'
            eq_temp += temp_bond
            
        return eq_temp[:-1]

    def solve_sij(self, variables, equations):
        """
        Solve system of network equations for bond valences
        
        Args:
            variables (list): List of bond variables
            equations (list): List of (equation, value) tuples
            
        Returns:
            dict: Bond valence solutions or None if no solution
        """
        symbols_list = sp.symbols(variables)
        eqs = [sp.Eq(sp.sympify(eq[0]), eq[1]) for eq in equations]
        solution = sp.solve(eqs, symbols_list)
        
        if not solution or any(isinstance(v, sp.Add) for v in solution.values()):
            return None
            
        return {str(k): float(v) for k,v in solution.items()}

    def get_eqs_from_valence_sum_rule(self, struct, bonds, dict_charge):
        """
        Generate equations from valence sum rule
        
        Args:
            struct (Structure): Crystal structure
            bonds (CrystalNN): Bonding information
            dict_charge (dict): Element to charge mapping
            
        Returns:
            tuple: (equations, bond variables, bond lengths)
        """
        equations = []
        bond_variables = set()
        bond_lengths = {}
        
        for i, site in enumerate(struct):
            site_element = self.get_element(site.label)
            atom_valence = getattr(site.specie, 'oxi_state', dict_charge[site_element])
            
            # Generate bond terms for current site
            bond_terms = []
            for neighbor in bonds.get_connected_sites(i):
                # Determine bond type based on valence
                bond_type = (f'{site.label}{neighbor.site.label}' if atom_valence > 0 
                            else f'{neighbor.site.label}{site.label}')
                
                # Store bond information
                bond_variables.add(bond_type)
                bond_lengths[bond_type] = neighbor.dist
                bond_terms.append(bond_type)
            
            # Create equation if bonds exist
            if bond_terms:
                equation = ' + '.join(bond_terms)
                equations.append((equation, abs(atom_valence)))
                
        return equations, bond_variables, bond_lengths

    def get_eqs_from_loops(self, struct, graph, dict_charge):
        """
        Generate equations from cycles using Kirchoff's laws
        
        Args:
            struct (Structure): Crystal structure
            graph (nx.Graph): Bonding graph
            dict_charge (dict): Element to charge mapping
            
        Returns:
            list: List of (equation, 0) tuples
        """
        id2label = self.idx2label(struct)
        cycle_list = self.get_cycle_path(graph.graph, id2label)
        return [(self.get_eq_cycle(cur_cycle, dict_charge), 0) 
                for cur_cycle in cycle_list]

    def get_sij(self, matID, struct, graph):
        """
        Calculate theoretical bond valences using network equations
        
        Args:
            matID (str): Material ID
            struct (Structure): Crystal structure
            graph (nx.Graph): Bonding graph
            
        Returns:
            tuple: (bond valences, bond variables, bond lengths, charge dict)
        """
        self.relabel_site_labels_sym(struct)
        dict_charge = self.get_element2charge(self.dict_species_matID.get(matID, [])) or self.dict_ele2charge
        
        equations_val_sum, bond_vars, bondL = self.get_eqs_from_valence_sum_rule(struct, graph, dict_charge)
        equations_cycle = self.get_eqs_from_loops(struct, graph, dict_charge)
        
        sijs = self.solve_sij(bond_vars, equations_val_sum + equations_cycle)
        return sijs, list(bond_vars), bondL, dict_charge

class BVParamSolver:
    """
    Optimizes bond valence parameters R0 and B using different optimization algorithms
    
    Args:
        save_dir (str): Directory to save results
        algo (str): Optimization algorithm to use ('shgo', 'brute', etc)
        no_sol (list): List to store materials with no solution
        
    Attributes:
        algo (str): Optimization algorithm
        no_sol (list): Materials with no solution
    """
    
    def __init__(self, save_dir='res', algo="shgo", no_sol=[]):
        self.algo = algo
        self.no_sol = no_sol
        
        # Create output directories
        os.makedirs(f'{save_dir}/R0Bs/{algo}', exist_ok=True)
        os.makedirs(f'{save_dir}/no_solu', exist_ok=True)

    def objective(self, variables, eqs):
        """
        Objective function for optimization - mean squared error
        
        Args:
            variables (list): [R0, B] values
            eqs (list): List of symbolic equations
            
        Returns:
            float: Mean squared error
        """
        R0, B = sp.symbols('R0 B')
        return sum(float(eq.subs({R0: variables[0], B: variables[1]}))**2 for eq in eqs)

    def get_eqs_for_R0B(self, cation, anion, bond_type_list, networkValence_dict, 
                       bondLen_dict, matID, reduced_formula, R0_bounds):
        """
        Generate equations for R0 and B optimization
        
        Args:
            cation (str): Cation element
            anion (str): Anion element
            bond_type_list (list): List of bond types
            networkValence_dict (dict): Bond valence values
            bondLen_dict (dict): Bond lengths
            matID (str): Material ID
            reduced_formula (str): Reduced formula
            R0_bounds (tuple): Bounds for R0
            
        Returns:
            tuple: (equations, B bounds)
        """
        target_bonds = [e for e in bond_type_list 
                       if re.split(r'\d+', e)[0] == cation 
                       and re.split(r'\d+', e)[1] == anion]

        bmax, bmin = -10, 1000
        eqs_list = []
        
        for bond in target_bonds:
            sij = networkValence_dict[bond]
            if sij <= 0:
                self.no_sol.append((matID, cation, anion, reduced_formula, 'negative_Sij'))
                return [], None
                
            eqs_list.append(f'R0 - B*log({sij}) - {bondLen_dict[bond]}')
            
            if sij != 1:
                b1 = (R0_bounds[0] - bondLen_dict[bond])/np.log(sij)
                b2 = (R0_bounds[1] - bondLen_dict[bond])/np.log(sij)
                bmax = max(bmax, b1, b2)
                bmin = min(bmin, b1, b2)
                
        if not (-10 < bmax < np.inf) or not (-10 < bmin < np.inf):
            bmin, bmax = -5, 5
            
        if not eqs_list:
            self.no_sol.append((matID, cation, anion, reduced_formula, 'no_eqs_from_graph'))
            
        return eqs_list, (bmin, bmax)

    def solve_R0Bs(self, cation, anion, bond_type_list, networkValence_dict,
                  bondLen_dict, materID, chem_formula, R0_bounds):
        """
        Optimize R0 and B parameters
        
        Args:
            cation (str): Cation element
            anion (str): Anion element
            bond_type_list (list): List of bond types
            networkValence_dict (dict): Bond valence values
            bondLen_dict (dict): Bond lengths
            materID (str): Material ID
            chem_formula (str): Chemical formula
            R0_bounds (tuple): Bounds for R0
            
        Returns:
            tuple: Optimized (R0, B) values
        """
        eqs_list_math, B_bounds = self.get_eqs_for_R0B(cation, anion, bond_type_list,
                                                      networkValence_dict, bondLen_dict,
                                                      materID, chem_formula, R0_bounds)
        if not eqs_list_math:
            return []
            
        eqs_list_sympy = [sp.sympify(eq) for eq in eqs_list_math]
        
        # Select optimization algorithm
        optimizers = {
            'brute': lambda: brute(self.objective, [R0_bounds, B_bounds], args=(eqs_list_sympy,)),
            'shgo': lambda: shgo(self.objective, [R0_bounds, B_bounds], args=(eqs_list_sympy,)).x,
            'diff': lambda: differential_evolution(self.objective, [R0_bounds, B_bounds], args=(eqs_list_sympy,)).x,
            'dual_annealing': lambda: dual_annealing(self.objective, [R0_bounds, B_bounds], args=(eqs_list_sympy,)).x,
            'direct': lambda: direct(self.objective, [R0_bounds, B_bounds], args=(eqs_list_sympy,)).x
        }
        
        # Get selected algorithm or default to SHGO
        if self.algo not in optimizers:
            print(f"Warning: Algorithm '{self.algo}' not found. Using default SHGO optimizer.")
        result = optimizers.get(self.algo, optimizers['shgo'])()
        return result if isinstance(result, tuple) else (result[0], result[1])