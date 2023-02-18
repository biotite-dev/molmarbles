from os.path import join, dirname, realpath
import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import openmm.app as app
import molmarbles


TEST_FILE = join(dirname(realpath(__file__)), "1aki.cif")


@pytest.mark.parametrize("include_box", [False, True])
def test_topology_conversion(include_box):
    """
    Converting an :class:`AtomArray` into a :class:`Topology` and back
    again should not change the :class:`AtomArray`.
    """
    ref_atoms = pdbx.get_structure(pdbx.PDBxFile.read(TEST_FILE), model=1)
    ref_atoms.bonds = struc.connect_via_residue_names(ref_atoms)
    if not include_box:
        ref_atoms.box = None
    
    topology = molmarbles.to_topology(ref_atoms)
    test_atoms = molmarbles.from_topology(topology)

    # The Topology cannot properly handle the aromatic bond types of Biotite
    ref_atoms.bonds.remove_aromaticity()
    _assert_equal_atom_arrays(test_atoms, ref_atoms)


def test_topology_consistency():
    """
    Test whether an :class:`AtomArray` converted from a
    :class:`Topology` read via OpenMM is equal to :class:`AtomArray`
    directly read via Biotite.
    """
    ref_atoms = pdbx.get_structure(
        pdbx.PDBxFile.read(TEST_FILE), model=1, extra_fields=["label_asym_id"]
    )
    # OpenMM uses author fields, except for the chain ID,
    # where it uses the label field
    ref_atoms.chain_id = ref_atoms.label_asym_id
    ref_atoms.del_annotation("label_asym_id")
    ref_atoms.bonds = struc.connect_via_residue_names(ref_atoms)

    topology = app.PDBxFile(TEST_FILE).topology
    test_atoms = molmarbles.from_topology(topology)

    # OpenMM does not parse the bond type from CIF files
    ref_atoms.bonds.remove_bond_order()
    # Biotite does not parse disulfide bridges
    # -> Remove them from the bonds parsed by OpenMM
    for i, j, _ in test_atoms.bonds.as_array():
        if test_atoms.element[i] == "S" and test_atoms.element[j] == "S":
            test_atoms.bonds.remove_bond(i, j)

    _assert_equal_atom_arrays(test_atoms, ref_atoms)


def _assert_equal_atom_arrays(test_atoms, ref_atoms):
    for category in ref_atoms.get_annotation_categories():
        assert np.array_equal(
            test_atoms.get_annotation(category),
             ref_atoms.get_annotation(category)
        )
    
    if ref_atoms.box is not None:
        assert np.allclose(test_atoms.box, ref_atoms.box)
    else:
        assert test_atoms.box is None
    
    # Do not compare array from 'BondList.as_array()',
    # as the comparison would not allow different order of bonds
    assert test_atoms.bonds == ref_atoms.bonds