from os.path import join, dirname, realpath
import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
import openmm
import openmm.unit as unit
import openmm.app as app
import molmarbles


TEST_FILE = join(dirname(realpath(__file__)), "1aki.cif")


@pytest.mark.parametrize("multi_state", [False, True])
def test_state_conversion(multi_state):
    """
    Test whether the :class:`AtomArray` obtained from a :class:`State`
    matches the original template atom array with newly generated
    positions.
    """
    template = pdbx.get_structure(
        pdbx.PDBxFile.read(TEST_FILE), model=1
    )
    system = molmarbles.to_system(template)
    # Create an arbitrary integrator
    integrator = openmm.VerletIntegrator(1)
    context = openmm.Context(system, integrator)

    # Generate arbitrary coordinates and box vectors
    coord = np.arange(template.array_length() * 3).reshape(-1, 3)
    box = np.array([
        [4, 0, 0],
        [2, 2, 0],
        [1, 1, 1],
    ])
    context.setPositions(coord * unit.angstrom)
    context.setPeriodicBoxVectors(*(box * unit.angstrom))
    
    if multi_state:
        ref_atoms = struc.from_template(
            template,
            np.stack([coord] * 2),
            np.stack([box] * 2)
        )
    else:
        ref_atoms = template.copy()
        ref_atoms.coord = coord
        ref_atoms.box = box

    if multi_state:
        states = [context.getState(getPositions=True) for _ in range(2)]
        test_atoms = molmarbles.from_states(template, states)
    else:
        state = context.getState(getPositions=True)
        test_atoms = molmarbles.from_state(template, state)
    
    assert np.allclose(test_atoms.coord, ref_atoms.coord)
    assert np.allclose(test_atoms.box, ref_atoms.box)