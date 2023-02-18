from os.path import join, dirname, realpath
import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
import openmm.app as app
import molmarbles


TEST_FILE = join(dirname(realpath(__file__)), "1aki.cif")


def test_system_consistency():
    """
    Test whether an :class:`System` converted from a
    :class:`AtomArray` equal to a :class:`System`
    directly read via OpenMM.
    Forces and constraints are not tested, as they are not set by
    :func:`to_system()`.
    """
    topology = app.PDBxFile(TEST_FILE).topology
    force_field = app.ForceField('amber14-all.xml')
    ref_system = force_field.createSystem(topology)

    atoms = pdbx.get_structure(
        pdbx.PDBxFile.read(TEST_FILE), model=1
    )
    test_system = molmarbles.to_system(atoms)

    assert test_system.getNumParticles() == ref_system.getNumParticles()
    test_masses = [
        test_system.getParticleMass(i)
        for i in range(test_system.getNumParticles())
    ]
    ref_masses = [
        ref_system.getParticleMass(i)
        for i in range(ref_system.getNumParticles())
    ]
    assert test_masses == ref_masses
    assert test_system.getDefaultPeriodicBoxVectors() \
        ==  ref_system.getDefaultPeriodicBoxVectors()