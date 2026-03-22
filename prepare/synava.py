import os
import math
import gzip
import pickle
import os.path as op
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from config import set_working_directory,FRAGSCORE_FILE

set_working_directory()
class SaScore:
    def __init__(self):
        self.readFragmentScores()

    def readFragmentScores(self):
        data = pickle.load(gzip.open(FRAGSCORE_FILE))
        outDict = {}
        for i in data:
            for j in range(1, len(i)):
                outDict[i[j]] = float(i[0])
        self._fscores = outDict

    @staticmethod
    def numBridgeheadsAndSpiro(mol):
        nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        return nBridgehead, nSpiro

    def calculateScore(self, mol):
        # Fragment score
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)  # 2 is the radius of the circular fingerprint
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += self._fscores.get(sfp, -4) * v
        score1 /= nf

        # Features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nBridgeheads, nSpiro = SaScore.numBridgeheadsAndSpiro(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.
        # ---------------------------------------
        # This differs from the paper, which defines: macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0
        return sascore

    def __call__(self, mol):
        s = self.calculateScore(mol)
        return s

