"""
Multi-view data integration and denoising methods:
- AJIVEReconstructor: AJIVE-based joint/individual recovery.
- MCCAJointIndividual: MCCA-based joint/individual decomposition.
- GCCAJointIndividual: GCCA-based decomposition.
- DISCO_SCA: DISCO-SCA via R multiblock package.
- MFAJointIndividual: Multiple Factor Analysis joint/individual recovery.
- HPCA: Hierarchical PCA denoising pipeline.
"""

import numpy as np
from mvlearn.decomposition import AJIVE
from mvlearn.embed import MCCA
from mvlearn.embed import GCCA
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from rpy2.rinterface_lib.sexp import NULLType
from sklearn.preprocessing import StandardScaler
from typing import List

import pandas as pd
import prince
from rpy2 import robjects
from rpy2.robjects import numpy2ri, r, ListVector
from rpy2.robjects.packages import importr
from rpy2.robjects import r as R
from rpy2.robjects import pandas2ri
from rpy2.rinterface_lib.embedded import RRuntimeError

# Activate automatic conversion between R and numpy
numpy2ri.activate()


# Import the multiblock package
multiblock = importr('multiblock')


# Using AJIVE to reconstruct the matrices
class AJIVEReconstructor:
    def __init__(self, rank_list, joint_rank):
        """
        Parameters
        ----------
        rank_list : list of int
            Number of leading individual components to recover for each view.
        joint_rank : int
            Assumed rank of the joint structure across views.
        """
        self.rank_list = rank_list
        self.joint_rank = joint_rank
        self.ajive_model = AJIVE(init_signal_ranks=rank_list, joint_rank=joint_rank)
        self.joint_scores_ = None

        self.reconstructions_ = {}      # X_approx per view
        self.U_denoised_ = {}           # denoised scores (left singular vectors)
        self.V_denoised_ = {}           # denoised loadings (right singular vectors)

    def fit(self, X_list):
        """
        Fit AJIVE and compute denoised low-rank reconstructions per view.

        Parameters
        ----------
        X_list : list of ndarray
            List of input views, shape (n_samples, p_k)

        Returns
        -------
        U_denoised_ : dict
            Dictionary mapping view index to denoised left singular vectors (scores).
        V_denoised_ : dict
            Dictionary mapping view index to denoised right singular vectors (loadings).
        """
        self.ajive_model.fit(X_list)
        self.joint_scores_ = self.ajive_model.joint_scores_

        for i, X in enumerate(X_list):
            # compute joint component
            J_i = self.joint_scores_ @ self.joint_scores_.T @ X

            # compute individual component
            X_resid = X - J_i
            r_i = self.rank_list[i]
            U_i, S_i, Vt_i = svds(X_resid, k=r_i)
            idx = np.argsort(-S_i)
            U_i, S_i, Vt_i = U_i[:, idx], S_i[idx], Vt_i[idx, :]
            I_i = U_i @ np.diag(S_i) @ Vt_i

            # compute total reconstruction
            X_approx = J_i + I_i
            self.reconstructions_[i] = X_approx

            # compute final denoised SVD
            U_approx, _, Vt_approx = np.linalg.svd(X_approx, full_matrices=False)
            self.U_denoised_[i] = U_approx[:, :r_i] * np.sqrt(X_approx.shape[0])
            self.V_denoised_[i] = Vt_approx.T[:, :r_i] * np.sqrt(X_approx.shape[1])

        return self.U_denoised_, self.V_denoised_


class MCCAJointIndividual:
    def __init__(self, individual_ranks, joint_rank, regs=1e-4):
        self.joint_rank = joint_rank
        self.individual_ranks = individual_ranks
        self.mcca = MCCA(n_components=joint_rank, regs=regs)

        # Outputs
        self.joint_scores_ = None
        self.joint_loadings_ = None
        self.individual_scores_ = []
        self.individual_loadings_ = []
        self.reconstructions_ = {}
        self.U_denoised_ = {}
        self.V_denoised_ = {}

    def fit(self, Xs):
        self.mcca.fit(Xs)
        self.joint_scores_ = self.mcca.transform(Xs)  # list of (n, joint_rank)
        self.joint_loadings_ = self.mcca.loadings_     # list of (p_i, joint_rank)

        for i, X in enumerate(Xs):
            r_i = self.individual_ranks[i]

            # compute joint component
            J_i = self.joint_scores_[i] @ self.joint_loadings_[i].T

            # compute individual component
            R_i = X - J_i
            if r_i > 0:
                U_indiv, S_indiv, Vt_indiv = svds(R_i, k=r_i)
                idx = np.argsort(-S_indiv)
                U_indiv = U_indiv[:, idx]
                S_indiv = S_indiv[idx]
                Vt_indiv = Vt_indiv[idx, :]

                I_i = U_indiv @ np.diag(S_indiv) @ Vt_indiv
                self.individual_scores_.append(U_indiv @ np.diag(S_indiv))
                self.individual_loadings_.append(Vt_indiv.T)
            else:
                I_i = np.zeros_like(X)
                self.individual_scores_.append(np.zeros((X.shape[0], 0)))
                self.individual_loadings_.append(np.zeros((X.shape[1], 0)))

            # compute total reconstruction
            X_approx = J_i + I_i
            self.reconstructions_[i] = X_approx

            # compute final denoised SVD
            U_approx, _, Vt_approx = np.linalg.svd(X_approx, full_matrices=False)
            self.U_denoised_[i] = U_approx[:, :r_i] * np.sqrt(X_approx.shape[0])
            self.V_denoised_[i] = Vt_approx.T[:, :r_i] * np.sqrt(X_approx.shape[1])

        return self.U_denoised_, self.V_denoised_

    def transform(self, Xs):
        return self.mcca.transform(Xs)

    def get_individual_components(self):
        return self.individual_scores_, self.individual_loadings_

    def get_denoised_factors(self):
        return self.U_denoised_, self.V_denoised_

    def get_reconstructions(self):
        return self.reconstructions_

class GCCAJointIndividual:
    def __init__(self, individual_ranks, joint_rank):
        self.joint_rank = joint_rank
        self.individual_ranks = individual_ranks
        self.gcca = GCCA(n_components=joint_rank)

        # Outputs
        self.joint_scores_ = None
        self.joint_loadings_ = None
        self.individual_scores_ = []
        self.individual_loadings_ = []
        self.reconstructions_ = {}
        self.U_denoised_ = {}
        self.V_denoised_ = {}

    def fit(self, Xs):
        self.gcca.fit(Xs)
        self.joint_scores_ = self.gcca.transform(Xs)  # list of (n, joint_rank)
        self.joint_loadings_ = [None] * len(Xs)       # initialize properly

        for i, X in enumerate(Xs):
            r_i = self.individual_ranks[i]
            # compute joint component
            self.joint_loadings_[i] = self.joint_scores_[i].T @ X
            J_i = self.joint_scores_[i] @ self.joint_scores_[i].T @ X

            # compute individual component
            R_i = X - J_i
            if r_i > 0:
                U_indiv, S_indiv, Vt_indiv = svds(R_i, k=r_i)
                idx = np.argsort(-S_indiv)
                U_indiv = U_indiv[:, idx]
                S_indiv = S_indiv[idx]
                Vt_indiv = Vt_indiv[idx, :]

                I_i = U_indiv @ np.diag(S_indiv) @ Vt_indiv
                self.individual_scores_.append(U_indiv @ np.diag(S_indiv))
                self.individual_loadings_.append(Vt_indiv.T)
            else:
                I_i = np.zeros_like(X)
                self.individual_scores_.append(np.zeros((X.shape[0], 0)))
                self.individual_loadings_.append(np.zeros((X.shape[1], 0)))

            # compute total reconstruction
            X_approx = J_i + I_i
            self.reconstructions_[i] = X_approx

            # compute final denoised SVD
            U_approx, _, Vt_approx = np.linalg.svd(X_approx, full_matrices=False)
            self.U_denoised_[i] = U_approx[:, :r_i] * np.sqrt(X_approx.shape[0])
            self.V_denoised_[i] = Vt_approx.T[:, :r_i] * np.sqrt(X_approx.shape[1])

        return self.U_denoised_, self.V_denoised_

    def transform(self, Xs):
        return self.gcca.transform(Xs)

    def get_individual_components(self):
        return self.individual_scores_, self.individual_loadings_

    def get_denoised_factors(self):
        return self.U_denoised_, self.V_denoised_

    def get_reconstructions(self):
        return self.reconstructions_

class DISCO_SCA:
    def __init__(self, n_components, individual_ranks):
        self.n_components = n_components
        self.individual_ranks = individual_ranks
        self.T_scores_ = None
        self.P_loadings_ = None
        self.comdist_ = None
        self.propExp_component_ = None

        self.individual_scores_ = []
        self.individual_loadings_ = []
        self.reconstructions_ = {}
        self.U_denoised_ = {}
        self.V_denoised_ = {}

    def fit(self, Xs):

        # Step 1: Reduce dimensionality via SVD
        Xs_reduced = []
        for X, r in zip(Xs, self.individual_ranks):
            svd = TruncatedSVD(n_components=r)
            X_reduced = svd.fit_transform(X) @ svd.components_
            Xs_reduced.append(X_reduced)

        # Concatenate data blocks horizontally
        DATA = np.hstack(Xs_reduced)
        Jk = [X.shape[1] for X in Xs_reduced]

        # Convert to R objects
        DATA_r = robjects.r.matrix(DATA, nrow=DATA.shape[0], ncol=DATA.shape[1])
        Jk_r = robjects.IntVector(Jk)

        # Run DISCOsca
        disco_result = multiblock.DISCOsca(DATA_r, max(self.n_components, 2), Jk_r)

        # Extract scores/loadings and comdist matrix
        self.T_scores_ = np.squeeze(np.array(disco_result.rx2('Trot_best')), axis=0)
        self.P_loadings_ = np.squeeze(np.array(disco_result.rx2('Prot_best')), axis=0)
        self.comdist_ = np.array(disco_result.rx2('comdist'))
        self.propExp_component_ = np.array(disco_result.rx2('propExp_component'))

        # Recover block-wise matrices
        col_splits = np.cumsum(Jk)[:-1]
        P_blocks = np.split(self.P_loadings_, col_splits, axis=0)

        for i, X in enumerate(Xs_reduced):
            r_i = self.individual_ranks[i]
            joint_idx = np.where(np.all(self.comdist_ == 1, axis=0))[0][:self.n_components]
            P_joint = P_blocks[i][:, joint_idx]
            if P_joint.ndim == 1:
               P_joint = P_joint.reshape(-1, 1)
            T_joint = self.T_scores_[:, joint_idx]
            if T_joint.ndim == 1:
               T_joint = T_joint.reshape(-1, 1)

            # compute joint component
            J_i = T_joint @ P_joint.T

            # compute individual component
            R_i = X - J_i
            if r_i > 0:
                U_indiv, S_indiv, Vt_indiv = svds(R_i, k=r_i)
                idx = np.argsort(-S_indiv)
                U_indiv = U_indiv[:, idx]
                S_indiv = S_indiv[idx]
                Vt_indiv = Vt_indiv[idx, :]

                I_i = U_indiv @ np.diag(S_indiv) @ Vt_indiv
                self.individual_scores_.append(U_indiv @ np.diag(S_indiv))
                self.individual_loadings_.append(Vt_indiv.T)
            else:
                I_i = np.zeros_like(X)
                self.individual_scores_.append(np.zeros((X.shape[0], 0)))
                self.individual_loadings_.append(np.zeros((X.shape[1], 0)))

            # compute total reconstruction
            X_approx = J_i + I_i
            self.reconstructions_[i] = X_approx

            # compute final denoised SVD
            U_approx, _, Vt_approx = np.linalg.svd(X_approx, full_matrices=False)
            self.U_denoised_[i] = U_approx[:, :r_i] * np.sqrt(X_approx.shape[0])
            self.V_denoised_[i] = Vt_approx.T[:, :r_i] * np.sqrt(X_approx.shape[1])

        return self.U_denoised_, self.V_denoised_

    def get_joint_components(self):
        joint_indices = np.where(np.all(self.comdist_ == 1, axis=0))[0]
        return self.T_scores_[:, joint_indices], self.P_loadings_[:, joint_indices]

    def get_individual_components(self):
        return self.individual_scores_, self.individual_loadings_

    def get_denoised_factors(self):
        return self.U_denoised_, self.V_denoised_

    def get_reconstructions(self):
        return self.reconstructions_

class MFAJointIndividual:
    def __init__(self, individual_ranks, joint_rank):
        """
        individual_ranks : list of int
            Number of individual components to keep for each block.
        joint_rank : int
            Number of shared components.
        """
        self.individual_ranks = individual_ranks
        self.joint_rank = joint_rank

        # Will be filled by fit:
        self.mfa_model = None
        self.joint_scores_ = None       # (n_samples, joint_rank)
        self.joint_loadings_ = None     # (sum(p_k), joint_rank)
        self.U_denoised_ = {}           # dict block_idx -> (n_samples, r_k)
        self.V_denoised_ = {}           # dict block_idx -> (p_k, r_k)
        self.reconstructions_ = {}      # dict block_idx -> (n_samples, p_k)
        self.individual_scores_ = []    # list of (n_samples, r_k)
        self.individual_loadings_ = []  # list of (p_k, r_k)

    def fit(self, Xs):
        """
        Xs : list of numpy arrays, each shape (n_samples, p_k)
        """
        # --- 1) Standardize each block and build a MultiIndex DataFrame
        scalers = [StandardScaler().fit(X) for X in Xs]
        dfs = []
        for i, (X, scaler) in enumerate(zip(Xs, scalers), start=1):
            X_std = scaler.transform(X)
            mi = pd.MultiIndex.from_product([[f"Block{i}"], list(range(X.shape[1]))])
            dfs.append(pd.DataFrame(X_std, columns=mi))
        X_concat = pd.concat(dfs, axis=1)

        # --- 2) Fit prince.MFA
        group_names = [f"Block{i}" for i in range(1, len(Xs)+1)]
        self.mfa_model = prince.MFA(
            n_components=self.joint_rank,
            random_state=0
        ).fit(X_concat, groups=group_names)

        # --- 3) Extract joint scores & loadings
        self.joint_scores_ = self.mfa_model.row_coordinates(X_concat).values  # (n, joint_rank)
        corr_df = self.mfa_model.column_correlations  # DataFrame (sum(p_k), joint_rank)
        self.joint_loadings_ = corr_df.values

        # --- 4) Split joint loadings by block
        pks = [X.shape[1] for X in Xs]
        splits = np.cumsum(pks)[:-1]
        loadings_blocks = np.split(self.joint_loadings_, splits, axis=0)

        # --- 5) For each block: reconstruct joint part, then individual via SVD
        for i, (X, r_i, L_block, scaler) in enumerate(zip(Xs, self.individual_ranks, loadings_blocks, scalers)):
            # compute joint component
            J_i = self.joint_scores_ @ L_block.T  # (n, joint_rank) @ (joint_rank, p_k) -> (n, p_k)
            X_std = scaler.transform(X)
            # compute individual component
            R_i = X_std - J_i

            # compute individual component
            if r_i > 0:
                U_full, S_full, Vt_full = np.linalg.svd(R_i, full_matrices=False)
                U_ind = U_full[:, :r_i]               # (n, r_i)
                S_ind = S_full[:r_i]                  # (r_i,)
                V_ind = Vt_full[:r_i, :].T            # (p_k, r_i)
                I_i = U_ind @ np.diag(S_ind) @ Vt_full[:r_i, :]
                self.individual_scores_.append(U_ind @ np.diag(S_ind))
                self.individual_loadings_.append(V_ind)
            else:
                I_i = np.zeros_like(R_i)
                self.individual_scores_.append(np.zeros((X.shape[0], 0)))
                self.individual_loadings_.append(np.zeros((X.shape[1], 0)))

            # compute total reconstruction
            X_approx = J_i + I_i
            self.reconstructions_[i] = X_approx

            # compute final denoised SVD
            if r_i > 0:
                U_f, S_f, Vt_f = svds(X_approx, k=r_i)
                # reorder in descending order
                idx = np.argsort(S_f)[::-1]
                U_f = U_f[:, idx]
                Vt_f = Vt_f[idx, :]
                U_den = U_f                          # (n, r_i)
                V_den = Vt_f.T                       # (p_k, r_i)
            else:
                U_den = np.zeros((X.shape[0], 0))
                V_den = np.zeros((X.shape[1], 0))

            self.U_denoised_[i] = U_den * np.sqrt(X_approx.shape[0])
            self.V_denoised_[i] = V_den * np.sqrt(X_approx.shape[1])

        return self.U_denoised_, self.V_denoised_


# Using HPCA to denoise   
class HPCA:
    def __init__(self, joint_rank, individual_ranks):
        """
        joint_rank : int
            number of global (shared) components
        individual_ranks : list of int, length = number of blocks
            number of local components per block
        """
        self.joint_rank = joint_rank
        self.individual_ranks = individual_ranks

        # to be filled in .fit()
        self.scalers_ = None
        self.global_svd_ = None
        self.U_denoised_ = {}      # block_idx -> (n_samples, r_block)
        self.V_denoised_ = {}      # block_idx -> (p_block, r_block)
        self.reconstructions_ = {} # block_idx -> (n_samples, p_block)

    def fit(self, Xs):
        """
        Xs : list of np.ndarray, each shape (n_samples, p_k)
        """
        # 1) standardize each block
        self.scalers_ = [StandardScaler().fit(X) for X in Xs]
        Xs_std = [scaler.transform(X) for scaler, X in zip(self.scalers_, Xs)]

        # 2) global PCA on concatenated data
        X_concat = np.hstack(Xs_std)
        self.global_svd_ = TruncatedSVD(n_components=self.joint_rank)
        Z = self.global_svd_.fit_transform(X_concat)                 # (n, joint_rank)
        L = self.global_svd_.components_.T                           # (sum p_k, joint_rank)

        # 3) split the global loadings L by block
        ps = [X.shape[1] for X in Xs_std]
        splits = np.cumsum(ps)[:-1]
        L_blocks = np.split(L, splits, axis=0)  # list of (p_k, joint_rank)

        # 4) for each block, reconstruct joint part, then local PCA on residual
        for i, (X_std, r_i, Lk) in enumerate(zip(Xs_std, self.individual_ranks, L_blocks)):
            # compute joint component
            Jk = Z @ Lk.T                    # (n, p_k)

            # compute individual component
            Rk = X_std - Jk

            # compute individual component
            if r_i > 0:
                U_loc, S_loc, Vt_loc = svds(Rk, k=r_i)
                # sort descending
                idx = np.argsort(-S_loc)
                U_loc, S_loc, Vt_loc = U_loc[:, idx], S_loc[idx], Vt_loc[idx, :]
                Ik = U_loc @ np.diag(S_loc) @ Vt_loc
            else:
                # no local components
                Ik = np.zeros_like(Rk)

            # compute total reconstruction
            Xk_approx = Jk + Ik
            self.reconstructions_[i] = Xk_approx

            # compute final denoised SVD
            # note: if r_i==0 we still return empty arrays
            if r_i > 0:
                U_f, S_f, Vt_f = svds(Xk_approx, k=r_i)
                idx = np.argsort(-S_f)
                U_f, S_f, Vt_f = U_f[:, idx], S_f[idx], Vt_f[idx, :]
                U_den = U_f * np.sqrt(Xk_approx.shape[0])   # normalize if desired
                V_den = (Vt_f.T * np.sqrt(Xk_approx.shape[1]))
            else:
                U_den = np.zeros((Xk_approx.shape[0], 0))
                V_den = np.zeros((Xk_approx.shape[1], 0))

            self.U_denoised_[i] = U_den
            self.V_denoised_[i] = V_den

        return self

    def get_denoised_factors(self):
        """
        Returns two dicts mapping block index -> (U_denoised, V_denoised)
        """
        return self.U_denoised_, self.V_denoised_

    def get_reconstructions(self):
        """
        Returns block index -> reconstructed (joint+individual) matrix
        """
        return self.reconstructions_
