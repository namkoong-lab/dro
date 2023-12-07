from .src import * 


def fetch_method(method, is_regression, input_dim=77):
    if method == 'unified_dro_l2':
        if is_regression == 1 or is_regression == 2:
            raise NotImplementedError("Unified DRO does not support regression!")
        else:
            return MOT_Robust_CLF_L2()
    if method == 'unified_dro_linf':
        if is_regression == 1 or is_regression == 2:
            raise NotImplementedError("Unified DRO does not support regression!")
        else:
            return MOT_Robust_CLF_Linf()
    elif method == 'hr_dro_lr':
        return HR_DRO_LR(is_regression=is_regression)
    elif method == 'chi2_dro':
        return chi2_DRO(input_dim=input_dim, is_regression=is_regression)
    elif method == 'kl_dro':
        return KL_DRO(input_dim=input_dim, is_regression=is_regression)
    elif method == 'tv_dro':
        return TV_DRO(input_dim=input_dim, is_regression=is_regression)
    elif method == 'marginal_cvar_dro':
        return Marginal_CVaR_DRO(input_dim=input_dim, is_regression=is_regression)
    elif method == 'cvar_dro':
        return CVaR_DRO(input_dim=input_dim, is_regression=is_regression)
    elif method == 'wasserstein_dro':
        return Wasserstein_DRO(input_dim=input_dim, is_regression=is_regression)
    elif method == 'wasserstein_dro_satisficing':
        return Wasserstein_DRO_satisficing(input_dim=input_dim, is_regression=is_regression)
    elif method == 'wasserstein_dro_aug':
        return Wasserstein_DRO_aug(input_dim=input_dim, is_regression=is_regression)
    elif method == 'sinkhorn_dro':
        return Sinkhorn_DRO_Linear(input_dim=input_dim, is_regression=is_regression)
    elif method == 'mmd_dro':
        return MMD_DRO(input_dim=input_dim, is_regression=is_regression)
    else:
        raise NotImplementedError(f"Algorithm {method} not implemented yet!")
