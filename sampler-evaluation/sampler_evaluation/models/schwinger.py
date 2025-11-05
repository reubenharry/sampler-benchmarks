from functools import partial
import sys
sys.path.append(".")  
sys.path.append("../../blackjax")
sys.path.append("../../sampler-benchmarks/sampler-comparison")
sys.path.append("../../sampler-benchmarks/sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
# print(os.listdir("../../src/inference-gym/spinoffs/inference_gym"))


import jax

import jax.numpy as jnp
from sampler_evaluation.models.model import make_model
import pickle
import os
from sampler_evaluation.models.model import SampleTransformation, make_model
from sampler_evaluation.models.u1 import U1
from sampler_evaluation.models.lft_lib import LFT
import h5py 
import jax.scipy as jsp

module_dir = os.path.dirname(os.path.abspath(__file__))


def Schwinger(Lt, Lx, beta= 1., m = -.188, load_from_file=False):
    """Args:
            lattice size = (Lt, Lx)
            beta: inverse temperature
            mass: renormalized
    """

    assert(Lt == Lx)
    Lxy = Lx
    if load_from_file:  
        with open(
            f"{module_dir}/data/U1_Lt{Lt}_Lx{Lx}_beta{beta}"+"_expectations.pkl",
            "rb",
        ) as f:
            stats = pickle.load(f)
        e_x_top_charge = jnp.asarray(stats["top_charge"])
        e_x2_top_charge = jnp.asarray(stats["top_charge_square"])
        e_std_top_charge = jnp.sqrt(e_x2_top_charge - e_x_top_charge**2)
        e_x_ppbs = jnp.asarray(stats["ppbs"])
        e_x2_ppbs = jnp.asarray(stats["ppbs_square"])
        e_std_ppbs = jnp.sqrt(e_x2_ppbs - e_x_ppbs**2)

    else:
        e_x_top_charge = 0.
        e_x2_top_charge = 0.
        e_std_top_charge = 0.
        e_x_ppbs = 0.
        e_x2_ppbs = 0.
        e_std_ppbs = 0.


    ndims = 2 * Lt*Lx
    unflatten = lambda links_flattened: links_flattened.reshape(2, Lt, Lx)
    locs = jnp.array([[i//Lx, i%Lx] for i in range(Lt*Lx)]) #the set of all possible lattice sites

    sample_init = lambda key: 2 * jnp.pi * jax.random.uniform(key, shape = (ndims, ))

    shape = (2, Lt, Lx)
    d = ndims
    blank_field = jnp.zeros((d,),dtype='complex64')
    zeros = jnp.zeros(shape,dtype='complex64')
    upSource = zeros.at[0,0,0].set(1.).flatten()
    downSource = zeros.at[1,0,0].set(1.).flatten()
    ones = jnp.ones(shape, dtype='complex64')
    """ Boundary conditions pertain to which row will cross voer from the other side of time. Ie rolling fermion array either +1 (B) or -1 (F) in time """
    B_BC = ones.at[:,:,[0]].multiply(-1)
    F_BC = ones.at[:,:,[-1]].multiply(-1)
    pauli0 = jnp.array([[1,0], [0,1]], dtype='complex64'),
    pauli1 = jnp.array([[0, 1], [1, 0]], dtype='complex64'),
    pauli2 = jnp.array([[0, -1j], [1j, 0]], dtype='complex64'),
    pauli3 = jnp.array([[1, 0], [0, -1]], dtype='complex64')
    utils = LFT(Lt,Lx,beta=beta,m=m)

    def g3psi(psi):
        psi = psi.reshape(shape)
        return jnp.stack((psi[0,:,:],-1.*psi[1,:,:])).flatten()
    
    def g2psi(psi):
        psi = psi.reshape(shape)
        return jnp.stack((-1.j*psi[1,:,:],1.j*psi[0,:,:])).flatten()
    
    def g1psi(psi):
        psi = psi.reshape(shape)
        return jnp.stack((psi[1,:,:],psi[0,:,:])).flatten()


    def gauge_logdensity(links):
        """Equation 27 in reference [1]"""
        action_density = jnp.cos(utils.plaquette(utils.unflatten(links)))
        return beta * jnp.sum(action_density)


    def Delpsi( psi, gauge_field, mu):
        """mu MUST be 1,2, (mu-1) is gauge axis direction"""
        g_mu = mu - 1
        psi = psi.reshape(shape)
        U_mu = jnp.exp(1.j * jnp.reshape(gauge_field,shape))[g_mu,:,:]
        bc = jax.lax.cond(mu == 2, lambda : F_BC, lambda : ones)
        result = U_mu*jnp.roll(psi,-1,axis=mu) * bc - psi 
        return result.flatten()

    def DelStarpsi( psi, gauge_field, mu):
        """mu MUST be 1,2, (mu-1) is gauge axis direction"""
        g_mu = mu - 1
        psi = psi.reshape(shape)
        U_mu = jnp.exp(1.j * jnp.reshape(gauge_field,shape))[g_mu,:,:]
        bc = jax.lax.cond(mu == 2, lambda : B_BC, lambda : ones)
        result = psi - jnp.roll(U_mu,1,axis=g_mu).conj() * jnp.roll(psi,1,axis=mu) * bc
        return result.flatten()


    def Dpsi( psi, gauge_field):
        """ Refer to https://arxiv.org/pdf/hep-lat/0101015.pdf for details of this Wilson Dirac operator """
        sig1 = 0.5 * (g1psi(Delpsi(psi,gauge_field,1)+DelStarpsi(psi,gauge_field,1)))
        sig2 = 0.5 * (g2psi(Delpsi(psi,gauge_field,2)+DelStarpsi(psi,gauge_field,2)))
        
        cross1 = 0.5 * DelStarpsi(Delpsi(psi,gauge_field,1),gauge_field,1)
        cross2 = 0.5 * DelStarpsi(Delpsi(psi,gauge_field,2),gauge_field,2)
        
        return (sig1 + sig2 - cross1 - cross2)
    
    def Ddagpsi(psi,gauge_field):
        
        psi_p = g3psi(psi)
        Dpsi_p = Dpsi(psi_p,gauge_field)

        return g3psi(Dpsi_p)

    Mpsi = lambda  psi, gauge_field : Dpsi(psi,gauge_field) + m*psi
    
    Mdagpsi = lambda  psi, gauge_field : Ddagpsi(psi,gauge_field) + m*psi

    MdagMpsi = lambda  psi, gauge_field: Mdagpsi(Mpsi(psi,gauge_field),gauge_field)

    def invMdagMsolve(phi,gauge_field):

        A = lambda psi: MdagMpsi(psi,gauge_field)
        # phitemp = jaxopt.linear_solve.solve_cg(A,phi)
        phitemp, _ = jsp.sparse.linalg.cg(A,phi)
        return phitemp

    def construct_op( op, gauge_field):
        get_diag = lambda k : op(jnp.roll(upSource,k),gauge_field)
        diag = jax.lax.map(get_diag, jnp.arange(d))
        return diag.T

    def ppb_sign(gauge_field):
        
        M = construct_op(Mpsi,gauge_field)
        sgn, _ = jax.numpy.linalg.slogdet(M)
        ans = jnp.trace(jsp.linalg.inv(M)) / (Lxy**2)

        return (ans,jnp.sign(sgn))

    def ferm_action(gauge_field):

        # jax.debug.print("ferm action: {x}", x=gauge_field[0])

        MdagM = construct_op(MdagMpsi,gauge_field)
        ans = jnp.linalg.slogdet(MdagM)

        return -jnp.real(ans[1])

    def ferm_pf_action(gauge_field,pf):

        A = lambda psi: MdagMpsi(psi,gauge_field)
        phitemp, _ = jsp.sparse.linalg.cg(A,pf)
        dot = jnp.sum(pf.conj()*phitemp)

        return -dot

    def action(gauge_field):
        jax.debug.print("action: {x}", x=gauge_field[0])

        return gauge_logdensity(gauge_field) + ferm_action(gauge_field)

    def action_pf(gauge_field,pf):

        return gauge_logdensity(gauge_field) + ferm_pf_action(gauge_field,pf)

    return make_model(
        logdensity_fn=action,
        ndims=ndims,
        default_event_space_bijector=lambda x:x,
        sample_transformations = {
        'top_charge':SampleTransformation(
            fn=utils.top_charge,
            ground_truth_mean=e_x_top_charge,
            ground_truth_standard_deviation=e_std_top_charge,
        ),
        'ppb_sign':SampleTransformation(
            fn=ppb_sign,
            ground_truth_mean=e_x2_ppbs,
            ground_truth_standard_deviation=jnp.nan,
        ),
            
        },
        exact_sample=None,
        sample_init=sample_init,
        name=f'Schwinger_Lt{Lt}_Lx{Lx}_beta{beta}_m{m}',
    )


if __name__ == "__main__":
    model = Schwinger(Lt=16, Lx=16, beta=6,load_from_file=False)

    from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc

    sampler = partial(unadjusted_mclmc,num_tuning_steps=100, desired_energy_var=5e-4, diagonal_preconditioning=True, integrator_type='mclachlan')

    # run unadjusted_mclmc
    samples = sampler()(model=model,
        num_steps=100,
        initial_position=jnp.ones(model.ndims),
        key=jax.random.PRNGKey(0),
    )
    # print(samples.shape)
    
    # print(model.sample_transformations)