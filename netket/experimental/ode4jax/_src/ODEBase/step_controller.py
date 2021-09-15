

@struct.dataclass
def PIController(AbstractController):
	beta1 
	beta2

	def stepsize_controller()




abstract type AbstractController end

@inline function stepsize_controller!(integrator, alg)
  stepsize_controller!(integrator, integrator.opts.controller, alg)
end

# checks whether the controller should accept a step based on the error estimate
@inline function accept_step_controller(integrator, controller::AbstractController)
  return integrator.EEst <= 1
end

@inline function step_accept_controller!(integrator, alg, q)
  step_accept_controller!(integrator, integrator.opts.controller, alg, q)
end

@inline function step_reject_controller!(integrator, alg)
  step_reject_controller!(integrator, integrator.opts.controller, alg)
end

reset_alg_dependent_opts!(controller::AbstractController, alg1, alg2) = nothing

DiffEqBase.reinit!(integrator::ODEIntegrator, controller::AbstractController) = nothing


# Standard integral (I) step size controller
"""
    IController()

The standard (integral) controller is the most basic step size controller.
This controller is usually the first one introduced in numerical analysis classes
but should only be used rarely in practice because of efficiency problems for
many problems/algorithms.

Construct an integral (I) step size controller adapting the time step
based on the formula
```
Δtₙ₊₁ = εₙ₊₁^(1/k) * Δtₙ
```
where `k = get_current_adaptive_order(alg, integrator.cache) + 1` and `εᵢ` is the
inverse of the error estimate `integrator.EEst` scaled by the tolerance
(Hairer, Nørsett, Wanner, 2008, Section II.4).
The step size factor is multiplied by the safety factor `gamma` and clipped to
the interval `[qmin, qmax]`.
A step will be accepted whenever the estimated error `integrator.EEst` is
less than or equal to unity. Otherwise, the step is rejected and re-tried with
the predicted step size.

## References
- Hairer, Nørsett, Wanner (2008)
  Solving Ordinary Differential Equations I Nonstiff Problems
  [DOI: 10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1)
"""
struct IController <: AbstractController
end

@inline function stepsize_controller!(integrator, controller::IController, alg)
  @unpack qmin, qmax, gamma = integrator.opts
  EEst = DiffEqBase.value(integrator.EEst)

  if iszero(EEst)
    q = inv(qmax)
  else
    expo = 1 / (get_current_adaptive_order(alg, integrator.cache) + 1)
    qtmp = DiffEqBase.fastpow(EEst, expo) / gamma
    @fastmath q = DiffEqBase.value(max(inv(qmax), min(inv(qmin), qtmp)))
    # TODO: Shouldn't this be in `step_accept_controller!` as for the PI controller?
    integrator.qold = DiffEqBase.value(integrator.dt) / q
  end
  q
end

function step_accept_controller!(integrator, controller::IController, alg, q)
  @unpack qsteady_min, qsteady_max = integrator.opts

  if qsteady_min <= q <= qsteady_max
    q = one(q)
  end
  integrator.dt / q # new dt
end

function step_reject_controller!(integrator, controller::IController, alg)
  @unpack qold = integrator
  integrator.dt = qold
end


# PI step size controller
"""
    PIController(beta1, beta2)

The proportional-integral (PI) controller is a widespread step size controller
with improved stability properties compared to the [`IController`](@ref).
This controller is the default for most algorithms in OrdinaryDiffEq.jl.

Construct a PI step size controller adapting the time step based on the formula
```
Δtₙ₊₁ = εₙ₊₁^β₁ * εₙ^β₂ * Δtₙ
```
where `εᵢ` are inverses of the error estimates scaled by the tolerance
(Hairer, Nørsett, Wanner, 2010, Section IV.2).
The step size factor is multiplied by the safety factor `gamma` and clipped to
the interval `[qmin, qmax]`.
A step will be accepted whenever the estimated error `integrator.EEst` is
less than or equal to unity. Otherwise, the step is rejected and re-tried with
the predicted step size.

!!! note
    The coefficients `beta1, beta2` are not scaled by the order of the method,
    in contrast to the [`PIDController`](@ref). For the `PIController`, this
    scaling by the order must be done when the controller is constructed.

## References
- Hairer, Nørsett, Wanner (2010)
  Solving Ordinary Differential Equations II Stiff and Differential-Algebraic Problems
  [DOI: 10.1007/978-3-642-05221-7](https://doi.org/10.1007/978-3-642-05221-7)
- Hairer, Nørsett, Wanner (2008)
  Solving Ordinary Differential Equations I Nonstiff Problems
  [DOI: 10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1)
"""
mutable struct PIController{QT} <: AbstractController
  beta1::QT
  beta2::QT
end

@inline function stepsize_controller!(integrator, controller::PIController, alg)
  @unpack qold = integrator
  @unpack qmin, qmax, gamma = integrator.opts
  @unpack beta1, beta2 = controller
  EEst = DiffEqBase.value(integrator.EEst)

  if iszero(EEst)
    q = inv(qmax)
  else
    q11 = DiffEqBase.fastpow(EEst, float(beta1))
    q = q11 / DiffEqBase.fastpow(qold, float(beta2))
    integrator.q11 = q11
    @fastmath q = max(inv(qmax), min(inv(qmin), q / gamma))
  end
  q
end

function step_accept_controller!(integrator, controller::PIController, alg, q)
  @unpack qsteady_min, qsteady_max, qoldinit = integrator.opts
  EEst = DiffEqBase.value(integrator.EEst)

  if qsteady_min <= q <= qsteady_max
    q = one(q)
  end
  integrator.qold = max(EEst, qoldinit)
  return integrator.dt / q # new dt
end

function step_reject_controller!(integrator, controller::PIController, alg)
  @unpack q11 = integrator
  @unpack qmin, gamma = integrator.opts
  integrator.dt /= min(inv(qmin), q11 / gamma)
end

function reset_alg_dependent_opts!(controller::PIController, alg1, alg2)
  if controller.beta2 == beta2_default(alg1)
    controller.beta2 = beta2_default(alg2)
  end
  if controller.beta1 == beta1_default(alg1, controller.beta2)
    controller.beta1 = beta1_default(alg2, controller.beta2)
  end
end


# PID step size controller
"""
    PIDController(beta1, beta2, beta3=zero(beta1);
                  limiter=default_dt_factor_limiter,
                  accept_safety=0.81)

The proportional-integral-derivative (PID) controller is a generalization of the
[`PIController`](@ref) and can have improved stability and efficiency properties.

Construct a PID step size controller adapting the time step based on the formula
```
Δtₙ₊₁ = εₙ₊₁^(β₁/k) * εₙ^(β₂/k) * εₙ₋₁^(β₃/ k) * Δtₙ
```
where `k = min(alg_order, alg_adaptive_order) + 1` and `εᵢ` are inverses of
the error estimates scaled by the tolerance (Söderlind, 2003).
The step size factor is limited by the `limiter` with default value
```
limiter(x) = one(x) + atan(x - one(x))
```
as proposed by Söderlind and Wang (2006). A step will be accepted whenever the
predicted step size change is bigger than `accept_safety`. Otherwise, the step
is rejected and re-tried with the predicted step size.

Some standard controller parameters suggested in the literature are

| Controller | `beta1` | `beta2` | `beta3` |
|:-----------|--------:|--------:|:-------:|
|    basic   |  `1.00` |  `0.00` |  `0`    |
|    PI42    |  `0.60` | `-0.20` |  `0`    |
|    PI33    |  `2//3` | `-1//3` |  `0`    |
|    PI34    |  `0.70` | `-0.40` |  `0`    |
|   H211PI   |  `1//6` |  `1//6` |  `0`    |
|   H312PID  | `1//18` |  `1//9` | `1//18` |

!!! note
    In contrast to the [`PIController`](@ref), the coefficients `beta1, beta2, beta3`
    are scaled by the order of the method. Thus, standard controllers such as PI42
    can use the same coefficients `beta1, beta2, beta3` for different algorithms.

!!! note
    In contrast to other controllers, the `PIDController` does not use the keyword
    arguments `qmin, qmax` to limit the step size change or the safety factor `gamma`.
    These common keyword arguments are replaced by the `limiter` and `accept_safety`
    to guarantee a smooth behavior (Söderlind and Wang, 2006).
    Because of this, a `PIDController` behaves different from a [`PIController`](@ref),
    even if `beta1, beta2` are adapted accordingly and `iszero(beta3)`.

## References
- Söderlind (2003)
  Digital Filters in Adaptive Time-Stepping
  [DOI: 10.1145/641876.641877](https://doi.org/10.1145/641876.641877)
- Söderlind, Wang (2006)
  Adaptive time-stepping and computational stability
  [DOI: 10.1016/j.cam.2005.03.008](https://doi.org/10.1016/j.cam.2005.03.008)
- Ranocha, Dalcin, Parsani, Ketcheson (2021)
  Optimized Runge-Kutta Methods with Automatic Step Size Control for
  Compressible Computational Fluid Dynamics
  [arXiv:2104.06836](https://arxiv.org/abs/2104.06836)
"""
struct PIDController{QT, Limiter} <: AbstractController
  beta::MVector{3,QT} # controller coefficients
  err ::MVector{3,QT} # history of the error estimates
  accept_safety::QT   # accept a step if the predicted change of the step size
                      # is bigger than this parameter
  limiter::Limiter    # limiter of the dt factor (before clipping)
end

function PIDController(beta1, beta2, beta3=zero(beta1); limiter=default_dt_factor_limiter,
                                                        accept_safety=0.81)
  beta = MVector(map(float, promote(beta1, beta2, beta3))...)
  QT = eltype(beta)
  err = MVector{3,QT}(true, true, true)
  return PIDController(beta, err, convert(QT, accept_safety), limiter)
end

function Base.show(io::IO, controller::PIDController)
  print(io, "PIDController(beta=", controller.beta,
            ", accept_safety=", controller.accept_safety,
            ", limiter=", controller.limiter,
            ")")
end

@inline default_dt_factor_limiter(x) = one(x) + atan(x - one(x))

@inline function stepsize_controller!(integrator, controller::PIDController, alg)
  @unpack qmax = integrator.opts
  beta1, beta2, beta3 = controller.beta
  controller.err[1] = inv(DiffEqBase.value(integrator.EEst))
  err1, err2, err3 = controller.err

  iszero(err1) && return qmax

  k = min(alg_order(alg), alg_adaptive_order(alg)) + 1
  dt_factor = err1^(beta1 / k) * err2^(beta2 / k) * err3^(beta3 / k)
  dt_factor = controller.limiter(dt_factor)
  # Note: No additional limiting of the form
  #   dt_factor = max(qmin, min(qmax, dt_factor))
  # is necessary since the `limiter` should take care of that. The default limiter
  # ensures
  #   0.21 ≈ limiter(0) <= dt_factor <= limiter(Inf) ≈ 2.57
  # See Söderlind, Wang (2006), Section 6.
  integrator.qold = dt_factor
  return dt_factor
end

@inline function accept_step_controller(integrator, controller::PIDController)
  return integrator.qold >= controller.accept_safety
end

function step_accept_controller!(integrator, controller::PIDController, alg, dt_factor)
  @unpack qsteady_min, qsteady_max = integrator.opts

  if qsteady_min <= inv(dt_factor) <= qsteady_max
    dt_factor = one(dt_factor)
  end
  @inbounds begin
    controller.err[3] = controller.err[2]
    controller.err[2] = controller.err[1]
  end
  return integrator.dt * dt_factor # new dt
end

function step_reject_controller!(integrator, controller::PIDController, alg)
  integrator.dt *= integrator.qold
end


# Gustafsson predictive step size controller
"""
    PredictiveController()

The Gustafsson acceleration algorithm accelerates changes so that way algorithms
can more swiftly change to handle quick transients. This algorithm is thus
well-suited for stiff solvers where this can be expected, and is the default
for algorithms like the (E)SDIRK methods.

```julia
gamma = integrator.opts.gamma
niters = integrator.cache.newton_iters
fac = min(gamma,(1+2*integrator.alg.max_newton_iter)*gamma/(niters+2*integrator.alg.max_newton_iter))
expo = 1/(alg_order(integrator.alg)+1)
qtmp = (integrator.EEst^expo)/fac
@fastmath q = max(inv(integrator.opts.qmax),min(inv(integrator.opts.qmin),qtmp))
if q <= integrator.opts.qsteady_max && q >= integrator.opts.qsteady_min
  q = one(q)
end
integrator.qold = q
q
```
In this case, `niters` is the number of Newton iterations which was required
in the most recent step of the algorithm. Note that these values are used
differently depending on acceptance and rejectance. When the step is accepted,
the following logic is applied:
```julia
if integrator.success_iter > 0
  expo = 1/(alg_adaptive_order(integrator.alg)+1)
  qgus=(integrator.dtacc/integrator.dt)*(((integrator.EEst^2)/integrator.erracc)^expo)
  qgus = max(inv(integrator.opts.qmax),min(inv(integrator.opts.qmin),qgus/integrator.opts.gamma))
  qacc=max(q,qgus)
else
  qacc = q
end
integrator.dtacc = integrator.dt
integrator.erracc = max(1e-2,integrator.EEst)
integrator.dt/qacc
```
When it rejects, it's the same as the [`IController`](@ref):
```julia
if integrator.success_iter == 0
  integrator.dt *= 0.1
else
  integrator.dt = integrator.dt/integrator.qold
end
```
"""
struct PredictiveController <: AbstractController
end

@inline function stepsize_controller!(integrator, controller::PredictiveController, alg)
  @unpack qmin, qmax, gamma = integrator.opts
  EEst = DiffEqBase.value(integrator.EEst)

  if iszero(EEst)
    q = inv(qmax)
  else
    if alg isa Union{RKC,IRKC,SERK2}
      fac = gamma
    else
      if alg isa Union{RadauIIA3, RadauIIA5}
        @unpack iter = integrator.cache
        @unpack maxiters = alg
      else
        @unpack iter, maxiters = integrator.cache.nlsolver
      end
      fac = min(gamma, ( 1 + 2 * maxiters) * gamma / (iter + 2 * maxiters))
    end
    expo = 1 / (get_current_adaptive_order(alg, integrator.cache) + 1)
    qtmp = DiffEqBase.fastpow(EEst, expo) / fac
    @fastmath q = DiffEqBase.value(max(inv(qmax), min(inv(qmin), qtmp)))
    integrator.qold = q
  end
  q
end

function step_accept_controller!(integrator, controller::PredictiveController, alg, q)
  @unpack qmin, qmax, gamma, qsteady_min, qsteady_max = integrator.opts
  EEst = DiffEqBase.value(integrator.EEst)

  if integrator.success_iter > 0
    expo = 1 / (get_current_adaptive_order(alg, integrator.cache) + 1)
    qgus = (integrator.dtacc / integrator.dt) * DiffEqBase.fastpow((EEst^2) / integrator.erracc, expo)
    qgus = max(inv(qmax), min(inv(qmin), qgus / gamma))
    qacc = max(q, qgus)
  else
    qacc = q
  end
  if qsteady_min <= qacc <= qsteady_max
    qacc = one(qacc)
  end
  integrator.dtacc = integrator.dt
  integrator.erracc = max(1e-2, EEst)
  return integrator.dt / qacc
end

function step_reject_controller!(integrator, controller::PredictiveController, alg)
  @unpack dt, success_iter, qold = integrator
  integrator.dt = success_iter == 0 ? 0.1 * dt : dt / qold
end


# Dummy controller without any method implementations.
# This is used to transfer the special controllers associated to certain
# algorithms to the new controller infrastructure with
struct DummyController <: AbstractController
end


# JVODE
function stepsize_controller!(integrator, alg::JVODE)
  if iszero(integrator.EEst)
    η = integrator.opts.qmax
  else
    η = integrator.cache.η
    integrator.qold = η
  end
  η
end

function step_accept_controller!(integrator,alg::JVODE,η)
  q = inv(η)
  if q <= integrator.opts.qsteady_max && q >= integrator.opts.qsteady_min
    q = one(q)
  end
  return integrator.dt/q  # dtnew
end

function step_reject_controller!(integrator,alg::JVODE)
  integrator.dt *= integrator.qold
end


# QNBDF
stepsize_controller!(integrator, alg::QNDF) = nothing

# this stepsize and order controller is taken from
# Implementation of an Adaptive BDF2 Formula and Comparison with the MATLAB Ode15s paper
# E. Alberdi Celaya, J. J. Anza Aguirrezabala, and P. Chatzipantelidis

function step_accept_controller!(integrator,alg::QNDF{max_order},q) where max_order
  #step is accepted, reset count of consecutive failed steps
  integrator.cache.consfailcnt = 0
  integrator.cache.nconsteps += 1
  if iszero(integrator.EEst)
    q = inv(integrator.opts.qmax)
  else
    est = integrator.EEst
    estₖ₋₁ = integrator.cache.EEst1
    estₖ₊₁ = integrator.cache.EEst2
    h = integrator.dt
    k = integrator.cache.order
    cache = integrator.cache
    if integrator.cache.nconsteps < integrator.cache.order + 2
      q = one(integrator.qold) #quasi-contsant steps
    else
      zₛ = 1.2 # equivalent to integrator.opts.gamma
      zᵤ = 0.1
      Fᵤ = 10
      expo = 1/(k+1)
      z = zₛ * ((est)^expo)
      F = inv(z)
      hₙ = h
      kₙ = k
      if z <= zᵤ
        hₖ = Fᵤ * h
      elseif zᵤ < z
        hₖ = F * h
      end
      hₖ₋₁ = 0.0
      hₖ₊₁ = 0.0

      if k > 1
        expo = 1/k
        zₖ₋₁ = 1.3 * ((estₖ₋₁)^expo)
        Fₖ₋₁ = inv(zₖ₋₁)
        if zₖ₋₁ <= 0.1
          hₖ₋₁ =  10* h
        elseif 1/10 < zₖ₋₁ <= 1.3
          hₖ₋₁ = Fₖ₋₁ * h
        end
        if hₖ₋₁ > hₖ
          hₙ = hₖ₋₁
          kₙ = k-1
        else
          hₙ = hₖ
          kₙ = k
        end
      end

      if k < max_order
        expo = 1/(k+2)
        zₖ₊₁ = 1.4 * ((estₖ₊₁)^expo)
        Fₖ₊₁ = inv(zₖ₊₁)

        if zₖ₊₁<= 0.1
          hₖ₊₁ = 10 * h
        elseif 0.1 < zₖ₊₁ <= 1.4
          hₖ₊₁ = Fₖ₊₁ * h
        end
        if hₖ₊₁ > hₙ
          hₙ = hₖ₊₁
          kₙ = k+1
        end
      end
      if hₙ <= h
        hₙ = h
        kₙ = k
      end
      cache.order = kₙ
      q = integrator.dt/hₙ
    end
  end
  if q <= integrator.opts.qsteady_max && q >= integrator.opts.qsteady_min
    q = one(q)
  end
  return integrator.dt/q
end

function step_reject_controller!(integrator,alg::QNDF)
  k = integrator.cache.order
  h = integrator.dt
  integrator.cache.consfailcnt += 1
  integrator.cache.nconsteps = 0
  if integrator.cache.consfailcnt > 1
    h = h/2
  end
  zₛ = 1.2  # equivalent to integrator.opts.gamma
  expo = 1/(k+1)
  z = zₛ * ((integrator.EEst)^expo)
  F = inv(z)
  if z <= 10
    hₖ = F * h
  elseif z > 10
    hₖ = 0.1 * h
  end
  hₙ = hₖ
  kₙ = k
  if k > 1
    expo = 1/k
    zₖ₋₁ = 1.3 * ((integrator.cache.EEst1)^expo)
    Fₖ₋₁ = inv(zₖ₋₁)
    if zₖ₋₁ <= 10
      hₖ₋₁ = Fₖ₋₁ * h
    elseif zₖ₋₁ > 10
      hₖ₋₁ = 0.1 * h
    end
    if hₖ₋₁ > hₖ
      hₙ = min(h,hₖ₋₁)
      kₙ = k-1
    end
  end
  integrator.dt = hₙ
  integrator.cache.order = kₙ
end

function stepsize_controller!(integrator, alg::Union{FBDF{max_order},DFBDF{max_order}}) where max_order
  @unpack t,dt,u,cache,uprev = integrator
  @unpack ts_tmp,terkm2, terkm1, terk, terkp1,u_history = cache
  cache.prev_order = cache.order
  k = cache.order

  if k < max_order && integrator.cache.nconsteps >= integrator.cache.order + 2 && ((k == 1 && terk > terkp1) ||
    (k == 2 && terkm1 > terk > terkp1) ||
    (k > 2 && terkm2 > terkm1 > terk > terkp1))
    k += 1
    terk = terkp1
  else
    while !(terkm2 > terkm1 > terk > terkp1) && k > 2
      terkp1 = terk
      terk = terkm1
      terkm1 = terkm2
      fd_weights = calc_finite_difference_weights(ts_tmp,t+dt,k-2,Val(max_order))
      if integrator.cache isa OrdinaryDiffEqMutableCache
        @unpack terk_tmp = integrator.cache
      end
      terk_tmp = @.. fd_weights[k-2,1] * u
      if typeof(u) <: Number
        for i in 2:k-2
          terk_tmp += fd_weights[i,k-2] * u_history[i-1]
        end
        #@show fd_weights,u_history,u,terk
        terk_tmp *= abs(dt^(k-2))
      else
        vc = _vec(terk_tmp)
        for i in 2:k-2
          @.. @views vc += fd_weights[i,k-2] * u_history[:,i-1]
        end
        @.. terk_tmp *= abs(dt^(k-2))
      end
      if integrator.cache isa OrdinaryDiffEqConstantCache
        atmp = calculate_residuals(_vec(terk_tmp), _vec(uprev), _vec(u), integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
      else
        @unpack atmp = integrator.cache
        calculate_residuals!(atmp,_vec(terk_tmp), _vec(uprev), _vec(u), integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
      end
      terkm2 = integrator.opts.internalnorm(atmp,t)
      k -= 1
    end
    if !(terkm1 > terk > terkp1) && k == 2
      k -= 1
      terk = terkm1
    end
  end
  if k != cache.order
    integrator.cache.nconsteps = 0
    cache.order = k
  end
  if iszero(terk)
    q = inv(integrator.opts.qmax)
  else
    q = ((2*terk/(k+1))^(1/(k+1)))
  end
  integrator.qold = q
  q
end

function step_accept_controller!(integrator,alg::Union{FBDF{max_order},DFBDF{max_order}},q) where max_order
  integrator.cache.consfailcnt = 0
  if q <= integrator.opts.qsteady_max && q >= integrator.opts.qsteady_min
    q = one(q)
  end
  integrator.cache.nconsteps += 1
  integrator.cache.nonevesuccsteps += 1
  return integrator.dt/q
end

function step_reject_controller!(integrator,alg::Union{FBDF,DFBDF})
  integrator.cache.consfailcnt += 1
  integrator.cache.nconsteps = 0
  dt = integrator.dt
  if integrator.cache.consfailcnt == 1
    dt *= 0.5
  else
    dt *= 0.25
  end
  integrator.dt = dt
end

# Extrapolation methods
mutable struct ExtrapolationController{QT} <: AbstractController
  beta1::QT
end

function reset_alg_dependent_opts!(controller::ExtrapolationController, alg1, alg2)
  if controller.beta1 == beta1_default(alg1, beta2_default(alg1))
    controller.beta1 = beta1_default(alg2, beta2_default(alg2))
  end
end

@inline function stepsize_controller!(integrator,alg::Union{ExtrapolationMidpointDeuflhard,ImplicitDeuflhardExtrapolation})
  # Dummy function
  # ExtrapolationMidpointDeuflhard's stepsize scaling is stored in the cache;
  # it is computed by  stepsize_controller_internal! (in perfom_step!) resp. stepsize_predictor!
  # (in step_accept_controller! and step_reject_controller!)
  zero(typeof(integrator.opts.qmax))
end

function stepsize_controller_internal!(integrator,alg::Union{ExtrapolationMidpointDeuflhard,ImplicitDeuflhardExtrapolation})
  # Standard step size controller
  # Compute and save the stepsize scaling based on the latest error estimate of the current order
  @unpack controller = integrator.opts

  if iszero(integrator.EEst)
    q = inv(integrator.opts.qmax)
  else
    # Update gamma and beta1
    controller.beta1 = typeof(controller.beta1)(1 // (2integrator.cache.n_curr + 1))
    integrator.opts.gamma = DiffEqBase.fastpow(typeof(integrator.opts.gamma)(1 // 4),controller.beta1)
    # Compute new stepsize scaling
    qtmp = DiffEqBase.fastpow(integrator.EEst,controller.beta1) / integrator.opts.gamma
    @fastmath q = max(inv(integrator.opts.qmax), min(inv(integrator.opts.qmin), qtmp))
  end
  integrator.cache.Q[integrator.cache.n_curr - alg.n_min + 1] = q
end

function stepsize_predictor!(integrator,alg::Union{ExtrapolationMidpointDeuflhard,ImplicitDeuflhardExtrapolation},n_new::Int)
  # Compute and save the stepsize scaling for order n_new based on the latest error estimate of the current order.
  @unpack controller = integrator.opts

  if iszero(integrator.EEst)
    q = inv(integrator.opts.qmax)
  else
    # Initialize
    @unpack t,EEst = integrator
    @unpack stage_number = integrator.cache
    tol = integrator.opts.internalnorm(integrator.opts.reltol,t) # Deuflhard's approach relies on EEstD ≈ ||relTol||
    s_curr = stage_number[integrator.cache.n_curr - alg.n_min + 1]
    s_new = stage_number[n_new - alg.n_min + 1]
    # Update gamma and beta1
    controller.beta1 = typeof(controller.beta1)(1 // (2integrator.cache.n_curr + 1))
    integrator.opts.gamma = DiffEqBase.fastpow(typeof(integrator.opts.gamma)(1 // 4),controller.beta1)
    # Compute new stepsize scaling
    qtmp = EEst * DiffEqBase.fastpow(DiffEqBase.fastpow(tol,(1.0 - s_curr / s_new)),controller.beta1) / integrator.opts.gamma
    @fastmath q = max(inv(integrator.opts.qmax),min(inv(integrator.opts.qmin),qtmp))
  end
  integrator.cache.Q[n_new - alg.n_min + 1] = q
end

function step_accept_controller!(integrator,alg::Union{ExtrapolationMidpointDeuflhard,ImplicitDeuflhardExtrapolation},q)
  # Compute new order and stepsize, return new stepsize
  @unpack n_min, n_max = alg
  @unpack n_curr, n_old, Q = integrator.cache
  s = integrator.cache.stage_number

  # Compute new order based on available quantities
  tmp = (n_min:n_curr) .- n_min .+ 1 # Index range of quantities computed so far
  dt_new = Vector{eltype(Q)}(undef,length(tmp)+1)
  dt_new[1:end-1] = integrator.dt ./ Q[tmp] # Store for the possible new stepsizes
  dtmin = timedepentdtmin(integrator)
  dt_new[1:end-1] = max.(dtmin, min.(abs(integrator.opts.dtmax), abs.(dt_new[1:end-1]))) # Safety scaling

  # n_new is the most efficient order of the last step
  work = s[tmp] ./ dt_new[1:end-1]
  n_new = argmin(work) + n_min - 1

  # Check if n_new may be increased
  if n_new == n_curr < min(n_max, n_old + 1) # cf. win_max in perfom_step! of the last step
    # Predict stepsize scaling for order (n_new + 1)
    stepsize_predictor!(integrator, alg, n_new+1) # Update cache.Q

    # Compute and scale the corresponding stepsize
    dt_new[end] = integrator.dt ./ Q[tmp[end]+1]
    dt_new[end] = max(dtmin, min(abs(integrator.opts.dtmax), abs.(dt_new[end])))

    # Check if (n_new  + 1) would have been more efficient than n_new
    if work[end] > s[tmp[end]+1] / dt_new[end]
      n_new = n_new + 1
    end
  end

  integrator.cache.n_curr = n_new
  dt_new[n_new - n_min + 1]
end

function step_reject_controller!(integrator, alg::Union{ExtrapolationMidpointDeuflhard,ImplicitDeuflhardExtrapolation})
  # Compute and save reduced stepsize dt_red of order n_old
  # Use the latest error estimate to predict dt_red if an estimate of order n_old is not available
  if integrator.cache.n_curr < integrator.cache.n_old
      stepsize_predictor!(integrator,alg,integrator.cache.n_old) # Update cache.Q
  end
  integrator.cache.n_curr = integrator.cache.n_old # Reset order for redoing the rejected step
  dt_red = integrator.dt / integrator.cache.Q[integrator.cache.n_old - integrator.alg.n_min + 1]
  dtmin = timedepentdtmin(integrator)
  dt_red = integrator.tdir*max(dtmin, min(abs(integrator.opts.dtmax), abs(dt_red))) # Safety scaling
  integrator.dt = dt_red
end

@inline function stepsize_controller!(integrator,alg::Union{ExtrapolationMidpointHairerWanner, ImplicitHairerWannerExtrapolation, ImplicitEulerExtrapolation, ImplicitEulerBarycentricExtrapolation})
  # Dummy function
  # ExtrapolationMidpointHairerWanner's stepsize scaling is stored in the cache;
  # it is computed by  stepsize_controller_internal! (in perfom_step!), step_accept_controller! or step_reject_controller!
  zero(typeof(integrator.opts.qmax))
end

function stepsize_controller_internal!(integrator,alg::Union{ExtrapolationMidpointHairerWanner, ImplicitHairerWannerExtrapolation, ImplicitEulerExtrapolation, ImplicitEulerBarycentricExtrapolation})
  # Standard step size controller
  # Compute and save the stepsize scaling based on the latest error estimate of the current order
  @unpack controller = integrator.opts

  if typeof(alg) <: Union{ImplicitEulerExtrapolation,ImplicitEulerBarycentricExtrapolation,ImplicitHairerWannerExtrapolation}
    if iszero(integrator.EEst)
      q = inv(integrator.opts.qmax)
    else
      # Update gamma and beta1
      if typeof(alg) <: ImplicitHairerWannerExtrapolation
        controller.beta1 = typeof(controller.beta1)(1 // (2integrator.cache.n_curr + 1))
      elseif typeof(alg) <: ImplicitEulerExtrapolation
        controller.beta1 = typeof(controller.beta1)(1 // (integrator.cache.n_curr))
      else
        controller.beta1 = typeof(controller.beta1)(1 // (integrator.cache.n_curr - 1))
      end
      integrator.opts.gamma = DiffEqBase.fastpow(typeof(integrator.opts.gamma)(65 // 100),controller.beta1)
      # Compute new stepsize scaling
      qtmp = DiffEqBase.fastpow(integrator.EEst,controller.beta1) / (integrator.opts.gamma)
      @fastmath q = max(inv(integrator.opts.qmax), min(inv(integrator.opts.qmin), qtmp))
    end
    integrator.cache.Q[integrator.cache.n_curr + 1] = q
  else
    if iszero(integrator.EEst)
      q = inv(integrator.opts.qmax)
    else
      # Update gamma and beta1
      controller.beta1 = typeof(controller.beta1)(1 // (2integrator.cache.n_curr + 1))
      integrator.opts.gamma = DiffEqBase.fastpow(typeof(integrator.opts.gamma)(65 // 100),controller.beta1)
      # Compute new stepsize scaling
      qtmp = DiffEqBase.fastpow(integrator.EEst,controller.beta1) / integrator.opts.gamma
      @fastmath q = max(inv(integrator.opts.qmax), min(inv(integrator.opts.qmin), qtmp))
    end
    integrator.cache.Q[integrator.cache.n_curr + 1] = q
  end
end

function step_accept_controller!(integrator,alg::Union{ExtrapolationMidpointHairerWanner, ImplicitHairerWannerExtrapolation, ImplicitEulerExtrapolation, ImplicitEulerBarycentricExtrapolation},q)
  # Compute new order and stepsize, return new stepsize
  @unpack n_min, n_max = alg
  @unpack n_curr, n_old, Q, sigma, work, dt_new = integrator.cache
  s = integrator.cache.stage_number

  # Compute new order based on available quantities
  win_min_old = min(n_old, n_curr) - 1 # cf. win_min in perfom_step! of the last step
  tmp = win_min_old:(max(n_curr, n_old) + 1) # Index range for the new order
  #@show size(dt_new)
  fill!(dt_new, zero(eltype(dt_new)))
  @.. Q = integrator.dt/Q
  copyto!(dt_new,win_min_old,Q,win_min_old,(max(n_curr, n_old) + 1) - win_min_old + 1)
  @.. Q = integrator.dt/Q
  dtmin = timedepentdtmin(integrator)
  fill!(work,zero(eltype(work))) # work[n] is the work for order (n-1)
  for i in tmp
    work[i] = s[i]/dt_new[i]
  end
  # Order selection
  n_new = n_old
  if n_curr == n_min # Enforce n_min + 1 ≦ n_new
    n_new = n_min + 1
  else
    if n_curr <= n_old
      if work[n_curr-1] < sigma * work[n_curr]
        n_new = max(n_curr-1,n_old-1,n_min+1) # Enforce n_min + 1≦ n_new
      elseif work[n_curr] < sigma * work[n_curr-1]
        n_new = min(n_curr+1,n_max-1) # Enforce n_new ≦ n_max - 1
      else
        n_new = n_curr # n_min + 1 ≦ n_curr
      end
    else
      if work[n_old] < sigma *  work[n_old+1]
        n_new = max(n_old-1,n_min+1)  # Enforce n_min + 1 ≦ n_new
      end
      if work[n_curr+1] <  sigma * work[n_new+1]
        n_new = min(n_new+1,n_max-1) # Enforce n_new ≦ n_max - 1
      end
    end
  end
  integrator.cache.n_curr = n_new

  # Stepsize selection
  if n_new == n_curr + 1
    # Compute the new stepsize of order n_new based on the optimal stepsize of order n_curr
    dt_new[n_new+1] = s[n_curr + 2]/s[n_curr + 1 ] * dt_new[n_curr+1]
    dt_new[n_new+1] = max(dtmin, min(abs(integrator.opts.dtmax), abs(dt_new[n_new+1])))
  end
  dt_new[n_new + 1]
end

function step_reject_controller!(integrator, alg::Union{ExtrapolationMidpointHairerWanner, ImplicitHairerWannerExtrapolation, ImplicitEulerExtrapolation, ImplicitEulerBarycentricExtrapolation})
  # Compute and save order and stepsize for redoing the current step
  @unpack n_old, n_curr, Q = integrator.cache

  # Order selection
  n_red = n_old
  if n_curr == n_old - 1
    n_red = max(alg.n_min+1,n_old-1) # Enforce n_min + 1 ≦ n_red
  end
  integrator.cache.n_curr = n_red

  # Stepsize selection
  dt_red = integrator.dt / Q[n_red + 1]
  dtmin = timedepentdtmin(integrator)
  dt_red = integrator.tdir*max(dtmin, min(abs(integrator.opts.dtmax), abs(dt_red))) # Safety scaling
  integrator.dt = dt_red
end
