## Reservoir Plan
- Initial Phase: ***Currently Working ***
    - Improve survey contents:
        - How Optimization works in this problem, enviornment swiches, successful modeling.
        *** Important ***: we need to figure out how physics constraints translate to optimization problem, then to reinforcement learning scenario.
        - How other solve this kind of problems?
    - Target:
        - [x] Project Structure & base actor created
        - [ ] Hydra based config
        - [ ] main fit loop for papers
        - [ ] enviornment modeling


- Second Phase:
    - Algorithm Implementation
    - Dataset Generation
    - Framework Engineering
- Third Phase:
    - Framework applied
    - Model Training
    - Model Evaluation
---------
## Modeling Specific Problem
Objective:

\begin{equation}
\max E = \omega_1 \times \sum_{i=1}^{N}\sum_{j=1}^{T}\frac{P_{i,j}\times \Delta t_j}{10^4}+\omega_3 \times \sum_{i=1}^{N} f_i(V_{i,T}, Z_{i,T})
\end{equation}

We now turn constraints to...
 - For decision variable Q_gen, Q_s, we predict them with neuro network combining reparameter techniques(i.e. using action bound in SAC algorithms)
 - For state transition, we tend to design a class method of ```class Reservoir.Envs.Multi_Reservoir.Simple_Env``` to simulate state transition from t to t+1. Note there two kinds of variables: one depends on specific reservoir, the other depends only on the enviornment.
### For reservoir related attributes:
- We model water balance equation, modeling the Volumn of the reservoir S_t+1 from Inflow and outflow(Q_gen, Q_s) of that time ***Problem*** what is q?
- WE model...
- Key: We get state transfer result from model, based on enviornment(like waterflow) we can calculate state at t+1 based on ***Constraints Equation***; For ***Constraints Inequation***, we model part of them as soft penalty, and other parts as action bounds.(Outflow Constraints)
### For enviornment related attributes:
- Main Bottleneck: How we get their state transfer function?
- different from traditional optimization problem, we need to model how agent/actions influence the enviornment;
    - Method I: ML prediction based on historical result.
    - Method II: Build a interpretable model, determined function for transformation calculation.
    - Method III: ....
