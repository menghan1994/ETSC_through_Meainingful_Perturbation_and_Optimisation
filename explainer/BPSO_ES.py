import pyswarms as ps
import numpy as np
import logging
import multiprocessing as mp
from collections import deque

from pyswarms.backend.operators import compute_pbest, compute_objective_function


class BPSO_ES(ps.discrete.binary.BinaryPSO):
    def __init___(
        self, 
        n_particles, 
        dimensions, 
        options, 
        init_pos=None, 
        velocity_clamp=None,
        vh_strategy="unmodified",
        ftol=-np.inf,
        ftol_iter=1,
        ):
        super(BPSO_ES, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            init_pos=init_pos,
            velocity_clamp=velocity_clamp,
            vh_strategy = vh_strategy,
            ftol=ftol,
            ftol_iter=ftol_iter,
        )

    def optimize(self, objective_func, iters, n_processes=None, verbose=True, **kwargs):
        
        self.early_stop = kwargs['early_stop']
        temp_best_cost = np.inf

        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET
        
        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log( "Optimize for {} iters with {}".format(iters, self.options), lvl=log_level,)
        self.vh.memory = self.swarm.position
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)

        ftol_history = deque(maxlen=self.ftol_iter)
        for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
            # Compute cost for current position and personal best
            self.swarm.current_cost = compute_objective_function(
                self.swarm, objective_func, pool, **kwargs
            )
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(
                self.swarm
            )
            best_cost_yet_found = np.min(self.swarm.best_cost)
            # Update gbest from neighborhood
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, p=self.p, k=self.k
            )

            if self.swarm.best_cost < temp_best_cost:
                temp_best_cost = self.swarm.best_cost
                early_stop = 0
            else:
                early_stop += 1
            
            if early_stop >= self.early_stop:
                break

            if verbose:
                # Print to console
                self.rep.hook(best_cost=self.swarm.best_cost)
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=np.mean(self.swarm.best_cost),
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            )
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform position velocity update
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh
            )
            self.swarm.position = self._compute_position(self.swarm)
        
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()

        return (final_best_cost, final_best_pos)

