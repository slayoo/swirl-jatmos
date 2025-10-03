# Copyright 2024 The swirl_jatmos Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Driver for Jatmo simulations."""

import os
import sys
import time
from typing import Callable, TypeAlias

from absl import flags
from absl import logging
import jax
import numpy as np
import orbax.checkpoint as ocp
from swirl_jatmos import config
from swirl_jatmos import navier_stokes_step
from swirl_jatmos import sim_initializer
from swirl_jatmos import stretched_grid_util
from swirl_jatmos import timestep_control
from swirl_jatmos.linalg import poisson_solver_interface
from swirl_jatmos.sim_setups import walker_circulation_diagnostics
from swirl_jatmos.utils import check_states_valid
from swirl_jatmos.utils import text_util
from swirl_jatmos.utils import utils


OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp/data',
    'The output directory for checkpoints to be saved in.',
    allow_override=True,
)
T_FINAL = flags.DEFINE_float(
    't_final',
    0.0,
    'Total time for simulation to run.',
    allow_override=True,
)
SEC_PER_CYCLE = flags.DEFINE_float(
    'sec_per_cycle',
    1,
    'Duration of a single cycle, in seconds.  Each cycle generates a set of'
    ' output.',
    allow_override=True,
)
_LABEL_CHECKPOINTS_BY_CYCLE = flags.DEFINE_bool(
    'label_checkpoints_by_cycle',
    True,
    'If True, orbax checkpoint directories are labeled by cycle number;'
    ' otherwise they are labeled by step_id.',
    allow_override=True,
)


FLAGS = flags.FLAGS
FLAGS(sys.argv)
DEBUG_CHECK_FOR_NANS = check_states_valid.DEBUG_CHECK_FOR_NANS

Array: TypeAlias = jax.Array
PoissonSolverType: TypeAlias = config.PoissonSolverType
StatesMap: TypeAlias = dict[str, Array]
InitFn: TypeAlias = Callable[[config.Config], StatesMap]
UpdateFn: TypeAlias = Callable[
    [StatesMap, config.Config], tuple[StatesMap, StatesMap]
]
DiagnosticsUpdateFn: TypeAlias = Callable[
    [StatesMap, StatesMap, StatesMap, Array, config.Config], StatesMap
]


class _NonRecoverableError(Exception):
  """Errors that cannot be recovered from via restarts."""

  pass


def get_init_state(customized_init_fn: InitFn, cfg: config.Config) -> StatesMap:
  """Creates the initial state using `customized_init_fn`."""
  t_start = time.time()
  states = {}

  # Add variables for grids / stretched grids
  grid_map_sharded = sim_initializer.initialize_grids(cfg)
  states |= grid_map_sharded

  # Add t, dt, and step_id to states.
  states |= sim_initializer.initialize_time_and_step_id(cfg)

  # Add the variables from the customized_init_fn.
  # Initialize:
  #  p_ref_xxc, rho_xxc, rho_xxf, theta_li_0, dtheta_li, q_t, q_r, q_s
  #  u, v, w, p
  states |= customized_init_fn(cfg)

  # Initialize sharded aux_output variables.
  aux_output = {}
  for fieldname in cfg.aux_output_fields:
    aux_output[fieldname] = sim_initializer.initialize_zeros_from_varname(
        fieldname, cfg
    )

  states_aux_output_map = states | aux_output

  logging.info('Initialization complete.')
  t_post_init = time.time()
  logging.info(
      'Initialization stage took %s.',
      text_util.seconds_to_string(t_post_init - t_start),
  )
  return states_aux_output_map


def initialize_diagnostics(cfg: config.Config) -> StatesMap:
  """Initialize sharded diagnostic variables as zeros."""
  diagnostics = {}
  for fieldname in cfg.diagnostic_fields:
    diagnostics[fieldname] = sim_initializer.initialize_zeros_from_varname(
        fieldname, cfg
    )
  return diagnostics


def run_driver(
    customized_init_fn: InitFn,
    rho_ref_xxc: np.ndarray,
    output_dir: str,
    t_final: float,
    sec_per_cycle: float,
    cfg: config.Config,
    preprocess_update_fn: UpdateFn | None = None,
    diagnostics_update_fn: DiagnosticsUpdateFn | None = None,
) -> tuple[StatesMap, StatesMap, StatesMap]:
  """Run the Jatmo solver."""
  logging.info('Devices: %s', jax.devices())
  logging.info('Config: %s', cfg)

  try:
    # Generate the initial state (which will be overwritten if a checkpoint is
    # restored).
    states_aux_output_map = get_init_state(customized_init_fn, cfg)

    # Initialize objects that need to be created statically, e.g., the Poisson
    # solver.
    poisson_solver = poisson_solver_interface.get_poisson_solver(
        rho_ref_xxc, cfg
    )
    return solver_loop(
        states_aux_output_map,
        output_dir,
        t_final,
        sec_per_cycle,
        poisson_solver,
        cfg,
        preprocess_update_fn,
        diagnostics_update_fn,
    )
  except _NonRecoverableError:
    logging.exception(
        'Non-recoverable error in simulation.  Returning None istead of raising'
        ' an exception to avoid automatic restarts.'
    )
    return {}, {}, {}


def solver_loop(
    init_states_aux_output_map: StatesMap,
    output_dir: str,
    t_final: float,
    sec_per_cycle: float,
    poisson_solver: poisson_solver_interface.PoissonSolver,
    cfg: config.Config,
    preprocess_update_fn: UpdateFn | None = None,
    diagnostics_update_fn: DiagnosticsUpdateFn | None = None,
) -> tuple[StatesMap, StatesMap, StatesMap]:
  """Runs the solver."""
  logging.info('Entering solver_loop.')

  logging.info('Getting checkpoint manager.')
  async_save = False
  options = ocp.CheckpointManagerOptions(enable_async_checkpointing=async_save)
  mngr = ocp.CheckpointManager(output_dir, options=options)

  diagnostics_mngr = None
  if diagnostics_update_fn is not None:
    diagnostics_dir = os.path.join(output_dir, 'diagnostics')
    diagnostics_options = ocp.CheckpointManagerOptions(
        enable_async_checkpointing=async_save
    )
    diagnostics_mngr = ocp.CheckpointManager(
        diagnostics_dir, options=diagnostics_options
    )

  logging.info('Got checkpoint_manager.')

  states_aux_output_map = init_states_aux_output_map
  diagnostics = {}
  write_initial_state = False

  # Restore from an existing checkpoint if present.
  if mngr.latest_step():
    msg = 'Detected checkpoint. Restoring from latest checkpoint...'
    logging.info(msg)
    utils.print_if_flag(msg)
    t_pre_restore = time.time()
    states_aux_output_map = mngr.restore(
        mngr.latest_step(), args=ocp.args.StandardRestore(states_aux_output_map)
    )
    t_post_restore = time.time()
    logging.info(
        '  ... Checkpoint restored.  Took %s.',
        text_util.seconds_to_string(t_post_restore - t_pre_restore),
    )
  else:
    write_initial_state = True
    msg = (
        'No checkpoint was found. Proceeding with default initializations for'
        ' all variables.'
    )
    logging.info(msg)
    utils.print_if_flag(msg)

  if write_initial_state and not cfg.disable_checkpointing:
    logging.info('Saving checkpoint for the initial state.')
    mngr.save(0, args=ocp.args.StandardSave(states_aux_output_map))

  logging.info(
      'Simulation loop starting. Running until %s completed, with %s'
      ' per cycle. Starting from step %d, time %s.',
      text_util.seconds_to_string(t_final),
      text_util.seconds_to_string(sec_per_cycle),
      int(states_aux_output_map['step_id']),
      text_util.seconds_to_string(1e-9 * states_aux_output_map['t_ns']),
  )

  # Extract states and aux_output from states_aux_output_map.
  states = {
      k: v
      for k, v in states_aux_output_map.items()
      if k not in cfg.aux_output_fields
  }
  aux_output = {
      k: v
      for k, v in states_aux_output_map.items()
      if k in cfg.aux_output_fields
  }

  one_cycle = jax.jit(
      generate_one_cycle_fn(
          sec_per_cycle,
          poisson_solver,
          cfg,
          preprocess_update_fn,
          diagnostics_update_fn,
      )
  )

  t_final_ns = int(1e9 * t_final)
  while states['t_ns'] < t_final_ns:
    # Add one to cycle number so that the initial condition corresponds to 0.
    cycle = 1 + int(round((states['t_ns'] // 1_000_000_000) / sec_per_cycle))
    logging.info('Starting cycle %d (step %d)', cycle, states['step_id'])
    t0 = time.time()  # Keep track of wall time.
    step_id_start = states['step_id']

    states, aux_output, diagnostics = one_cycle(states, aux_output)
    jax.block_until_ready(states)

    t1 = time.time()
    num_steps = states['step_id'] - step_id_start
    # For colab
    msg = (
        'Finished cycle {}. Simulation time {}. Current step={}.'
        ' Took {} s for the last cycle ({} steps). Avg dt of last cycle: {}.'
    ).format(
        cycle,
        text_util.seconds_to_string(1e-9 * states['t_ns']),
        states['step_id'],
        text_util.seconds_to_string(t1 - t0),
        num_steps,
        f'{sec_per_cycle / num_steps:.2f} s',
    )
    logging.info(msg)
    utils.print_if_flag(msg)

    # Check for NaNs.
    if not states['all_valid']:
      msg = f'NaN or Inf detected. Exiting early at cycle {cycle}.'
      logging.info(msg)
      utils.print_if_flag(msg)
      raise _NonRecoverableError(
          f'NaNs found in u. Early exit from cycle {cycle}.'
      )

    # Save the checkpoints and diagnostics
    if not cfg.disable_checkpointing:
      if _LABEL_CHECKPOINTS_BY_CYCLE.value:
        ckpt_label = cycle
      else:
        ckpt_label = states['step_id']

      # Save checkpoints every `cfg.checkpoint_cycle_interval` cycles.
      if cycle % cfg.checkpoint_cycle_interval == 0:
        states_to_save = states | aux_output
        mngr.save(ckpt_label, args=ocp.args.StandardSave(states_to_save))
        t2 = time.time()
        logging.info(
            'Saved checkpoint for step %d, took %s (is async).',
            states['step_id'],
            text_util.seconds_to_string(t2 - t1),
        )

      if diagnostics_mngr is not None:
        # Save the diagnostics every cycle.
        diagnostics_mngr.save(
            ckpt_label, args=ocp.args.StandardSave(diagnostics)
        )
      logging.info('Saved diagnostics for cycle %d.', cycle)

  mngr.wait_until_finished()
  logging.info('Simulation complete.')
  return states, aux_output, diagnostics


def generate_one_step_fn(
    poisson_solver: poisson_solver_interface.PoissonSolver,
    sec_per_cycle: float,
    cfg: config.Config,
    preprocess_update_fn: UpdateFn | None = None,
    diagnostics_update_fn: DiagnosticsUpdateFn | None = None,
) -> Callable[
    [tuple[StatesMap, StatesMap, StatesMap]],
    tuple[StatesMap, StatesMap, StatesMap],
]:
  """Return a function that runs one step of the simulation."""

  def one_step_fn(
      states_and_aux_output_and_diags: tuple[StatesMap, StatesMap, StatesMap],
  ) -> tuple[StatesMap, StatesMap, StatesMap]:
    """Perform one step of the simulation.

    One step is defined as:
    1. Step Preprocessing / states update (optional)
    2. Run the Navier-Stokes solver for velocities, pressure, and scalars.

    Args:
      states_and_aux_output_and_diags: Tuple of the current states, aux_output,
        and diagnostics.

    Returns:
      The updated states, aux_output, and diagnostics.
    """
    states, _, diagnostics = states_and_aux_output_and_diags

    # Add an additional_states update if desired.
    if preprocess_update_fn is not None:
      updated_states, aux_output1 = preprocess_update_fn(states, cfg)
      states |= updated_states
    else:
      aux_output1 = {}

    # Run the Navier-Stokes step.
    states, aux_output2 = navier_stokes_step.step(
        states, poisson_solver, cfg
    )
    aux_output = aux_output1 | aux_output2

    # Perform any diagnostic updates.
    if diagnostics_update_fn is not None:
      # Get dt in seconds.
      dt = (1e-9 * states['dt_ns']).astype(states['u'].dtype)
      # Compute the fractional amount of the cycle this dt represents; used for
      # accumulating mean values in the diagnostics.
      dt_over_cycle_time = dt / sec_per_cycle
      diagnostics |= diagnostics_update_fn(
          states, aux_output, diagnostics, dt_over_cycle_time, cfg
      )

    # Update the timestep based on CFL condition.
    dx_f, dy_f, dz_f = stretched_grid_util.get_dxdydz(
        states, cfg.grid_spacings, faces=True
    )
    states['dt_ns'] = timestep_control.compute_next_dt(
        states['u'],
        states['v'],
        states['w'],
        dx_f,
        dy_f,
        dz_f,
        states['dt_ns'],
        states['step_id'],
        cfg.timestep_control_cfg,
    )

    # Filter out fields from `aux_output` that are not specified to be saved in
    # checkpoints by being listed `cfg.aux_output_fields` (the fields filtered
    # out here may have been used to calculate diagnostics).
    aux_output = {
        k: v for k, v in aux_output.items() if k in cfg.aux_output_fields
    }
    return states, aux_output, diagnostics

  return one_step_fn


def generate_one_cycle_fn(
    sec_per_cycle: float,
    poisson_solver: poisson_solver_interface.PoissonSolver,
    cfg: config.Config,
    preprocess_update_fn: UpdateFn | None = None,
    diagnostics_update_fn: DiagnosticsUpdateFn | None = None,
) -> Callable[[StatesMap, StatesMap], tuple[StatesMap, StatesMap, StatesMap]]:
  """Return a function that runs one cycle of the simulation."""
  one_step_fn = generate_one_step_fn(
      poisson_solver,
      sec_per_cycle,
      cfg,
      preprocess_update_fn,
      diagnostics_update_fn,
  )
  ns_per_cycle = int(sec_per_cycle * 1e9)

  def one_cycle(
      states: StatesMap, aux_output: StatesMap
  ) -> tuple[StatesMap, StatesMap, StatesMap]:
    logging.info('Starting tracing and compiling of one_cycle.')
    t_ns_beginning_of_cycle = states['t_ns']
    t_ns_end_of_cycle = t_ns_beginning_of_cycle + ns_per_cycle
    # Initialize sharded diagnostic variables to zeros.
    diagnostics = initialize_diagnostics(cfg)
    init_val = (states, aux_output, diagnostics)


    def cond_fun(
        states_and_aux_output_and_diags: tuple[StatesMap, StatesMap, StatesMap],
    ):
      states, _, _ = states_and_aux_output_and_diags
      cycle_not_finished = states['t_ns'] < t_ns_end_of_cycle
      if not DEBUG_CHECK_FOR_NANS.value:
        return cycle_not_finished
      else:
        states_are_finite = states['all_valid']
        return cycle_not_finished & states_are_finite

    def wrapped_one_step_fn(
        states_and_aux_output_and_diags: tuple[StatesMap, StatesMap, StatesMap],
    ):
      states, _, _ = states_and_aux_output_and_diags
      # states_beginning_of_step = dict(states)

      # Reduce step size if step would overshoot the end point
      states['dt_ns'] = jax.lax.cond(
          states['t_ns'] + states['dt_ns'] > t_ns_end_of_cycle,
          lambda: t_ns_end_of_cycle - states['t_ns'],
          lambda: states['dt_ns'],
      )

      states, aux_output, diagnostics = one_step_fn(
          states_and_aux_output_and_diags
      )

      if DEBUG_CHECK_FOR_NANS.value:
        # check for NaNs and Infs.
        states_are_finite = check_states_valid.check_states_are_finite(states)

        # If NaN, return the states from the prior, good state.
        # states = jax.lax.cond(
        #     states_are_finite,
        #     lambda: states,
        #     lambda: states_beginning_of_step,
        # )

        # Record if there is an invalid value.
        states['all_valid'] = states_are_finite

      return states, aux_output, diagnostics

    states, aux_output, diagnostics = jax.lax.while_loop(
        cond_fun, wrapped_one_step_fn, init_val
    )


    # At the end of a cycle, always check for NaNs.
    states['all_valid'] = check_states_valid.check_states_are_finite(states)

    # At the end of a cycle, add the time and step_id to the diagnostics.
    diagnostics['t_ns'] = states['t_ns']
    diagnostics['step_id'] = states['step_id']
    return states, aux_output, diagnostics

  return one_cycle
