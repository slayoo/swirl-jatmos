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

import functools
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
from swirl_jatmos import constants
from swirl_jatmos import jatmos_types
from swirl_jatmos import kernel_ops
from swirl_jatmos import test_util
from swirl_jatmos.rrtmgp.config import radiative_transfer
from swirl_jatmos.rrtmgp.optics import atmospheric_state
from swirl_jatmos.rrtmgp.optics import optics
from swirl_jatmos.rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array

_VMR_GLOBAL_MEAN_FILENAME = 'rrtmgp/optics/test_data/vmr_global_means.json'
_ATMOSPHERIC_STATE_FILENAME = 'rrtmgp/optics/test_data/clearsky_as.nc'
_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g256.nc'
_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g224.nc'
_LW_LOOKUP_TABLE_COMPACT_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g128.nc'
_SW_LOOKUP_TABLE_COMPACT_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g112.nc'
_CLD_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'
_CLD_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_sw.nc'
_LW_FLUX_DOWN_REFERENCE_FILENAME = 'rrtmgp/optics/test_data/clearsky_lw_flux_dn_TwoStream.nc'
_LW_FLUX_UP_REFERENCE_FILENAME = 'rrtmgp/optics/test_data/clearsky_lw_flux_up_TwoStream.nc'
_SW_FLUX_DOWN_REFERENCE_FILENAME = 'rrtmgp/optics/test_data/clearsky_sw_flux_dn_TwoStream.nc'
_SW_FLUX_UP_REFERENCE_FILENAME = 'rrtmgp/optics/test_data/clearsky_sw_flux_up_TwoStream.nc'

root = epath.resource_path('swirl_jatmos')
_VMR_GLOBAL_MEAN_FILEPATH = root / _VMR_GLOBAL_MEAN_FILENAME
_ATMOSPHERIC_STATE_FILEPATH = root / _ATMOSPHERIC_STATE_FILENAME
_LW_LOOKUP_TABLE_FILEPATH = root / _LW_LOOKUP_TABLE_FILENAME
_SW_LOOKUP_TABLE_FILEPATH = root / _SW_LOOKUP_TABLE_FILENAME
_LW_LOOKUP_TABLE_COMPACT_FILEPATH = root / _LW_LOOKUP_TABLE_COMPACT_FILENAME
_SW_LOOKUP_TABLE_COMPACT_FILEPATH = root / _SW_LOOKUP_TABLE_COMPACT_FILENAME
_CLD_LW_LOOKUP_TABLE_FILEPATH = root / _CLD_LW_LOOKUP_TABLE_FILENAME
_CLD_SW_LOOKUP_TABLE_FILEPATH = root / _CLD_SW_LOOKUP_TABLE_FILENAME
_LW_FLUX_DOWN_REFERENCE_FILEPATH = root / _LW_FLUX_DOWN_REFERENCE_FILENAME
_LW_FLUX_UP_REFERENCE_FILEPATH = root / _LW_FLUX_UP_REFERENCE_FILENAME
_SW_FLUX_DOWN_REFERENCE_FILEPATH = root / _SW_FLUX_DOWN_REFERENCE_FILENAME
_SW_FLUX_UP_REFERENCE_FILEPATH = root / _SW_FLUX_UP_REFERENCE_FILENAME


def _remove_halos(f: Array) -> Array:
  """Remove the halos from the output."""
  return f[:, :, 1:-1]


def _setup_radiation_params(
    site: int,
    use_compact_lookup: bool,
    atmos_state_ds: nc.Dataset,
) -> radiative_transfer.RadiativeTransfer:
  """Create an instance of `RadiativeTransfer`."""
  if use_compact_lookup:
    lw_lookup_table_nc_filepath = _LW_LOOKUP_TABLE_COMPACT_FILEPATH
    sw_lookup_table_nc_filepath = _SW_LOOKUP_TABLE_COMPACT_FILEPATH
  else:
    lw_lookup_table_nc_filepath = _LW_LOOKUP_TABLE_FILEPATH
    sw_lookup_table_nc_filepath = _SW_LOOKUP_TABLE_FILEPATH

  return radiative_transfer.RadiativeTransfer(
      optics=radiative_transfer.OpticsParameters(
          optics=radiative_transfer.RRTMOptics(
              longwave_nc_filepath=lw_lookup_table_nc_filepath,
              shortwave_nc_filepath=sw_lookup_table_nc_filepath,
              cloud_longwave_nc_filepath=_CLD_LW_LOOKUP_TABLE_FILEPATH,
              cloud_shortwave_nc_filepath=_CLD_SW_LOOKUP_TABLE_FILEPATH,
          )
      ),
      atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
          sfc_emis=atmos_state_ds['surface_emissivity'][:].data[site],
          sfc_alb=atmos_state_ds['surface_albedo'][:].data[site],
          zenith=np.radians(atmos_state_ds['solar_zenith_angle'][:].data[site]),
          irrad=atmos_state_ds['total_solar_irradiance'][:].data[site],
          toa_flux_lw=0.0,
          vmr_global_mean_filepath=_VMR_GLOBAL_MEAN_FILEPATH,
      ),
  )


def _setup_atmospheric_profiles() -> tuple[
    nc.Dataset,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
  """Load vertical profiles of pressure and temperature from RFMIP."""
  halo_width = 1
  atmos_state_ds = nc.Dataset(_ATMOSPHERIC_STATE_FILEPATH, 'r')

  # Reverse order of the profiles so they correspond to increasing altitude.
  p_internal = np.flip(atmos_state_ds['pres_layer'][:].data, axis=-1)
  pres_level = np.flip(atmos_state_ds['pres_level'][:].data, axis=-1)
  # Shape of p_internal is (100, 60) and shape of pres_level is (100, 61).

  paddings_2d = ((0, 0), (halo_width, halo_width))
  paddings_3d = ((0, 0), (0, 0), (halo_width, halo_width))
  pressure = np.pad(p_internal, paddings_2d, mode='edge')
  pressure_level = np.pad(pres_level, ((0, 0), (halo_width, halo_width - 1)))
  # Shape of pressure is (100, 62) and shape of pressure_level is (100, 62).

  # shape of temperature_internal is (18, 100, 60)
  temp_internal = np.flip(atmos_state_ds['temp_layer'][:].data, axis=-1)
  # shape of temperature_level is (18, 100, 61)
  temp_level = np.flip(atmos_state_ds['temp_level'][:].data, axis=-1)

  # Fill in halos by extrapolating from face (temp_level) and node values
  # (temp_internal).
  nx, ny, nz = temp_internal.shape  # 18, 100, 60
  nz_with_halos = nz + 2 * halo_width
  temperature = np.zeros((nx, ny, nz_with_halos), dtype=jatmos_types.f_dtype)
  temperature[:, :, halo_width:-halo_width] = temp_internal
  temperature[:, :, 0] = 2 * temp_level[:, :, 0] - temp_internal[:, :, 0]
  temperature[:, :, -1] = 2 * temp_level[:, :, -1] - temp_internal[:, :, -1]

  temperature_level = np.zeros_like(temperature)
  temperature_level[:, :, 1:] = temp_level
  # Fill in halo with same halo value as in `temperature`.
  temperature_level[:, :, 0] = temperature[:, :, 0]

  h2o_vmr = np.pad(
      np.flip(atmos_state_ds['water_vapor'][:].data, axis=-1),
      paddings_3d,
      mode='edge',
  )
  o3_vmr = np.pad(
      np.flip(atmos_state_ds['ozone'][:].data, axis=-1),
      paddings_3d,
      mode='edge',
  )

  sfc_temperature = atmos_state_ds['surface_temperature'][:].data
  # shape of sfc_temperature is (18, 100).

  # Here `_level` means these are the values on faces.
  return (
      atmos_state_ds,
      pressure,
      pressure_level,
      temperature,
      temperature_level,
      h2o_vmr,
      o3_vmr,
      sfc_temperature,
  )


def _load_expected_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
  """Loads the expected fluxes from the reference netCDF files."""
  reference_lw_up_ds = nc.Dataset(_LW_FLUX_UP_REFERENCE_FILEPATH, 'r')
  reference_lw_down_ds = nc.Dataset(_LW_FLUX_DOWN_REFERENCE_FILEPATH, 'r')
  reference_sw_up_ds = nc.Dataset(_SW_FLUX_UP_REFERENCE_FILEPATH, 'r')
  reference_sw_down_ds = nc.Dataset(_SW_FLUX_DOWN_REFERENCE_FILEPATH, 'r')

  halo_width = 1
  paddings = ((0, 0), (0, 0), (halo_width, halo_width - 1))

  lw_flux_up = np.pad(
      np.flip(reference_lw_up_ds['rlu'][:].data, axis=-1), paddings
  )
  lw_flux_down = np.pad(
      np.flip(reference_lw_down_ds['rld'][:].data, axis=-1), paddings
  )
  sw_flux_up = np.pad(
      np.flip(reference_sw_up_ds['rsu'][:].data, axis=-1), paddings
  )
  sw_flux_down = np.pad(
      np.flip(reference_sw_down_ds['rsd'][:].data, axis=-1), paddings
  )
  return lw_flux_up, lw_flux_down, sw_flux_up, sw_flux_down


def _air_molecules_per_area(p_bottom: Array, vmr_h2o: Array) -> Array:
  """Compute the number of molecules in an atmospheric grid cell per area."""
  dp = kernel_ops.forward_difference(p_bottom, dim=2)
  mol_m_air = constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * vmr_h2o
  return -(dp / constants.G) * constants.AVOGADRO / mol_m_air


class ClearSkyTest(parameterized.TestCase):

  @parameterized.product(
      rfmip_site=list(range(10)),
      use_compact_lookup=[True, False],
      use_scan=[True, False],
  )
  def test_two_stream_solver_with_clear_sky(
      self, rfmip_site: int, use_compact_lookup: bool, use_scan: bool
  ):
    rfmip_expt_id = 0

    (
        atmos_state_ds,
        pressure_allsites,
        pressure_level_allsites,
        temperature_allsites,
        _,
        h2o_vmr_allsites,
        o3_vmr_allsites,
        sfc_temperature_allsites,
    ) = _setup_atmospheric_profiles()
    # Pressure & pressure_level are 2D arrays where the last dimension is
    # the vertical dimension.  Temperature & the vmr data are 3D arrays where
    # the last dimension is the vertical dimension.

    radiation_params = _setup_radiation_params(
        rfmip_site, use_compact_lookup, atmos_state_ds
    )
    atmos_state = atmospheric_state.from_config(
        radiation_params.atmospheric_state_cfg
    )
    optics_lib = optics.optics_factory(radiation_params.optics, atmos_state.vmr)

    # Model inputs.
    n_horiz = 2
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n_horiz
    )

    sfc_temperature_value = sfc_temperature_allsites[rfmip_expt_id, rfmip_site]
    sfc_temperature = sfc_temperature_value * jnp.ones(
        (n_horiz, n_horiz), dtype=jatmos_types.f_dtype
    )
    p = convert_to_3d(pressure_allsites[rfmip_site, :])
    pressure_level = convert_to_3d(pressure_level_allsites[rfmip_site, :])
    temperature = convert_to_3d(
        temperature_allsites[rfmip_expt_id, rfmip_site, :]
    )
    h2o_vmr = convert_to_3d(h2o_vmr_allsites[rfmip_expt_id, rfmip_site, :])
    o3_vmr = convert_to_3d(o3_vmr_allsites[rfmip_expt_id, rfmip_site, :])
    molecules = _air_molecules_per_area(pressure_level, h2o_vmr)
    vmr_fields = {'h2o': h2o_vmr, 'o3': o3_vmr}

    # ACTION
    lw_fluxes = two_stream.solve_lw(
        p,
        temperature,
        molecules,
        optics_lib,
        atmos_state,
        vmr_fields,
        sfc_temperature,
        use_scan=use_scan,
    )
    sw_fluxes = two_stream.solve_sw(
        p,
        temperature,
        molecules,
        optics_lib,
        atmos_state,
        vmr_fields,
        use_scan=use_scan,
    )

    # VERIFICATION
    (
        expected_lw_flux_up,
        expected_lw_flux_down,
        expected_sw_flux_up,
        expected_sw_flux_down,
    ) = _load_expected_data()

    expected_flux_up_lw = convert_to_3d(
        expected_lw_flux_up[rfmip_expt_id, rfmip_site, :]
    )
    expected_flux_down_lw = convert_to_3d(
        expected_lw_flux_down[rfmip_expt_id, rfmip_site, :]
    )
    expected_flux_up_sw = convert_to_3d(
        expected_sw_flux_up[rfmip_expt_id, rfmip_site, :]
    )
    expected_flux_down_sw = convert_to_3d(
        expected_sw_flux_down[rfmip_expt_id, rfmip_site, :]
    )

    atmos_state_ds.close()
    atol = 0.2
    np.testing.assert_allclose(
        _remove_halos(lw_fluxes['flux_down']),
        _remove_halos(expected_flux_down_lw),
        rtol=1e-2,
        atol=atol,
    )
    np.testing.assert_allclose(
        _remove_halos(lw_fluxes['flux_up']),
        _remove_halos(expected_flux_up_lw),
        rtol=7e-3,
        atol=atol,
    )

    rtol = 7e-3 if use_compact_lookup else 1e-3
    np.testing.assert_allclose(
        _remove_halos(sw_fluxes['flux_down']),
        _remove_halos(expected_flux_down_sw),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        _remove_halos(sw_fluxes['flux_up']),
        _remove_halos(expected_flux_up_sw),
        rtol=rtol,
        atol=atol,
    )


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
