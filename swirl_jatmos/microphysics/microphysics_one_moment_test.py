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

from typing import TypeAlias

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos.microphysics import microphysics_config
from swirl_jatmos.microphysics import microphysics_one_moment
from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array


class MicrophysicsOneMomentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Parameters used to generate expected values.
    self.wp = water.WaterParams(
        r_v=461.89,
        t_0=273.0,
        t_min=150.0,
        t_freeze=273.15,
        t_triple=273.16,
        t_icenuc=233.0,
        p_triple=611.7,
        exner_reference_pressure=1.013e5,
        lh_v0=2.258e6,
        lh_s0=2.592e6,
        cp_v=1859.0,
        cp_l=4219.9,
        cp_i=2050.0,
    )

  def test_autoconversion_rain(self):
    # SETUP
    q_liq = jnp.array([1e-3, 0.0])
    autoconversion_params = microphysics_config.AutoconversionParams()

    # ACTION
    dqr_dt_acnv = microphysics_one_moment.autoconversion_rain(
        q_liq, autoconversion_params
    )

    # VERIFICATION
    expected_dqr_dt_acnv = jnp.array([5e-7, 0.0])
    np.testing.assert_allclose(
        dqr_dt_acnv, expected_dqr_dt_acnv, rtol=1e-5, atol=0
    )

  def test_autoconversion_snow(self):
    # SETUP
    temperature = jnp.array([263.0, 263.0])
    rho = jnp.array([0.7, 0.7])
    q_v = jnp.array([5e-3, 5e-3])
    q_liq = jnp.array([1e-3, 0.0])
    q_c = jnp.array([1.5e-3, 0.0])
    q_ice = q_c - q_liq
    autoconversion_params = microphysics_config.AutoconversionParams()
    snow_params = microphysics_config.SnowParams()
    ice_params = microphysics_config.IceParams()

    # ACTION
    dq_s_dt_acnv = microphysics_one_moment.autoconversion_snow(
        temperature,
        rho,
        q_v,
        q_ice,
        self.wp,
        autoconversion_params,
        snow_params,
        ice_params,
    )

    # VERIFICATION
    expected_dq_s_dt_acnv = jnp.array([2.133467e-07, 0.0])

    np.testing.assert_allclose(
        dq_s_dt_acnv, expected_dq_s_dt_acnv, rtol=1e-5, atol=0
    )

  # pylint: disable=invalid-name
  def test_accretion_liq_rain(self):
    """Check the q_liq -> q_r accretion rate."""
    # SETUP
    rain_params = microphysics_config.RainParams()
    accretion_params = microphysics_config.AccretionParams()
    q_liq = jnp.array([5e-3, 1e-3, 0.0])
    q_r = jnp.array([1e-3, 0.0, 1e-3])
    rho = jnp.array([0.9, 0.9, 0.9])

    # ACTION
    S_qr = microphysics_one_moment.accretion(
        'liq', q_liq, rain_params, q_r, rho, accretion_params
    )

    # VERIFICATION
    expected_S_qr = jnp.array([2.330098e-05, 0.0, 0.0])
    np.testing.assert_allclose(S_qr, expected_S_qr, rtol=1e-5, atol=0)

  def test_accretion_ice_snow(self):
    # SETUP
    snow_params = microphysics_config.SnowParams()
    accretion_params = microphysics_config.AccretionParams()
    q_ice = jnp.array([5e-5, 1e-5, 0.0])
    q_s = jnp.array([1e-5, 0.0, 1e-5])
    rho = jnp.array([0.5, 0.5, 0.5])

    # ACTION
    S_qs = microphysics_one_moment.accretion(
        'ice', q_ice, snow_params, q_s, rho, accretion_params
    )

    # VERIFICATION
    expected_S_qs = jnp.array([1.763682e-10, 0.0, 0.0])
    np.testing.assert_allclose(S_qs, expected_S_qs, rtol=1e-5, atol=0)

  def test_accretion_liq_snow(self):
    # SETUP
    snow_params = microphysics_config.SnowParams()
    accretion_params = microphysics_config.AccretionParams()
    q_liq = jnp.array([5e-3, 1e-3, 0.0])
    q_s = jnp.array([1e-3, 0.0, 1e-3])
    rho = jnp.array([0.9, 0.9, 0.9])

    # ACTION
    S_qs = microphysics_one_moment.accretion(
        'liq', q_liq, snow_params, q_s, rho, accretion_params
    )

    # VERIFICATION
    expected_S_qs = jnp.array([3.725894e-06, 0.0, 0.0])
    np.testing.assert_allclose(S_qs, expected_S_qs, rtol=1e-5, atol=0)

  def test_accretion_ice_rain(self):
    # SETUP
    rain_params = microphysics_config.RainParams()
    accretion_params = microphysics_config.AccretionParams()
    q_ice = jnp.array([5e-3, 1e-5, 0.0])
    q_r = jnp.array([1e-3, 0.0, 1e-5])
    rho = jnp.array([0.5, 0.5, 0.5])

    # ACTION
    S_qr = microphysics_one_moment.accretion(
        'ice', q_ice, rain_params, q_r, rho, accretion_params
    )

    # VERIFICATION
    expected_S_qr = jnp.array([2.33692e-05, 0.0, 0.0])
    np.testing.assert_allclose(S_qr, expected_S_qr, rtol=1e-5, atol=0)

  # pylint: enable=invalid-name

  def test_accretion_rain_sink(self):
    # SETUP
    rain_params = microphysics_config.RainParams()
    ice_params = microphysics_config.IceParams()
    accretion_params = microphysics_config.AccretionParams()
    q_ice = jnp.array([1e-3, 0.0, 1e-3])
    q_r = jnp.array([1e-3, 1e-3, 0.0])
    rho = jnp.array([0.8, 0.8, 0.8])

    # ACTION
    rain_sink = microphysics_one_moment.accretion_rain_sink(
        q_ice, q_r, rho, rain_params, ice_params, accretion_params
    )

    # VERIFICATION
    expected_rain_sink = jnp.array([0.000113132, 0.0, 0.0])
    np.testing.assert_allclose(rain_sink, expected_rain_sink, rtol=1e-5, atol=0)

  def test_accretion_snow_rain(self):
    # SETUP
    microphysics_cfg = microphysics_config.MicrophysicsConfig()
    rain_params = microphysics_cfg.rain_params
    snow_params = microphysics_cfg.snow_params
    q_r = jnp.array([1e-3, 0.0, 1e-3])
    q_s = jnp.array([1e-5, 1e-3, 0.0])
    rho = jnp.array([0.8, 0.8, 0.8])

    # ACTION
    accretion_rate_r_to_s = microphysics_one_moment.accretion_snow_rain(
        snow_params, rain_params, q_s, q_r, rho, microphysics_cfg
    )
    accretion_rate_s_to_r = microphysics_one_moment.accretion_snow_rain(
        rain_params, snow_params, q_s, q_r, rho, microphysics_cfg
    )

    # VERIFICATION
    expected_accretion_rate_r_to_s = jnp.array([9.062462e-06, 0.0, 0.0])
    expected_accretion_rate_s_to_r = jnp.array([7.508936e-06, 0.0, 0.0])
    np.testing.assert_allclose(
        accretion_rate_r_to_s, expected_accretion_rate_r_to_s, rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        accretion_rate_s_to_r, expected_accretion_rate_s_to_r, rtol=1e-5, atol=0
    )

  def test_rain_evaporation_unsaturated(self):
    # SETUP
    pp_rain = microphysics_config.RainParams()
    temperature = jnp.array([289.0, 289.0])
    rho = jnp.array([1.0, 1.0])
    q_v = jnp.array([5e-3, 5e-3])
    q_p = jnp.array([2e-3, 0.0])

    # ACTION
    dq_t_dt_evap = microphysics_one_moment.evaporation_sublimation(
        pp_rain, temperature, rho, q_v, q_p, self.wp
    )

    # VERIFICATION
    expected_dq_t_dt_evap = jnp.array([9.385621e-06, 0.0])
    np.testing.assert_allclose(
        dq_t_dt_evap, expected_dq_t_dt_evap, rtol=1e-6, atol=0
    )

  def test_rain_evaporation_saturated(self):
    """Check that evaporation is zero in saturated conditions."""
    # SETUP
    pp_rain = microphysics_config.RainParams()
    temperature = jnp.array([263.0, 263.0])
    rho = jnp.array([1.0, 1.0])
    q_v = jnp.array([5e-3, 5e-3])
    q_p = jnp.array([2e-3, 0.0])

    # ACTION
    dq_t_dt_evap = microphysics_one_moment.evaporation_sublimation(
        pp_rain, temperature, rho, q_v, q_p, self.wp
    )

    # VERIFICATION
    expected_dq_t_dt_evap = jnp.array([0.0, 0.0])
    np.testing.assert_allclose(
        dq_t_dt_evap, expected_dq_t_dt_evap, rtol=1e-6, atol=0
    )

  def test_snow_sublimation(self):
    """Check that sublimation is computed correctly."""
    # SETUP
    snow_params = microphysics_config.SnowParams()
    temperature = jnp.array([263.0, 263.0])
    rho = jnp.array([0.7, 0.7])
    q_v = jnp.array([5e-3, 5e-3])
    q_p = jnp.array([2e-3, 0.0])

    # ACTION
    dq_t_dt_subl = microphysics_one_moment.evaporation_sublimation(
        snow_params, temperature, rho, q_v, q_p, self.wp
    )

    # VERIFICATION
    expected_dq_t_dt_subl = jnp.array([-2.358309e-05, 0.0])
    np.testing.assert_allclose(
        dq_t_dt_subl, expected_dq_t_dt_subl, rtol=1e-5, atol=0
    )

  def test_snow_melt(self):
    """Check that snow melting is computed correctly."""
    # SETUP
    snow_params = microphysics_config.SnowParams()
    temperature = jnp.array([263.0, 274.0, 283.0])
    rho = jnp.array([0.7, 0.7, 0.7])
    q_s = jnp.array([1e-4, 1e-4, 0.0])

    # ACTION
    dq_r_dt_melt = microphysics_one_moment.snow_melt(
        temperature, rho, q_s, self.wp, snow_params
    )

    # VERIFICATION
    expected_dq_r_dt_melt = jnp.array([0.0, 4.1981325e-6, 0.0])
    np.testing.assert_allclose(
        dq_r_dt_melt, expected_dq_r_dt_melt, rtol=1e-5, atol=0
    )

  def test_terminal_velocity_rain(self):
    # SETUP
    pp = microphysics_config.RainParams()
    q_r = jnp.array([1e-3, 0.0])
    rho = jnp.array([0.7, 0.7])

    # ACTION
    v_terminal = microphysics_one_moment._terminal_velocity_power_law(
        pp, rho, q_r
    )

    # VERIFICATION
    expected_v_terminal = jnp.array([7.219745, 0.0])
    np.testing.assert_allclose(
        v_terminal, expected_v_terminal, rtol=1e-6, atol=0
    )

  def test_terminal_velocity_snow(self):
    # SETUP
    pp = microphysics_config.SnowParams()
    q_s = jnp.array([1e-3, 0.0])
    rho = jnp.array([0.7, 0.7])

    # ACTION
    v_terminal = microphysics_one_moment._terminal_velocity_power_law(
        pp, rho, q_s
    )

    # VERIFICATION
    expected_v_terminal = jnp.array([0.87173, 0.0])
    np.testing.assert_allclose(
        v_terminal, expected_v_terminal, rtol=1e-6, atol=0
    )

  def test_cloud_particle_effective_radius(self):
    # SETUP
    rho = jnp.array([1.0, 1.0, 1.0])
    q_c = jnp.array([1e-3, 5e-3, 0.0])

    # ACTION
    r_eff_liq = microphysics_one_moment.cloud_particle_effective_radius(
        rho, q_c, phase='liq'
    )
    r_eff_ice = microphysics_one_moment.cloud_particle_effective_radius(
        rho, q_c, phase='ice'
    )

    # VERIFICATION
    expected_r_eff_liq = jnp.array([1.439706e-5, 2.4618626e-5, 0.0])
    expected_r_eff_ice = jnp.array([1.813916e-5, 3.101752e-5, 0.0])
    np.testing.assert_allclose(r_eff_liq, expected_r_eff_liq, rtol=1e-6, atol=0)
    np.testing.assert_allclose(r_eff_ice, expected_r_eff_ice, rtol=1e-6, atol=0)


if __name__ == '__main__':
  absltest.main()
