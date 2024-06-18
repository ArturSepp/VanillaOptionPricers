
import timeit
import numpy as np

import vanilla_option_pricers.black_scholes as bsm
from enum import Enum


def test_vector_vs_loop(size: int = 1000):
    forward = np.random.uniform(1.0, 200.0, size)
    strike = np.random.uniform(1.0, 200.0, size)
    ttm = np.random.uniform(0.0, 1.0, size)
    vol = np.random.uniform(0.1, 1.0, size)
    optiontype = np.random.choice(['C', 'P'], size)
    discfactor = np.random.uniform(0.9, 1.0, size)

    def vector_pricer():
        return bsm.compute_bsm_vanilla_price_vector(forward=forward,
                                                    strike=strike,
                                                    ttm=ttm,
                                                    vol=vol,
                                                    optiontype=optiontype,
                                                    discfactor=discfactor)

    def slice_pricer():
        return bsm.compute_bsm_vanilla_slice_prices(ttm=ttm[0],
                                                    forward=forward[0],
                                                    strikes=strike,
                                                    vols=vol,
                                                    optiontypes=optiontype,
                                                    discfactor=discfactor[0])

    def spot_grid():
        return bsm.compute_bsm_forward_grid_prices(ttm=ttm[0],
                                                   forwards=forward,
                                                   strike=strike[0],
                                                   vol=vol[0],
                                                   optiontype=optiontype[0],
                                                   discfactor=discfactor[0])

    slice_pricer()
    vector_pricer()
    spot_grid()
    n = 20
    print(f"slice_pricer using numba: avg={timeit.Timer(slice_pricer).timeit(n)/n:.4f}")
    print(f"vector_pricer using vectorize: avg={timeit.Timer(vector_pricer).timeit(n)/n:.4f}")
    print(f"spot_grid using numba: avg={timeit.Timer(spot_grid).timeit(n)/n:.4f}")


class UnitTests(Enum):
    VECTOR_VS_LOOP = 1


def run_unit_test(unit_test: UnitTests):
    np.random.seed(3)
    if unit_test == UnitTests.VECTOR_VS_LOOP:
        test_vector_vs_loop(size=10000)


if __name__ == '__main__':

    unit_test = UnitTests.VECTOR_VS_LOOP

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
